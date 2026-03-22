from __future__ import annotations

import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import sympy as sp

from super_agent.app.domain.math_schemas import SymCodeRequest, SymCodeResult

# Gemini sometimes returns 200 with placeholder text instead of math; block as SymPy output.
_BOGUS_RESULT_RE = re.compile(
    r"(API_|_ERROR_|EXHAUSTED|RESOURCE_EXHAUSTED|NOT_FOUND|PLACEHOLDER|INSTEAD_OF|"
    r"RATE_LIMIT|QUOTA_|UNKNOWN_ERROR|GENERATION_FAILED)",
    re.I,
)


def llm_output_suspicious_for_symcode(raw: str) -> bool:
    """True when the model returned error-like prose instead of a Python code block."""
    t = raw.strip()
    if not t:
        return True
    if "```" in t:
        return False
    return bool(_BOGUS_RESULT_RE.search(t))


def _bogus_symbolic_result(result: object) -> str | None:
    """Return error message if `result` looks like an LLM/API error token, not math."""
    r = repr(result)
    if _BOGUS_RESULT_RE.search(r):
        return "rejected: SymPy result looks like an API error token, not mathematics"
    if isinstance(result, str) and _BOGUS_RESULT_RE.search(result):
        return "rejected: string result looks like an API error token"
    if hasattr(result, "name") and isinstance(getattr(result, "name"), str):
        name = getattr(result, "name")
        if _BOGUS_RESULT_RE.search(name):
            return "rejected: symbol name looks like an API error token"
    return None


def _validate_ast(source: str) -> None:
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ("sympy", "math"):
                    raise ValueError(f"disallowed import: {root}")
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in ("sympy", "math"):
                raise ValueError(f"disallowed import from: {mod}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("open", "exec", "eval", "__import__"):
                raise ValueError("disallowed call")


_SAFE_BUILTINS: dict[str, object] = {
    "__import__": __import__,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "slice": slice,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "print": print,
}


def run_symcode(req: SymCodeRequest) -> SymCodeResult:
    """
    Execute SymPy script with `result` bound in a restricted namespace.

    The script should set `result` to a SymPy object.
    """
    try:
        _validate_ast(req.source)
    except Exception as e:
        return SymCodeResult(ok=False, error=f"ast validation: {e}")

    namespace: dict[str, object] = {
        "__builtins__": _SAFE_BUILTINS,
        "sp": sp,
        "sympy": sp,
    }
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(compile(req.source, "<symcode>", "exec"), namespace, namespace)
        result = namespace.get("result")
        if result is None:
            return SymCodeResult(
                ok=False,
                stdout=out_buf.getvalue(),
                error="script must set variable `result`",
            )
        bogus = _bogus_symbolic_result(result)
        if bogus:
            return SymCodeResult(
                ok=False,
                stdout=out_buf.getvalue(),
                error=bogus,
            )
        simplified = sp.simplify(result)
        bogus2 = _bogus_symbolic_result(simplified)
        if bogus2:
            return SymCodeResult(
                ok=False,
                stdout=out_buf.getvalue(),
                error=bogus2 + " (after simplify)",
            )
        meta: dict[str, object] = {}
        if req.verify_with_diff and hasattr(result, "diff"):
            meta["note"] = "has .diff(); add domain-specific checks in caller"
        return SymCodeResult(
            ok=True,
            stdout=out_buf.getvalue(),
            result_repr=repr(result),
            simplified=repr(simplified),
            metadata=meta,
        )
    except Exception:
        return SymCodeResult(
            ok=False,
            stdout=out_buf.getvalue(),
            error=traceback.format_exc() + err_buf.getvalue(),
        )
