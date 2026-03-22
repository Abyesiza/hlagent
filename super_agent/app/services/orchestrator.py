from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.domain.math_schemas import SymCodeRequest, SymCodeResult, SymbolicIntent
from super_agent.app.infrastructure.hdc_memory_store import HDCMemoryStore
from super_agent.app.infrastructure.intent_router import classify_intent
from super_agent.app.infrastructure.sympy_runner import llm_output_suspicious_for_symcode, run_symcode
from super_agent.app.services.session_store import Session, SessionStore
from super_agent.app.services.workspace_context import build_system_preamble

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings
    from super_agent.app.core.gemini_client import GeminiClient

_CURRENT_EVENTS_RE = re.compile(
    r"\b(current|today|now|latest|recent|2025|2026|news|happening|"
    r"this (year|month|week|day)|right now|what is going on|what happened)\b",
    re.I,
)


def _today_str() -> str:
    return datetime.now(UTC).strftime("%A, %B %d, %Y")


def _needs_search(text: str) -> bool:
    return bool(_CURRENT_EVENTS_RE.search(text))

_SYM_CODE_PROMPT = """You are a SymPy code generator. Output ONLY one Python code block in markdown.

Rules:
- Use `import sympy as sp` (and `math` only if needed).
- Define symbols with sp.Symbol or sp.symbols.
- Set the final answer in a variable named `result` (a SymPy expression).
- `result` must be real mathematics (Expr, Matrix, etc.). Never set it to a string error message,
  API status, placeholder, or token like API_* / *_ERROR_* / EXHAUSTED — those are forbidden.
- No file I/O, no network, no input().
- Do not print unless debugging; prefer assigning to `result`.

User problem:
---
{message}
---
"""


def _extract_python(llm_text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", llm_text)
    if m:
        return m.group(1).strip()
    lines = [ln for ln in llm_text.strip().splitlines() if ln.strip()]
    if any("import sympy" in ln or "import sp" in ln.replace(" ", "") for ln in lines[:15]):
        return llm_text.strip()
    return llm_text.strip()


class SuperAgentOrchestrator:
    """
    Connects workspace context → intent → SymPy or Gemini → HDC memory.

    Each call can carry a session_id so the agent remembers prior turns.
    Current date is always injected; current-events queries use Google Search.
    """

    def __init__(
        self,
        settings: "Settings",
        gemini: "GeminiClient",
        memory: HDCMemoryStore,
        sessions: SessionStore,
    ) -> None:
        self._settings = settings
        self._gemini = gemini
        self._memory = memory
        self._sessions = sessions

    def _system_instruction(self, preamble: str) -> str:
        today = _today_str()
        base = (
            f"Today's date is {today}. You are the Super Agent — a neuro-symbolic research assistant.\n"
            "Always state the current date when relevant. Never confuse training data with live facts.\n"
            "When you do NOT have live search results, explicitly say you are drawing on training knowledge.\n"
        )
        if preamble:
            base += "\n## Directives\n" + preamble
        return base

    def run(self, message: str, session_id: str | None = None) -> ChatTurnResult:
        message = (message or "").strip()
        if not message:
            return ChatTurnResult(route="error", answer="Empty message.", intent="unknown")

        session: Session | None = None
        if session_id:
            session = self._sessions.get_or_create(session_id)

        preamble = build_system_preamble(self._settings.data_dir)
        history = session.history_for_llm() if session else None

        prior, sim, matched_fp = self._memory.retrieve(message)
        hdc_hint = ""
        if prior and sim > 0.2:
            hdc_hint = f"\n[Memory: similar prior answer (sim={sim:.3f}): {prior[:600]}]\n"

        intent = classify_intent(message)
        meta: dict[str, object] = {
            "preamble_chars": len(preamble),
            "hdc_similarity": sim,
            "hdc_matched_fp": matched_fp,
            "session_id": session_id,
            "date": _today_str(),
        }

        if intent == SymbolicIntent.SYMBOLIC:
            result = self._run_symbolic(message, preamble, hdc_hint, meta, matched_fp, history)
        else:
            result = self._run_neural(message, preamble, hdc_hint, intent.value, meta, matched_fp, history)

        if session is not None:
            session.add("user", message)
            session.add("assistant", result.answer, route=result.route, grounded=result.metadata.get("grounded", False))  # type: ignore[arg-type]
            self._sessions.save(session_id)  # type: ignore[arg-type]

        return result

    def improve_self(self, instruction: str, target_file: str | None = None) -> "ImproveResult":
        """
        Self-improvement pipeline:
        1. Ask Gemini to identify target file (if not given)
        2. Read current file content
        3. Gemini generates improved version
        4. AST check → write → git commit → refresh CODEBASE.md
        5. Log to data/improvements.jsonl
        """
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        from super_agent.app.domain.chat_schemas import ImproveResult
        from super_agent.app.infrastructure.ast_liveness import parse_src_ok
        from super_agent.app.services.codebase_scanner import refresh_codebase_md
        from super_agent.app.services.sica_loop import sica_step
        from super_agent.app.services.workspace_context import load_codebase_snapshot

        repo_root = Path(__file__).resolve().parents[3]
        ts = datetime.now(UTC).isoformat()

        _ALLOWED_PREFIXES = ("super_agent/", "tests/")

        if not target_file:
            snapshot = load_codebase_snapshot(self._settings.data_dir, max_chars=3000)
            plan_prompt = (
                "You are helping improve the super_agent FastAPI codebase.\n"
                f"Codebase summary (excerpt):\n{snapshot}\n\n"
                f"User instruction: {instruction}\n\n"
                "Identify the single best Python file to edit. "
                "Reply with ONLY the relative path from the repo root, "
                "e.g.: super_agent/app/api/routes.py\n"
                "No explanation, no markdown, just the path."
            )
            raw_path = self._gemini.generate_text(plan_prompt).strip()
            target_file = raw_path.splitlines()[0].strip().strip("`").strip()

        if not any(target_file.startswith(p) for p in _ALLOWED_PREFIXES):
            return ImproveResult(
                ok=False, target_file=target_file, instruction=instruction,
                error=f"Safety: only files under {_ALLOWED_PREFIXES} may be modified.", timestamp=ts,
            )

        target_path = repo_root / target_file
        if not target_path.exists():
            return ImproveResult(
                ok=False, target_file=target_file, instruction=instruction,
                error=f"File not found: {target_file}", timestamp=ts,
            )

        old_code = target_path.read_text(encoding="utf-8")
        snapshot = load_codebase_snapshot(self._settings.data_dir, max_chars=2000)
        improve_prompt = (
            "You are improving a Python module in the super_agent project.\n"
            f"Architecture (excerpt):\n{snapshot}\n\n"
            f"Current content of `{target_file}`:\n"
            f"```python\n{old_code}\n```\n\n"
            f"User instruction: {instruction}\n\n"
            "Generate the COMPLETE improved version as a single Python code block.\n"
            "Rules:\n"
            "- Output ONLY one ```python ... ``` block containing the full file\n"
            "- Preserve all existing functionality unless explicitly told otherwise\n"
            "- Match the existing code style exactly\n"
            "- Do not truncate or use placeholders like '# ... rest of file'"
        )
        raw = self._gemini.generate_text(improve_prompt, model=self._settings.gemini_model_pro)

        code_m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw)
        if not code_m:
            return ImproveResult(
                ok=False, target_file=target_file, instruction=instruction,
                old_code=old_code, error="Gemini did not return a code block.", timestamp=ts,
            )

        new_code = code_m.group(1).strip()
        ast_ok, ast_err = parse_src_ok(new_code)
        if not ast_ok:
            return ImproveResult(
                ok=False, target_file=target_file, instruction=instruction,
                old_code=old_code, new_code=new_code, ast_ok=False,
                error=f"AST check failed: {ast_err}", timestamp=ts,
            )

        step = sica_step(repo_root, target_path, new_code + "\n", f"improve: {instruction[:80]}")
        refresh_codebase_md(self._settings.data_dir)

        commit_hash: str | None = step.get("stable_hash") if step.get("committed") else None  # type: ignore[assignment]
        result = ImproveResult(
            ok=bool(step.get("ok", True)),
            target_file=target_file,
            instruction=instruction,
            old_code=old_code,
            new_code=new_code,
            ast_ok=True,
            committed=bool(step.get("committed", False)),
            commit_hash=commit_hash,
            error=None if step.get("ok", True) else str(step.get("error", "")),
            timestamp=ts,
        )

        history_file = self._settings.data_dir / "improvements.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with history_file.open("a", encoding="utf-8") as fh:
            fh.write(result.model_dump_json() + "\n")

        return result

    def _run_symbolic(
        self,
        message: str,
        preamble: str,
        hdc_hint: str,
        meta: dict[str, object],
        matched_fp: str | None,
        history: list[dict[str, str]] | None,
    ) -> ChatTurnResult:
        sim = meta.get("hdc_similarity")
        hdc_sim = float(sim) if isinstance(sim, (int, float)) else None
        sys_instr = self._system_instruction(preamble)

        if not self._gemini.enabled:
            return ChatTurnResult(
                route="error",
                intent="symbolic",
                answer=(
                    "Symbolic routing detected but Gemini is not configured. "
                    "Set SUPER_AGENT_GEMINI_API_KEY, or POST your script to /api/v1/sympy/run."
                ),
                sympy=None,
                hdc_similarity=hdc_sim,
                hdc_matched_task=matched_fp,
                context_snippet=preamble[:500],
                metadata=meta,
            )

        prompt = hdc_hint + "\n" + _SYM_CODE_PROMPT.format(message=message)
        raw = self._gemini.generate_text(
            prompt, model=self._settings.gemini_model_flash,
            history=history, system_instruction=sys_instr,
        )
        meta["sympy_raw_chars"] = len(raw)
        if llm_output_suspicious_for_symcode(raw):
            res = SymCodeResult(ok=False, error="LLM returned error-like text; retrying.")
        else:
            res = run_symcode(SymCodeRequest(source=_extract_python(raw)))

        if res.ok and res.simplified:
            self._memory.remember(message, res.simplified, route="symbolic")
            answer = f"Symbolic result:\n{res.simplified}"
            if res.stdout:
                answer += f"\n\nstdout:\n{res.stdout}"
            return ChatTurnResult(
                route="symbolic", intent="symbolic", answer=answer, sympy=res,
                hdc_similarity=hdc_sim, hdc_matched_task=matched_fp,
                context_snippet=preamble[:300], metadata=meta,
            )

        retry_prompt = prompt + f"\n\nPrevious attempt failed:\n{res.error or 'unknown'}\nFix the code block."
        raw2 = self._gemini.generate_text(
            retry_prompt, model=self._settings.gemini_model_flash,
            history=history, system_instruction=sys_instr,
        )
        if llm_output_suspicious_for_symcode(raw2):
            res2 = SymCodeResult(ok=False, error="LLM retry returned error-like text.")
        else:
            res2 = run_symcode(SymCodeRequest(source=_extract_python(raw2)))
        if res2.ok and res2.simplified:
            self._memory.remember(message, res2.simplified, route="symbolic")
            return ChatTurnResult(
                route="symbolic", intent="symbolic",
                answer=f"Symbolic result (after retry):\n{res2.simplified}",
                sympy=res2, hdc_similarity=hdc_sim, hdc_matched_task=matched_fp,
                context_snippet=preamble[:300], metadata={**meta, "retried": True},
            )

        fallback = self._gemini.generate_text(
            f"The symbolic executor failed for: {message}\nExplain and give the mathematical answer in plain text.",
            model=self._settings.gemini_model_pro,
            history=history, system_instruction=sys_instr,
        )
        return ChatTurnResult(
            route="neural", intent="symbolic", answer=fallback, sympy=res2,
            hdc_similarity=hdc_sim, hdc_matched_task=matched_fp,
            metadata={**meta, "retried": True, "fallback": "neural_after_sympy_fail"},
        )

    def _run_neural(
        self,
        message: str,
        preamble: str,
        hdc_hint: str,
        intent_value: str,
        meta: dict[str, object],
        matched_fp: str | None,
        history: list[dict[str, str]] | None,
    ) -> ChatTurnResult:
        sys_instr = self._system_instruction(preamble)
        prompt = hdc_hint + "\nUser:\n" + message if hdc_hint else message
        grounded = False

        if _needs_search(message):
            text, grounded = self._gemini.generate_with_search(
                prompt, model=self._settings.gemini_model_flash,
                history=history, system_instruction=sys_instr,
            )
        else:
            text = self._gemini.generate_text(
                prompt, model=self._settings.gemini_model_flash,
                history=history, system_instruction=sys_instr,
            )

        sim = meta.get("hdc_similarity")
        hdc_sim = float(sim) if isinstance(sim, (int, float)) else None
        if len(text) < 2000:
            self._memory.remember(message, text[:500], route="neural")
        meta["grounded"] = grounded
        return ChatTurnResult(
            route="neural", intent=intent_value, answer=text,
            hdc_similarity=hdc_sim, hdc_matched_task=matched_fp,
            context_snippet=preamble[:300], metadata=meta,
        )
