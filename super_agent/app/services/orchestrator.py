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
    from super_agent.app.domain.chat_schemas import FullStackImproveResult, ImproveResult

_CURRENT_EVENTS_RE = re.compile(
    r"\b(current|today|now|latest|recent|2025|2026|news|happening|"
    r"this (year|month|week|day)|right now|what is going on|what happened)\b",
    re.I,
)

_IMPROVE_VERBS_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+you\s+to\s+)?"
    r"(?:add|implement|create|build|write|fix|refactor|extend|integrate|"
    r"update|improve|enable|remove|delete|make\s+the)\b",
    re.I,
)

_CODE_NOUN_RE = re.compile(
    r"\b(?:endpoint|route|api|feature|module|function|class|method|codebase|"
    r"the\s+code|the\s+agent|the\s+system|capability|to\s+the\s+code|"
    r"in\s+the\s+code|session|memory|search|heartbeat|sica|improvement|"
    r"rate\s+limit|middleware|authentication|websocket|pipeline|loop|service)\b",
    re.I,
)


def _today_str() -> str:
    return datetime.now(UTC).strftime("%A, %B %d, %Y")


def _needs_search(text: str) -> bool:
    return bool(_CURRENT_EVENTS_RE.search(text))


def _is_improve_intent(text: str) -> bool:
    return bool(_IMPROVE_VERBS_RE.match(text)) and bool(_CODE_NOUN_RE.search(text))

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

    def run(self, message: str, session_id: str | None = None, auto_improve: bool = False) -> ChatTurnResult:
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

        if auto_improve and self._gemini.enabled and _is_improve_intent(message):
            result = self._run_with_improve(message, preamble, hdc_hint, meta, matched_fp, history)
        elif intent == SymbolicIntent.SYMBOLIC:
            result = self._run_symbolic(message, preamble, hdc_hint, meta, matched_fp, history)
        else:
            result = self._run_neural(message, preamble, hdc_hint, intent.value, meta, matched_fp, history)

        if session is not None:
            session.add("user", message)
            session.add("assistant", result.answer, route=result.route, grounded=result.metadata.get("grounded", False))  # type: ignore[arg-type]
            self._sessions.save(session_id)  # type: ignore[arg-type]

        return result

    def _run_with_improve(
        self,
        message: str,
        preamble: str,
        hdc_hint: str,
        meta: dict[str, object],
        matched_fp: str | None,
        history: list[dict[str, str]] | None,
    ) -> ChatTurnResult:
        """Run the self-improvement pipeline and narrate the result as a chat turn."""
        sys_instr = self._system_instruction(preamble)
        sim = meta.get("hdc_similarity")
        hdc_sim = float(sim) if isinstance(sim, (int, float)) else None

        improve_result = self.improve_self(message)

        if improve_result.ok:
            explain_prompt = (
                f"You just applied a code improvement to `{improve_result.target_file}`.\n"
                f"Instruction: {message}\n"
                "Write a concise 2–3 sentence explanation for the user: what was changed and why it matters."
            )
            explanation = self._gemini.generate_text(
                explain_prompt, model=self._settings.gemini_model_flash,
                history=history, system_instruction=sys_instr,
            )
            answer = explanation
        else:
            answer = (
                f"I attempted to apply that change but encountered an issue:\n\n"
                f"> {improve_result.error}\n\n"
                "You can try again with a more specific instruction or target file."
            )

        meta["improve_result"] = improve_result.model_dump()
        return ChatTurnResult(
            route="neural", intent="improve", answer=answer,
            hdc_similarity=hdc_sim, hdc_matched_task=matched_fp,
            context_snippet=preamble[:300], metadata=meta,
        )

    def improve_full_stack(
        self,
        instruction: str,
        target_file: str | None = None,
    ) -> "FullStackImproveResult":
        """
        Full-stack improvement pipeline:
        1. Improve the backend Python code (improve_self)
        2. Update nextjstester/lib/agent-api.ts to expose any new endpoints
        3. Update the AgentTester.tsx endpoint list + suggestions
        """
        from datetime import UTC, datetime
        from super_agent.app.domain.chat_schemas import FullStackImproveResult

        ts = datetime.now(UTC).isoformat()
        backend = self.improve_self(instruction, target_file)
        if not backend.ok:
            return FullStackImproveResult(
                ok=False, instruction=instruction, backend=backend, timestamp=ts
            )

        frontend_api = self._improve_frontend_api(instruction, backend)
        frontend_ui = self._improve_frontend_ui(instruction, backend)

        return FullStackImproveResult(
            ok=True,
            instruction=instruction,
            backend=backend,
            frontend_api=frontend_api,
            frontend_ui=frontend_ui,
            timestamp=ts,
        )

    def _improve_frontend_api(
        self,
        instruction: str,
        backend: "ImproveResult",
    ) -> "ImproveResult":
        """Update nextjstester/lib/agent-api.ts to add functions for new backend endpoints."""
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        from super_agent.app.domain.chat_schemas import ImproveResult
        from super_agent.app.services.sica_loop import write_non_python_file

        repo_root = Path(__file__).resolve().parents[3]
        ts = datetime.now(UTC).isoformat()
        target = "nextjstester/lib/agent-api.ts"
        target_path = repo_root / target

        if not target_path.exists():
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                error="agent-api.ts not found", timestamp=ts,
            )

        old_code = target_path.read_text(encoding="utf-8")
        prompt = (
            "You are updating a Next.js TypeScript API client file after a backend improvement.\n\n"
            f"Instruction applied to backend: {instruction}\n\n"
            f"Backend file modified: `{backend.target_file}`\n"
            f"Backend new code excerpt:\n```python\n{backend.new_code[:2000]}\n```\n\n"
            f"Current `{target}`:\n```typescript\n{old_code}\n```\n\n"
            "Task: If the backend change introduced new API endpoints or modified existing ones, "
            "update this file to add or update the corresponding exported async functions.\n"
            "If there is nothing to add (no new endpoints), return the file unchanged.\n\n"
            "Rules:\n"
            "- Output ONLY one ```typescript ... ``` block with the complete file\n"
            "- Preserve all existing functions exactly unless updating them\n"
            "- Match existing code style (async/await, error handling, return types)\n"
            "- Do not truncate with placeholders — output the full file"
        )

        raw = self._gemini.generate_text(prompt, model=self._settings.gemini_model_pro)
        code_m = re.search(r"```(?:typescript|ts)?\s*([\s\S]*?)```", raw)
        if not code_m:
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                old_code=old_code, error="Gemini did not return a TypeScript block", timestamp=ts,
            )

        new_code = code_m.group(1).strip()
        if len(new_code) < 200 or "export async function" not in new_code:
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                old_code=old_code, new_code=new_code,
                error="Generated code appears incomplete or missing exports", timestamp=ts,
            )

        step = write_non_python_file(
            repo_root, target_path, new_code + "\n",
            f"frontend-api: {instruction[:80]}",
        )

        history_file = self._settings.data_dir / "improvements.jsonl"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        result = ImproveResult(
            ok=bool(step.get("ok", True)),
            target_file=target,
            instruction=instruction,
            old_code=old_code,
            new_code=new_code,
            ast_ok=True,
            committed=bool(step.get("committed", False)),
            commit_hash=step.get("stable_hash"),  # type: ignore[arg-type]
            error=None if step.get("ok", True) else str(step.get("error", "")),
            timestamp=ts,
        )
        with history_file.open("a", encoding="utf-8") as fh:
            fh.write(result.model_dump_json() + "\n")
        return result

    def _improve_frontend_ui(
        self,
        instruction: str,
        backend: "ImproveResult",
    ) -> "ImproveResult":
        """
        Update AgentTester.tsx: add the new endpoint to the Status tab's endpoint list
        and insert a relevant suggestion chip if applicable.
        """
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        from super_agent.app.domain.chat_schemas import ImproveResult
        from super_agent.app.services.sica_loop import write_non_python_file

        repo_root = Path(__file__).resolve().parents[3]
        ts = datetime.now(UTC).isoformat()
        target = "nextjstester/app/components/AgentTester.tsx"
        target_path = repo_root / target

        if not target_path.exists():
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                error="AgentTester.tsx not found", timestamp=ts,
            )

        old_code = target_path.read_text(encoding="utf-8")
        prompt = (
            "You are updating a Next.js React component (AgentTester.tsx) to expose a new backend feature.\n\n"
            f"Instruction applied: {instruction}\n"
            f"Backend file changed: `{backend.target_file}`\n\n"
            f"Current component file:\n```tsx\n{old_code[:8000]}\n```\n\n"
            "Task: Make MINIMAL, TARGETED changes only:\n"
            "1. If a new API endpoint was added, add it to the endpoints array in the Status tab "
            '   (look for the array of ["METHOD", "/path"] pairs).\n'
            "2. If the feature is user-facing and significant, add ONE new suggestion chip to the "
            "   SUGGESTIONS array at the top of the component.\n"
            "3. No other changes — do not restructure, rename, or reformat anything.\n\n"
            "Rules:\n"
            "- Output ONLY one ```tsx ... ``` block with the complete file\n"
            "- Preserve every line that is not being changed exactly as-is\n"
            "- Do not truncate with placeholders — output the full file\n"
            "- If there is nothing meaningful to add, return the file unchanged"
        )

        raw = self._gemini.generate_text(prompt, model=self._settings.gemini_model_pro)
        code_m = re.search(r"```(?:tsx|typescript|ts)?\s*([\s\S]*?)```", raw)
        if not code_m:
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                old_code=old_code, error="Gemini did not return a TSX block", timestamp=ts,
            )

        new_code = code_m.group(1).strip()
        # Sanity: must be a substantial React component
        if len(new_code) < 1000 or "export default function AgentTester" not in new_code:
            return ImproveResult(
                ok=False, target_file=target, instruction=instruction,
                old_code=old_code, new_code=new_code,
                error="Generated UI code too short or missing AgentTester export", timestamp=ts,
            )

        step = write_non_python_file(
            repo_root, target_path, new_code + "\n",
            f"frontend-ui: {instruction[:80]}",
        )

        history_file = self._settings.data_dir / "improvements.jsonl"
        result = ImproveResult(
            ok=bool(step.get("ok", True)),
            target_file=target,
            instruction=instruction,
            old_code=old_code,
            new_code=new_code,
            ast_ok=True,
            committed=bool(step.get("committed", False)),
            commit_hash=step.get("stable_hash"),  # type: ignore[arg-type]
            error=None if step.get("ok", True) else str(step.get("error", "")),
            timestamp=ts,
        )
        with history_file.open("a", encoding="utf-8") as fh:
            fh.write(result.model_dump_json() + "\n")
        return result

    # ── two-phase self-improvement (localize → implement) ────────────────────

    def _localize_files(self, instruction: str, snapshot: str) -> list[dict[str, str]]:
        """
        Phase 1 (localize): ask Gemini which files to create/modify.
        Returns a list like:
          [{"file": "super_agent/app/api/routes.py", "action": "modify", "reason": "..."}]
        Uses the SICA/SWE-Adept pattern: localization agent → resolution agent.
        """
        import json as _json
        prompt = (
            "You are a code localization agent for a FastAPI + Next.js monorepo.\n\n"
            f"Codebase overview:\n{snapshot}\n\n"
            f"User instruction: {instruction}\n\n"
            "Identify ALL files that must be CREATED or MODIFIED to fully implement this.\n"
            "Output ONLY a JSON array — no markdown fences, no explanation:\n"
            "[\n"
            '  {"file": "super_agent/app/api/routes.py", "action": "modify", "reason": "add new endpoint"}\n'
            "]\n\n"
            "Rules:\n"
            "- 1–4 files maximum\n"
            "- Exact repo-relative paths (e.g. super_agent/app/services/orchestrator.py)\n"
            "- Dependency order: list files imported by others LAST\n"
            "- Only files directly needed — no unrelated tests or docs"
        )
        raw = self._gemini.generate_text(prompt)
        # strip markdown fences if the model adds them
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        m = re.search(r"\[[\s\S]*?\]", cleaned)
        if not m:
            return []
        try:
            candidates = _json.loads(m.group(0))
            return [c for c in candidates if isinstance(c, dict) and c.get("file")]
        except Exception:
            return []

    def _generate_file_code(
        self,
        file_path: str,
        instruction: str,
        all_targets: list[dict[str, str]],
        snapshot: str,
        existing_code: str,
    ) -> str | None:
        """
        Phase 2 (implement): generate the complete new content of a single file.
        Returns the raw code string, or None if generation failed.
        """
        is_python = file_path.endswith(".py")
        lang = "python" if is_python else ("tsx" if file_path.endswith(".tsx") else "typescript")
        plan_summary = "\n".join(
            f"- {t['file']}: {t.get('reason', '')}" for t in all_targets
        )
        action_note = "CREATING (new file)" if not existing_code else "MODIFYING (existing file)"
        prompt = (
            f"You are implementing a change across multiple files in the super-agent project.\n\n"
            f"Architecture overview:\n{snapshot}\n\n"
            f"All files being changed in this operation:\n{plan_summary}\n\n"
            f"Now generate the content for: `{file_path}` ({action_note})\n"
            f"User instruction: {instruction}\n\n"
            + (f"Current content:\n```{lang}\n{existing_code}\n```\n\n" if existing_code else "")
            + f"Generate the COMPLETE new content as a single ```{lang} ... ``` block.\n"
            "Rules:\n"
            "- Output ONLY one code block — the full file content\n"
            "- Preserve ALL existing functionality unless told to remove it\n"
            "- Match the exact code style of the existing file\n"
            "- Do NOT truncate with '# ... rest unchanged' or similar placeholders\n"
            "- Do NOT add explanatory comments about what you changed"
        )
        raw = self._gemini.generate_text(prompt, model=self._settings.gemini_model_pro)
        pattern = rf"```(?:{lang}|python|typescript|tsx|ts)?\s*([\s\S]*?)```"
        m = re.search(pattern, raw)
        return m.group(1).strip() if m else None

    def improve_self(self, instruction: str, target_file: str | None = None) -> "ImproveResult":
        """
        Two-phase self-improvement pipeline (SWE-Adept / SICA pattern):
          Phase 1 — Localize: identify ALL files to change
          Phase 2 — Implement: generate + write + commit each file
        Returns an ImproveResult whose file_changes list carries every modified file.
        """
        import json as _json
        from datetime import UTC, datetime
        from pathlib import Path

        from super_agent.app.domain.chat_schemas import FileChange, ImproveResult
        from super_agent.app.infrastructure.ast_liveness import parse_src_ok
        from super_agent.app.services.codebase_scanner import refresh_codebase_md
        from super_agent.app.services.sica_loop import sica_step, write_non_python_file
        from super_agent.app.services.workspace_context import load_codebase_snapshot

        repo_root = Path(__file__).resolve().parents[3]
        ts = datetime.now(UTC).isoformat()

        _ALLOWED_PREFIXES = ("super_agent/", "tests/", "nextjstester/")

        snapshot = load_codebase_snapshot(self._settings.data_dir, max_chars=3500)

        # ── Phase 1: localize ──────────────────────────────────────────────────
        if target_file:
            targets = [{"file": target_file.strip(), "action": "modify", "reason": instruction}]
        else:
            targets = self._localize_files(instruction, snapshot)

        if not targets:
            return ImproveResult(
                ok=False, target_file="unknown", instruction=instruction,
                error="Localization failed: Gemini could not identify which files to change. "
                      "Try being more specific (e.g. 'add X to routes.py').",
                timestamp=ts,
            )

        # Safety check — only allow known paths
        blocked = [t["file"] for t in targets if not any(t["file"].startswith(p) for p in _ALLOWED_PREFIXES)]
        if blocked:
            return ImproveResult(
                ok=False, target_file=blocked[0], instruction=instruction,
                error=f"Safety: blocked modification to {blocked}. Only {_ALLOWED_PREFIXES} are allowed.",
                timestamp=ts,
            )

        # ── Phase 2: implement ─────────────────────────────────────────────────
        file_changes: list[FileChange] = []
        primary: FileChange | None = None

        for target in targets:
            fp = target["file"]
            target_path = repo_root / fp
            is_python = fp.endswith(".py")

            existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            new_code = self._generate_file_code(fp, instruction, targets, snapshot, existing)

            if not new_code:
                fc = FileChange(
                    file=fp, action=target.get("action", "modify"),
                    reason=target.get("reason", ""), old_code=existing,
                    error="Code generation failed (no code block returned)",
                )
                file_changes.append(fc)
                continue

            if is_python:
                ast_ok, ast_err = parse_src_ok(new_code)
                if not ast_ok:
                    fc = FileChange(
                        file=fp, action=target.get("action", "modify"),
                        reason=target.get("reason", ""), old_code=existing,
                        new_code=new_code, ast_ok=False,
                        error=f"AST check failed: {ast_err}",
                    )
                    file_changes.append(fc)
                    continue
                step = sica_step(repo_root, target_path, new_code + "\n", f"improve: {instruction[:60]}")
            else:
                # For TypeScript/TSX/Markdown — basic sanity, then write+commit
                if len(new_code) < 50:
                    fc = FileChange(
                        file=fp, action=target.get("action", "modify"),
                        reason=target.get("reason", ""), old_code=existing,
                        new_code=new_code,
                        error="Generated content too short, skipped.",
                    )
                    file_changes.append(fc)
                    continue
                step = write_non_python_file(repo_root, target_path, new_code + "\n", f"improve: {instruction[:60]}")

            fc = FileChange(
                file=fp,
                action=target.get("action", "modify"),
                reason=target.get("reason", ""),
                old_code=existing,
                new_code=new_code,
                ast_ok=is_python,
                committed=bool(step.get("committed", False)),
                commit_hash=step.get("stable_hash"),  # type: ignore[arg-type]
                error=None if step.get("ok", True) else str(step.get("error", "")),
            )
            file_changes.append(fc)
            if primary is None and fc.error is None:
                primary = fc

        refresh_codebase_md(self._settings.data_dir)

        # Primary is the first successfully changed file
        if primary is None:
            primary = file_changes[0] if file_changes else FileChange(
                file="unknown", error="No files were successfully changed."
            )

        result = ImproveResult(
            ok=primary.error is None,
            target_file=primary.file,
            instruction=instruction,
            old_code=primary.old_code,
            new_code=primary.new_code,
            ast_ok=primary.ast_ok,
            committed=primary.committed,
            commit_hash=primary.commit_hash,
            error=primary.error,
            timestamp=ts,
            file_changes=file_changes,
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

        if _needs_search(message) and self._settings.use_google_search_grounding():
            text, grounded = self._gemini.generate_with_search(
                prompt, model=self._settings.gemini_model_flash,
                history=history, system_instruction=sys_instr,
            )
        elif _needs_search(message):
            # Vercel / short-timeout hosts: search tool + AFC often exceeds serverless limits → "failed to fetch"
            meta["grounded"] = False
            meta["search_skipped"] = "vercel_or_config"
            extra = (
                "\n\n[Runtime: live Google Search is disabled on this deployment to avoid timeouts. "
                "Answer from training knowledge; state uncertainty for fast-changing topics "
                "(politics, wars, prices, sports scores).]"
            )
            text = self._gemini.generate_text(
                prompt + extra,
                model=self._settings.gemini_model_flash,
                history=history, system_instruction=sys_instr,
            )
            grounded = False
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
