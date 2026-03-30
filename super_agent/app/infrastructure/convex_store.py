from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from convex import ConvexClient


class ConvexAgentStore:
    """Convex-backed research memory, heartbeat config (topics, cursor, persona), and tasks."""

    def __init__(self, url: str) -> None:
        self._client = ConvexClient(url.strip())
        self._research_seeded: bool = False

    def append_entry(self, title: str, body: str) -> None:
        ts = int(datetime.now(UTC).timestamp() * 1000)
        self._client.mutation(
            "memory:append",
            {"title": title, "body": body.strip(), "createdAt": ts},
        )

    def read_concatenated(self, max_chars: int = 8000) -> str:
        rows: list[dict[str, Any]] = list(
            self._client.query("memory:listAsc", {"limit": 500}) or [],
        )
        parts: list[str] = []
        for r in rows:
            created = r.get("createdAt")
            if isinstance(created, (int, float)):
                ts_str = datetime.fromtimestamp(
                    created / 1000.0,
                    tz=UTC,
                ).strftime("%Y-%m-%d %H:%M UTC")
            else:
                ts_str = "?"
            title = r.get("title") or ""
            body = r.get("body") or ""
            parts.append(f"## {title}\n*{ts_str}*\n\n{body}\n")
        full = "\n".join(parts).strip()
        if len(full) <= max_chars:
            return full
        return full[-max_chars:]

    def clear_all(self) -> None:
        self._client.mutation("memory:clearAll", {})

    def create_task(self, kind: str, detail: str | None = None) -> str:
        tid = self._client.mutation(
            "tasks:create",
            {"kind": kind, "detail": detail},
        )
        return str(tid)

    def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        args: dict[str, Any] = {"id": task_id, "status": status}
        if error is not None:
            args["error"] = error
        self._client.mutation("tasks:setStatus", args)

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._client.query("tasks:listRecent", {"limit": limit}) or [])

    def get_research_config(self) -> dict[str, Any]:
        raw = self._client.query("researchConfig:get", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def ensure_seeded_from_disk(self, data_dir: Path) -> None:
        """If Convex has no research row, copy topics / cursor / persona from local files once."""
        if self._research_seeded:
            return
        cfg = self.get_research_config()
        if cfg.get("initialized"):
            self._research_seeded = True
            return

        from super_agent.app.services import research_loop as _rl

        topics = _rl.read_heartbeat_topics_file(data_dir)
        cur = _rl.read_cursor_file(data_dir)
        persona = _rl.read_persona_file(data_dir)
        total_runs = int(cur.get("runs", 0) or 0)
        if not topics and not persona.strip() and total_runs == 0 and not cur.get("last_run"):
            self._research_seeded = True
            return

        last_run = cur.get("last_run")
        self._client.mutation(
            "researchConfig:seed",
            {
                "topics": topics[:20],
                "cursorIndex": int(cur.get("index", 0) or 0),
                "lastRun": last_run if isinstance(last_run, str) else None,
                "totalRuns": total_runs,
                "persona": persona,
            },
        )
        self._research_seeded = True

    def set_heartbeat_topics(self, topics: list[str]) -> None:
        capped = [t.strip() for t in topics if t.strip()][:20]
        self._client.mutation("researchConfig:setTopics", {"topics": capped})
        self._research_seeded = True

    def append_persona_block(self, block: str) -> None:
        self._client.mutation("researchConfig:appendPersona", {"block": block.strip()})
        self._research_seeded = True

    def record_research_run(self, next_cursor_index: int, last_run_iso: str) -> None:
        self._client.mutation(
            "researchConfig:recordRun",
            {
                "nextCursorIndex": next_cursor_index,
                "lastRunIso": last_run_iso,
            },
        )

    # ── HDC model weights ─────────────────────────────────────────────────────

    def save_model_weights(
        self,
        dim: int,
        context_size: int,
        assoc_count: int,
        assoc_memory_b64: str,
        vocab_labels: str,
        training_tokens: int,
        training_docs: int,
        last_trained: str | None,
        created_at: str,
    ) -> None:
        """Persist HDC model weights to Convex (survives Vercel cold starts)."""
        args: dict[str, Any] = {
            "dim": dim,
            "contextSize": context_size,
            "assocCount": assoc_count,
            "assocMemoryB64": assoc_memory_b64,
            "vocabLabels": vocab_labels,
            "trainingTokens": training_tokens,
            "trainingDocs": training_docs,
            "createdAt": created_at,
        }
        if last_trained:
            args["lastTrained"] = last_trained
        self._client.mutation("hdcModel:saveWeights", args)

    def load_model_weights(self) -> dict[str, Any] | None:
        """Retrieve the persisted HDC model weights from Convex."""
        result = self._client.query("hdcModel:loadWeights", {})
        return dict(result) if isinstance(result, dict) else None
