from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_TURNS_IN_CONTEXT = 20  # turns kept for LLM context window
_MAX_TURNS_ON_DISK = 200   # turns written to JSON file


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    text: str
    route: str = ""
    grounded: bool = False
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class Session:
    session_id: str
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=_MAX_TURNS_ON_DISK))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def history_for_llm(self) -> list[dict[str, str]]:
        """Last N turns in {role, text} format for Gemini multi-turn contents."""
        recent = list(self.turns)[-_MAX_TURNS_IN_CONTEXT:]
        out: list[dict[str, str]] = []
        for t in recent:
            role = "model" if t.role == "assistant" else "user"
            out.append({"role": role, "text": t.text})
        return out

    def add(self, role: str, text: str, route: str = "", grounded: bool = False) -> None:
        self.turns.append(Turn(role=role, text=text, route=route, grounded=grounded))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turns": [asdict(t) for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        sess = cls(session_id=d["session_id"], created_at=d.get("created_at", ""))
        for t in d.get("turns", []):
            sess.turns.append(Turn(**t))
        return sess


class SessionStore:
    """
    In-memory sessions, optionally persisted to data/sessions/<id>.json.

    Thread-safe. One session per browser tab / user context.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = sessions_dir
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = self._load_from_disk(session_id)
            return self._sessions[session_id]

    def save(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                path = self._dir / f"{session_id}.json"
                path.write_text(json.dumps(sess.to_dict(), indent=2), encoding="utf-8")
            except OSError as e:
                logger.warning("Session save failed %s: %s", session_id, e)

    def _load_from_disk(self, session_id: str) -> Session:
        path = self._dir / f"{session_id}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return Session.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Session load failed %s: %s", session_id, e)
        return Session(session_id=session_id)

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())
