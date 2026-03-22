from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from super_agent.app.core.gemini_client import GeminiClient
from super_agent.app.services.workspace_context import load_soul

logger = logging.getLogger(__name__)

_CURSOR_FILE = "heartbeat_cursor.json"


def read_heartbeat_topics(data_dir: Path) -> list[str]:
    path = data_dir / "HEARTBEAT.md"
    if not path.is_file():
        return []
    topics: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith(("-", "*")):
            topics.append(ln.lstrip("-* ").strip())
        else:
            topics.append(ln)
    return topics[:20]


def _read_cursor(data_dir: Path) -> dict[str, object]:
    path = data_dir / _CURSOR_FILE
    if not path.is_file():
        return {"index": 0, "last_run": None, "runs": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"index": 0, "last_run": None, "runs": 0}


def _write_cursor(data_dir: Path, index: int, ts: str) -> None:
    cursor = _read_cursor(data_dir)
    cursor["index"] = index
    cursor["last_run"] = ts
    cursor["runs"] = int(cursor.get("runs", 0)) + 1  # type: ignore[arg-type]
    (data_dir / _CURSOR_FILE).write_text(json.dumps(cursor, indent=2), encoding="utf-8")


def heartbeat_status(data_dir: Path) -> dict[str, object]:
    cursor = _read_cursor(data_dir)
    topics = read_heartbeat_topics(data_dir)
    idx = int(cursor.get("index", 0))
    return {
        "last_run": cursor.get("last_run"),
        "total_runs": cursor.get("runs", 0),
        "topic_count": len(topics),
        "next_topic_index": idx % len(topics) if topics else 0,
        "next_topic": topics[idx % len(topics)] if topics else None,
        "topics": topics,
    }


def read_persona(data_dir: Path) -> str:
    path = data_dir / "PERSONA.md"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def append_persona(data_dir: Path, persona_update: str) -> None:
    path = data_dir / "PERSONA.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + "\n\n" + persona_update.strip(), encoding="utf-8")
    else:
        path.write_text(f"# User Persona\n\n{persona_update.strip()}", encoding="utf-8")


def append_memory(data_dir: Path, title: str, body: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    path = data_dir / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n## {title}\n*{ts}*\n\n{body.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(f"# Agent Memory\n{block}", encoding="utf-8")


def read_memory(data_dir: Path, max_chars: int = 8000) -> str:
    path = data_dir / "MEMORY.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[-max_chars:] if len(text) > max_chars else text


def clear_memory(data_dir: Path) -> None:
    path = data_dir / "MEMORY.md"
    if path.is_file():
        path.write_text("# Agent Memory\n\n*Cleared.*\n", encoding="utf-8")


def run_proactive_research(gemini: GeminiClient, data_dir: Path) -> str:
    """
    Performs proactive research. Each call advances through the HEARTBEAT.md
    topic list so every topic gets researched in rotation.
    If no topics, falls back to persona-driven research.
    """
    if not gemini.enabled:
        return "gemini disabled — research skipped"

    soul = load_soul(data_dir, max_chars=800)
    topics = read_heartbeat_topics(data_dir)
    persona_text = read_persona(data_dir)
    ts = datetime.now(UTC).isoformat()

    if topics:
        cursor = _read_cursor(data_dir)
        idx = int(cursor.get("index", 0)) % len(topics)
        topic = topics[idx]
        next_idx = (idx + 1) % len(topics)

        context_prefix = f"Considering the user's persona:\n{persona_text}\n\n" if persona_text else ""
        prompt = (
            f"{f'Directives:{chr(10)}{soul}{chr(10)}{chr(10)}' if soul else ''}"
            f"{context_prefix}"
            f"Research and write a concise summary (5–10 bullet points) about:\n\n**{topic}**\n\n"
            "Focus on practical, actionable insights. Include specific tools, papers, or developments "
            "that are relevant to someone building AI systems in 2026."
        )

        report = gemini.generate_with_search(prompt)[0]
        append_memory(data_dir, f"Research: {topic}", report)
        _write_cursor(data_dir, next_idx, ts)
        logger.info("heartbeat: researched topic[%d/%d]: %s", idx + 1, len(topics), topic)
        return f"researched ({idx + 1}/{len(topics)}): {topic}"

    elif persona_text:
        prompt = (
            f"{f'Directives:{chr(10)}{soul}{chr(10)}{chr(10)}' if soul else ''}"
            f"Research recent advancements relevant to this persona:\n{persona_text}\n\n"
            "Be concise; bullet points."
        )
        report = gemini.generate_with_search(prompt)[0]
        append_memory(data_dir, "Persona-driven Research", report)
        cursor = _read_cursor(data_dir)
        _write_cursor(data_dir, 0, ts)
        logger.info("heartbeat: persona-driven research complete")
        return "persona-driven research stored"

    return "no topics in HEARTBEAT.md and no persona — nothing researched"
