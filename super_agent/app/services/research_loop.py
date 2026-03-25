from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from super_agent.app.core.config import get_settings
from super_agent.app.core.gemini_client import GeminiClient
from super_agent.app.services import email_notify
from super_agent.app.services.workspace_context import load_soul

if TYPE_CHECKING:
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore

logger = logging.getLogger(__name__)

_CURSOR_FILE = "heartbeat_cursor.json"


def read_heartbeat_topics_file(data_dir: Path) -> list[str]:
    path = data_dir / "HEARTBEAT.md"
    if not path.is_file():
        return []
    topics: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # Only accept bullet-list lines (- or *); skip plain description text
        if ln.startswith(("-", "*")):
            t = ln.lstrip("-* ").strip()
            if t:
                topics.append(t)
    return topics[:20]


def read_cursor_file(data_dir: Path) -> dict[str, object]:
    path = data_dir / _CURSOR_FILE
    if not path.is_file():
        return {"index": 0, "last_run": None, "runs": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"index": 0, "last_run": None, "runs": 0}


def write_cursor_file(data_dir: Path, index: int, ts: str) -> None:
    cursor = read_cursor_file(data_dir)
    cursor["index"] = index
    cursor["last_run"] = ts
    cursor["runs"] = int(cursor.get("runs", 0)) + 1  # type: ignore[arg-type]
    (data_dir / _CURSOR_FILE).write_text(json.dumps(cursor, indent=2), encoding="utf-8")


def read_persona_file(data_dir: Path) -> str:
    path = data_dir / "PERSONA.md"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def read_heartbeat_topics(
    data_dir: Path,
    *,
    convex: "ConvexAgentStore | None" = None,
) -> list[str]:
    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        cfg = convex.get_research_config()
        return list(cfg.get("topics") or [])[:20]
    return read_heartbeat_topics_file(data_dir)


def heartbeat_status(
    data_dir: Path,
    *,
    convex: "ConvexAgentStore | None" = None,
) -> dict[str, object]:
    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        cfg = convex.get_research_config()
        topics = list(cfg.get("topics") or [])
        idx = int(cfg.get("cursorIndex", 0))
        return {
            "last_run": cfg.get("lastRun"),
            "total_runs": int(cfg.get("totalRuns") or 0),
            "topic_count": len(topics),
            "next_topic_index": idx % len(topics) if topics else 0,
            "next_topic": topics[idx % len(topics)] if topics else None,
            "topics": topics,
        }
    cursor = read_cursor_file(data_dir)
    topics = read_heartbeat_topics_file(data_dir)
    idx = int(cursor.get("index", 0))
    return {
        "last_run": cursor.get("last_run"),
        "total_runs": cursor.get("runs", 0),
        "topic_count": len(topics),
        "next_topic_index": idx % len(topics) if topics else 0,
        "next_topic": topics[idx % len(topics)] if topics else None,
        "topics": topics,
    }


def read_persona(
    data_dir: Path,
    *,
    convex: "ConvexAgentStore | None" = None,
) -> str:
    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        cfg = convex.get_research_config()
        return str(cfg.get("persona") or "").strip()
    return read_persona_file(data_dir)


def append_persona(
    data_dir: Path,
    persona_update: str,
    *,
    convex: "ConvexAgentStore | None" = None,
) -> None:
    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        convex.append_persona_block(persona_update)
        return
    path = data_dir / "PERSONA.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + "\n\n" + persona_update.strip(), encoding="utf-8")
    else:
        path.write_text(f"# User Persona\n\n{persona_update.strip()}", encoding="utf-8")


def append_memory(
    data_dir: Path,
    title: str,
    body: str,
    *,
    convex: ConvexAgentStore | None = None,
) -> None:
    if convex is not None:
        convex.append_entry(title, body)
        return
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    path = data_dir / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n## {title}\n*{ts}*\n\n{body.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(f"# Agent Memory\n{block}", encoding="utf-8")


def read_memory(
    data_dir: Path,
    max_chars: int = 8000,
    *,
    convex: ConvexAgentStore | None = None,
) -> str:
    if convex is not None:
        return convex.read_concatenated(max_chars)
    path = data_dir / "MEMORY.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[-max_chars:] if len(text) > max_chars else text


def clear_memory(
    data_dir: Path,
    *,
    convex: ConvexAgentStore | None = None,
) -> None:
    if convex is not None:
        convex.clear_all()
        return
    path = data_dir / "MEMORY.md"
    if path.is_file():
        path.write_text("# Agent Memory\n\n*Cleared.*\n", encoding="utf-8")


def run_proactive_research(
    gemini: GeminiClient,
    data_dir: Path,
    *,
    convex: ConvexAgentStore | None = None,
) -> str:
    """
    Performs proactive research. Each call advances through the heartbeat topic list
    (Convex or HEARTBEAT.md) in rotation. If no topics, falls back to persona-driven research.
    """
    if not gemini.enabled:
        return "gemini disabled — research skipped"

    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)

    soul = load_soul(data_dir, max_chars=800)
    topics = read_heartbeat_topics(data_dir, convex=convex)
    persona_text = read_persona(data_dir, convex=convex)
    ts = datetime.now(UTC).isoformat()

    if topics:
        if convex is not None:
            cfg = convex.get_research_config()
            idx = int(cfg.get("cursorIndex", 0)) % len(topics)
        else:
            cursor = read_cursor_file(data_dir)
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
        title = f"Research: {topic}"
        append_memory(data_dir, title, report, convex=convex)
        email_notify.notify_research_saved(
            get_settings(),
            title=title,
            report=report,
            source="proactive_research",
        )
        if convex is not None:
            convex.record_research_run(next_idx, ts)
        else:
            write_cursor_file(data_dir, next_idx, ts)
        logger.info("heartbeat: researched topic[%d/%d]: %s", idx + 1, len(topics), topic)
        return f"researched ({idx + 1}/{len(topics)}): {topic}"

    if persona_text:
        prompt = (
            f"{f'Directives:{chr(10)}{soul}{chr(10)}{chr(10)}' if soul else ''}"
            f"Research recent advancements relevant to this persona:\n{persona_text}\n\n"
            "Be concise; bullet points."
        )
        report = gemini.generate_with_search(prompt)[0]
        title = "Persona-driven Research"
        append_memory(data_dir, title, report, convex=convex)
        email_notify.notify_research_saved(
            get_settings(),
            title=title,
            report=report,
            source="persona_research",
        )
        if convex is not None:
            convex.record_research_run(0, ts)
        else:
            write_cursor_file(data_dir, 0, ts)
        logger.info("heartbeat: persona-driven research complete")
        return "persona-driven research stored"

    return "no topics in HEARTBEAT.md and no persona — nothing researched"
