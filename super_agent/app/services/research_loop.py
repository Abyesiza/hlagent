from __future__ import annotations

import logging
from pathlib import Path

from super_agent.app.core.gemini_client import GeminiClient
from super_agent.app.services.workspace_context import load_soul

logger = logging.getLogger(__name__)


def read_heartbeat_topics(data_dir: Path) -> list[str]:
    path = data_dir / "HEARTBEAT.md"
    if not path.is_file():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    topics: list[str] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith(("-", "*")):
            topics.append(ln.lstrip("-* ").strip())
        else:
            topics.append(ln)
    return topics[:20]


def append_memory(data_dir: Path, title: str, body: str) -> None:
    path = data_dir / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n## {title}\n\n{body.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(f"# Memory\n{block}", encoding="utf-8")


def run_proactive_research(gemini: GeminiClient, data_dir: Path) -> str:
    topics = read_heartbeat_topics(data_dir)
    if not topics:
        return "no topics in HEARTBEAT.md"
    topic = topics[0]
    if not gemini.enabled:
        return f"would research: {topic} (gemini disabled)"
    soul = load_soul(data_dir, max_chars=1200)
    prefix = f"Directives:\n{soul}\n\n" if soul else ""
    report = gemini.generate_text(
        prefix + f"Research and summarize actionable steps for: {topic}. Be concise; bullet points.",
        model=None,
    )
    append_memory(data_dir, f"Heartbeat: {topic}", report)
    logger.info("heartbeat research stored for %s", topic)
    return f"stored research for: {topic}"
