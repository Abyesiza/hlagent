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


def read_persona(data_dir: Path) -> str:
    """Reads the user's persona from PERSONA.md."""
    path = data_dir / "PERSONA.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def append_persona(data_dir: Path, persona_update: str) -> None:
    """
    Appends or creates the user's persona in PERSONA.md.
    This allows for iterative refinement of the persona.
    """
    path = data_dir / "PERSONA.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        # Append with a double newline for separation
        path.write_text(path.read_text(encoding="utf-8") + "\n\n" + persona_update.strip(), encoding="utf-8")
    else:
        path.write_text(f"# User Persona\n\n{persona_update.strip()}", encoding="utf-8")


def append_memory(data_dir: Path, title: str, body: str) -> None:
    path = data_dir / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n## {title}\n\n{body.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(f"# Memory\n{block}", encoding="utf-8")


def run_proactive_research(gemini: GeminiClient, data_dir: Path) -> str:
    """
    Performs proactive research based on HEARTBEAT topics or, if none,
    derives research topics from the user's persona.
    """
    persona_text = read_persona(data_dir)
    soul = load_soul(data_dir, max_chars=1200)

    # Determine the primary research query and logging topic
    topics = read_heartbeat_topics(data_dir)
    research_goal_prompt: str
    memory_title: str
    log_topic: str

    if topics:
        log_topic = topics[0]
        research_goal_prompt = f"Research and summarize actionable steps for: {log_topic}."
        memory_title = f"Heartbeat: {log_topic}"
        if persona_text:
            # If both heartbeat and persona exist, persona provides context for the heartbeat topic
            research_goal_prompt = f"Considering the user's persona:\n{persona_text}\n\n" + research_goal_prompt
    elif persona_text:
        log_topic = "relevant advancements for user persona"
        research_goal_prompt = (
            f"Research and summarize actionable steps relevant to the user's described persona, "
            f"focusing on recent advancements or common pain points for their described technologies. "
            f"The persona is:\n{persona_text}"
        )
        memory_title = "Persona-driven Research"
    else:
        return "no topics in HEARTBEAT.md and no persona established"

    if not gemini.enabled:
        return f"would research: {log_topic} (gemini disabled)"

    # Build the full prompt for the LLM
    prefix = f"Directives:\n{soul}\n\n" if soul else ""
    full_prompt = prefix + research_goal_prompt + "\nBe concise; bullet points."

    report = gemini.generate_text(full_prompt, model=None)
    append_memory(data_dir, memory_title, report)
    logger.info("research stored for %s", log_topic)
    return f"stored research for: {log_topic}"
