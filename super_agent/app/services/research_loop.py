"""
Research loop helpers — heartbeat topic management using the HDC research tool.

Replaces the former Gemini-based research_loop. Now uses the web scraper and
feeds results directly into the HDC training pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
        if ln.startswith(("-", "*")):
            t = ln.lstrip("-* ").strip()
            if t:
                topics.append(t)
    return topics[:100]


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
    cursor["runs"] = int(cursor.get("runs", 0)) + 1
    (data_dir / _CURSOR_FILE).write_text(json.dumps(cursor, indent=2), encoding="utf-8")


def read_heartbeat_topics(
    data_dir: Path,
    *,
    convex: "ConvexAgentStore | None" = None,
) -> list[str]:
    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        cfg = convex.get_research_config()
        return list(cfg.get("topics") or [])[:100]
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


def append_memory(
    data_dir: Path,
    title: str,
    body: str,
    *,
    convex: "ConvexAgentStore | None" = None,
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
    convex: "ConvexAgentStore | None" = None,
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
    convex: "ConvexAgentStore | None" = None,
) -> None:
    if convex is not None:
        convex.clear_all()
        return
    path = data_dir / "MEMORY.md"
    if path.is_file():
        path.write_text("# Agent Memory\n\n*Cleared.*\n", encoding="utf-8")


def _expand_topic(topic: str, run_count: int) -> list[str]:
    """
    Return a list of query variants for a topic.

    To defeat the deduplication cache (which correctly prevents re-training on
    the *same* article), we rotate through progressively narrower sub-queries so
    every heartbeat cycle fetches *different* articles for the same broad topic.
    """
    variants = [
        topic,
        f"{topic} introduction overview",
        f"{topic} history origins",
        f"{topic} applications examples",
        f"{topic} latest research 2025",
        f"{topic} tutorial explained",
        f"{topic} advantages disadvantages",
        f"{topic} comparison alternatives",
    ]
    return [variants[run_count % len(variants)], topic]


def run_proactive_research(
    data_dir: Path,
    lm: object,
    *,
    convex: "ConvexAgentStore | None" = None,
    max_pages: int = 5,
    pipeline: object = None,
) -> str:
    """
    Research the next heartbeat topic, train the HDC model, and persist state.

    Accepts an optional *pipeline* (TrainingPipeline) to reuse the shared instance
    so stats are visible at /train/status.  If not provided a fresh instance is
    created (backward-compatible).
    """
    from super_agent.app.domain.hdc_lm import HDCLanguageModel
    from super_agent.app.services.research_tool import research_topic, DEFAULT_SEED_TOPICS
    from super_agent.app.services.training_pipeline import TrainingPipeline

    if not isinstance(lm, HDCLanguageModel):
        return "No HDC language model available"

    topics = read_heartbeat_topics(data_dir, convex=convex)
    if not topics:
        topics = DEFAULT_SEED_TOPICS

    if convex is not None:
        convex.ensure_seeded_from_disk(data_dir)
        cfg = convex.get_research_config()
        idx = int(cfg.get("cursorIndex", 0)) % len(topics)
        run_count = int(cfg.get("totalRuns") or 0)
    else:
        cursor = read_cursor_file(data_dir)
        idx = int(cursor.get("index", 0)) % len(topics)
        run_count = int(cursor.get("runs", 0))

    topic = topics[idx]
    next_idx = (idx + 1) % len(topics)
    ts = datetime.now(UTC).isoformat()

    logger.info("Heartbeat[run=%d]: researching topic[%d/%d]: %s", run_count, idx + 1, len(topics), topic)

    # Use the shared pipeline if provided, else create a local one
    if not isinstance(pipeline, TrainingPipeline):
        pipeline = TrainingPipeline(lm, data_dir, convex_store=convex)

    # Expand into sub-queries to get fresh articles past the dedup cache
    queries = _expand_topic(topic, run_count)
    total_pairs = 0
    total_docs = 0
    total_words = 0

    for query in queries:
        result = research_topic(query, max_pages=max_pages)
        pairs = pipeline.train_result(result)
        total_pairs += pairs
        total_docs += len(result.documents)
        total_words += result.total_words

    # Save checkpoint after every heartbeat cycle
    pipeline._checkpoint()

    summary = (
        f"Researched '{topic}' ({len(queries)} queries): "
        f"{total_docs} docs, {total_words:,} words, "
        f"{total_pairs} n-gram pairs. Vocab: {lm.stats.vocab_size:,}."
    )
    append_memory(data_dir, f"Research+Train: {topic}", summary, convex=convex)

    if convex is not None:
        convex.record_research_run(next_idx, ts)
    else:
        write_cursor_file(data_dir, next_idx, ts)

    return f"trained ({idx + 1}/{len(topics)}): {topic} — {total_pairs} pairs, vocab={lm.stats.vocab_size:,}"
