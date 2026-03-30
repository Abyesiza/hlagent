"""
Training Pipeline — feeds scraped research data into the HDC Language Model.

Design
======
- Online (streaming) training: each document trains the model immediately
- Deduplication: content hashes prevent training on the same text twice
- Convex integration: training stats and corpus metadata logged to Convex
- Checkpointing: model saved after every N documents
- Graceful degradation: errors in one document don't stop the pipeline
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from super_agent.app.domain.hdc_lm import HDCLanguageModel, tokenise
from super_agent.app.services.research_tool import (
    ResearchResult,
    ScrapedDocument,
    research_topic,
    DEFAULT_SEED_TOPICS,
)

if TYPE_CHECKING:
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore

logger = logging.getLogger(__name__)

_CHECKPOINT_EVERY = 10       # save model every N documents trained


# ── dedup store ───────────────────────────────────────────────────────────────


class SeenHashStore:
    """Persist a set of content hashes to avoid duplicate training."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._hashes: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._hashes = set(data.get("hashes", []))
            except Exception:
                self._hashes = set()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"hashes": list(self._hashes)}, indent=2),
            encoding="utf-8",
        )

    def seen(self, h: str) -> bool:
        return h in self._hashes

    def add(self, h: str) -> None:
        self._hashes.add(h)
        self._save()

    def __len__(self) -> int:
        return len(self._hashes)


# ── pipeline stats ────────────────────────────────────────────────────────────


@dataclass
class PipelineStats:
    documents_trained: int = 0
    documents_skipped: int = 0
    tokens_trained: int = 0
    research_runs: int = 0
    errors: list[str] = field(default_factory=list)
    last_run: str = ""


# ── training pipeline ─────────────────────────────────────────────────────────


class TrainingPipeline:
    """
    Orchestrates: research → scrape → train HDC model.

    Usage
    -----
    pipeline = TrainingPipeline(lm, data_dir)
    await pipeline.run_topic("hyperdimensional computing")
    await pipeline.run_seed_topics()
    """

    def __init__(
        self,
        lm: HDCLanguageModel,
        data_dir: Path,
        convex_store: "ConvexAgentStore | None" = None,
        checkpoint_every: int = _CHECKPOINT_EVERY,
    ) -> None:
        self.lm = lm
        self.data_dir = data_dir
        self.convex_store = convex_store
        self.checkpoint_every = checkpoint_every
        self.stats = PipelineStats()
        self._seen = SeenHashStore(data_dir / "trained_hashes.json")
        self._docs_since_checkpoint = 0

    # ── train one document ────────────────────────────────────────────────────

    def train_document(self, doc: ScrapedDocument) -> int:
        """
        Train the model on a single ScrapedDocument.
        Returns number of n-gram pairs learned (0 if skipped/duplicate).
        """
        h = doc.content_hash or hashlib.md5(doc.text.encode()).hexdigest()
        if self._seen.seen(h):
            self.stats.documents_skipped += 1
            return 0

        try:
            pairs = self.lm.train_text(doc.text)
            self._seen.add(h)
            self.stats.documents_trained += 1
            self.stats.tokens_trained += doc.word_count
            self._docs_since_checkpoint += 1

            if self._docs_since_checkpoint >= self.checkpoint_every:
                self._checkpoint()
                self._docs_since_checkpoint = 0

            logger.info(
                "Trained doc [%s] '%s': %d pairs (vocab=%d)",
                doc.source, doc.title[:60], pairs, self.lm.stats.vocab_size,
            )
            return pairs
        except Exception as exc:
            err = f"train_document error ({doc.url}): {exc}"
            logger.warning(err)
            self.stats.errors.append(err)
            return 0

    # ── train a full research result ──────────────────────────────────────────

    def train_result(self, result: ResearchResult) -> int:
        """Train on all documents from a ResearchResult."""
        total = 0
        for doc in result.documents:
            total += self.train_document(doc)
        self.stats.last_run = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Log to Convex memory
        if self.convex_store is not None and result.documents:
            try:
                summary = (
                    f"Trained on topic '{result.topic}': "
                    f"{len(result.documents)} docs, "
                    f"{result.total_words:,} words, "
                    f"vocab now {self.lm.stats.vocab_size:,}"
                )
                self.convex_store.append_entry(f"Training: {result.topic}", summary)
            except Exception as exc:
                logger.debug("Convex log failed: %s", exc)
        return total

    # ── run a full research + train cycle for a topic ─────────────────────────

    async def run_topic(
        self,
        topic: str,
        max_pages: int = 5,
    ) -> PipelineStats:
        """Research a topic (async) and train the model on the results.

        Both the scraping AND the training are offloaded to a thread executor
        so the asyncio event loop never blocks.
        """
        logger.info("Pipeline: researching topic '%s'", topic)
        loop = asyncio.get_event_loop()

        def _research_and_train() -> int:
            result = research_topic(topic, max_pages=max_pages)
            self.stats.research_runs += 1
            return self.train_result(result), len(result.documents)

        pairs, doc_count = await loop.run_in_executor(None, _research_and_train)
        logger.info(
            "Pipeline topic '%s': %d docs → %d pairs",
            topic, doc_count, pairs,
        )
        return self.stats

    # ── run seed topics (initial bootstrap) ──────────────────────────────────

    async def run_seed_topics(self, topics: list[str] | None = None) -> PipelineStats:
        """Bootstrap the model on a set of foundational topics."""
        seed = topics or DEFAULT_SEED_TOPICS
        logger.info("Pipeline: seeding model on %d topics", len(seed))
        for topic in seed:
            await self.run_topic(topic, max_pages=3)
            await asyncio.sleep(2.0)   # polite delay between topics
        self._checkpoint()
        return self.stats

    # ── continuous learning loop ──────────────────────────────────────────────

    async def continuous_loop(
        self,
        topics: list[str],
        interval_seconds: int = 300,
    ) -> None:
        """
        Endlessly rotate through topics, researching and training.
        Designed to run as a background APScheduler job or asyncio task.
        """
        idx = 0
        while True:
            topic = topics[idx % len(topics)]
            idx += 1
            try:
                await self.run_topic(topic, max_pages=4)
            except Exception as exc:
                logger.warning("continuous_loop error on '%s': %s", topic, exc)
            await asyncio.sleep(interval_seconds)

    # ── checkpoint ───────────────────────────────────────────────────────────

    def _checkpoint(self) -> None:
        # Always try disk first (fast, zero-cost)
        model_path = self.data_dir / "hdc_model.json"
        try:
            self.lm.save(model_path)
        except Exception as exc:
            logger.warning("Disk checkpoint save failed: %s", exc)

        # Persist to Convex so the model survives Vercel cold starts
        if self.convex_store is not None:
            try:
                payload = self.lm.to_convex_payload()
                self.convex_store.save_model_weights(**payload)
                logger.info(
                    "Model checkpoint saved to Convex (vocab=%d tokens=%d)",
                    self.lm.stats.vocab_size, self.lm.stats.training_tokens,
                )
            except Exception as exc:
                logger.warning("Convex checkpoint save failed: %s", exc)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "pipeline": {
                "documents_trained": self.stats.documents_trained,
                "documents_skipped": self.stats.documents_skipped,
                "tokens_trained": self.stats.tokens_trained,
                "research_runs": self.stats.research_runs,
                "unique_docs_seen": len(self._seen),
                "last_run": self.stats.last_run,
                "recent_errors": self.stats.errors[-5:],
            },
            "model": {
                "vocab_size": self.lm.stats.vocab_size,
                "training_tokens": self.lm.stats.training_tokens,
                "training_docs": self.lm.stats.training_docs,
                "assoc_memory_size": self.lm.space.assoc_memory_size,
                "dim": self.lm.dim,
                "context_size": self.lm.context_size,
                "last_trained": self.lm.stats.last_trained,
            },
        }
