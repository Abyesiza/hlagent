"""
HDC Super-Agent Orchestrator.

Routes incoming queries to the appropriate reasoning mode:
  - Math expressions → SymPy symbolic solver
  - Analogy/similarity → HDC vector algebra
  - General text     → HDC Language Model generation
  - Research queries → scrape + train + answer

Replaces the former Gemini-based orchestrator.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import AsyncIterator, TYPE_CHECKING

from super_agent.app.domain.hdc_lm import HDCLanguageModel, tokenise
from super_agent.app.domain.reasoning import ReasoningEngine, ReasoningResult
from super_agent.app.infrastructure.hdc_memory_store import HDCMemoryStore
from super_agent.app.services.session_store import Session, SessionStore

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore

logger = logging.getLogger(__name__)


class ChatTurnResult:
    """Result of a single orchestrator turn."""

    def __init__(
        self,
        answer: str,
        mode: str = "generation",
        confidence: float = 0.0,
        session_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.answer = answer
        self.mode = mode
        self.confidence = confidence
        self.session_id = session_id
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "details": self.details,
        }


class SuperAgentOrchestrator:
    """
    Main orchestration layer.

    Holds references to:
    - HDCLanguageModel (the brain)
    - ReasoningEngine (routes + synthesises)
    - HDCMemoryStore (legacy associative memory, retained for recall context)
    - SessionStore (multi-turn conversation history)
    - ConvexAgentStore (optional cloud memory + tasks)
    """

    def __init__(
        self,
        settings: "Settings",
        lm: HDCLanguageModel,
        hdc_memory: HDCMemoryStore,
        sessions: SessionStore,
        *,
        convex_store: "ConvexAgentStore | None" = None,
    ) -> None:
        self.settings = settings
        self.lm = lm
        self.hdc_memory = hdc_memory
        self.sessions = sessions
        self.convex_store = convex_store
        self.reasoning = ReasoningEngine(lm)

    # ── main chat turn ────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ChatTurnResult:
        """
        Process one user message and return a ChatTurnResult.
        """
        session: Session | None = None
        if session_id:
            session = self.sessions.get_or_create(session_id)
            session.add("user", message)

        # Build enriched prompt with conversation history
        prompt = self._build_prompt(message, session)

        # Route and generate — pass the raw user message separately so the
        # reasoning engine can use it for topic extraction without being
        # polluted by session history.
        result: ReasoningResult = self.reasoning.reason(
            prompt, user_message=message
        )

        # Store in HDC memory for future retrieval
        try:
            self.hdc_memory.remember(
                task=message[:200],
                solution_repr=result.answer[:200],
                route=result.mode,
            )
        except Exception as exc:
            logger.debug("HDC memory store failed: %s", exc)

        # Persist to Convex if available
        if self.convex_store is not None:
            try:
                self.convex_store.append_entry(
                    title=f"Chat [{result.mode}]",
                    body=f"Q: {message[:200]}\nA: {result.answer[:400]}",
                )
            except Exception as exc:
                logger.debug("Convex append failed: %s", exc)

        if session_id and session:
            session.add("assistant", result.answer)

        return ChatTurnResult(
            answer=result.answer,
            mode=result.mode,
            confidence=result.confidence,
            session_id=session_id,
            details=result.details,
        )

    async def chat_stream(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming variant — yields tokens one at a time.
        Uses HDC generate_stream() for the generative path.
        """
        session: Session | None = None
        if session_id:
            session = self.sessions.get_or_create(session_id)

        prompt = self._build_prompt(message, session)
        full_answer: list[str] = []

        for token in self.lm.generate_stream(prompt, max_tokens=80, temperature=0.75):
            full_answer.append(token)
            yield token + " "

        if session_id and session:
            session.add("user", message)
            session.add("assistant", " ".join(full_answer))

    # ── research + train + answer ─────────────────────────────────────────────

    async def research_and_answer(
        self,
        topic: str,
        question: str | None = None,
    ) -> ChatTurnResult:
        """
        Research a topic on the internet, train the model on the results,
        then answer the question using the freshly learned knowledge.
        """
        from super_agent.app.services.research_tool import research_topic
        from super_agent.app.services.training_pipeline import TrainingPipeline
        import asyncio

        pipeline = TrainingPipeline(
            self.lm,
            self.settings.data_dir,
            convex_store=self.convex_store,
        )

        loop = asyncio.get_event_loop()
        research_result = await loop.run_in_executor(
            None,
            lambda: research_topic(
                topic,
                max_pages=self.settings.scraper_max_pages,
                include_wikipedia=self.settings.include_wikipedia,
            ),
        )
        pairs = pipeline.train_result(research_result)

        if question:
            answer = self.lm.answer_from_context(
                question=question,
                context_text=" ".join(d.text[:2000] for d in research_result.documents[:3]),
                max_tokens=80,
            )
        else:
            # Summarise what was learned
            doc_titles = [d.title for d in research_result.documents[:5]]
            answer = (
                f"Researched '{topic}': trained on {len(research_result.documents)} documents "
                f"({research_result.total_words:,} words). "
                f"Sources: {', '.join(doc_titles[:3])}. "
                f"Vocabulary now: {self.lm.stats.vocab_size:,} words."
            )

        return ChatTurnResult(
            answer=answer,
            mode="research",
            confidence=0.9 if research_result.documents else 0.1,
            details={
                "topic": topic,
                "docs_count": len(research_result.documents),
                "total_words": research_result.total_words,
                "pairs_trained": pairs,
                "vocab_after": self.lm.stats.vocab_size,
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(self, message: str, session: Session | None) -> str:
        if session is None or not session.turns:
            return message
        # Include last 3 turns as context prefix
        recent = list(session.turns)[-6:]   # up to 3 user + 3 assistant turns
        lines: list[str] = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.text[:200]}")
        lines.append(f"User: {message}")
        return "\n".join(lines)

    # ── status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "model": repr(self.lm),
            "vocab_size": self.lm.stats.vocab_size,
            "training_tokens": self.lm.stats.training_tokens,
            "training_docs": self.lm.stats.training_docs,
            "assoc_memory": self.lm.space.assoc_memory_size,
            "last_trained": self.lm.stats.last_trained,
            "hdc_memory_records": len(self.hdc_memory._records),
            "convex_connected": self.convex_store is not None,
        }
