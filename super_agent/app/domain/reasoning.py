"""
HDC Reasoning Engine.

Routes each user query to the best available reasoning mode and formats
a clean, conversational response regardless of the model's learning stage.

Modes
=====
- math       — SymPy symbolic evaluation
- analogy    — A:B::C:? via HDC vector algebra
- similarity — nearest-neighbour word lookup
- generation — HDC generative continuation (when confident enough)
- learning   — graceful fallback while model is still training
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from super_agent.app.domain.hdc_lm import HDCLanguageModel

logger = logging.getLogger(__name__)

# ── intent patterns ───────────────────────────────────────────────────────────

_MATH_RE = re.compile(
    r"(\b\d[\d\s\+\-\*/\^\.\(\)=]+\b"
    r"|solve\s+|integral\s+|derivative\s+"
    r"|equation|factor|simplify|expand"
    r"|sin|cos|tan|log|sqrt|exp)",
    re.I,
)
_ANALOGY_RE = re.compile(
    r"\b(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to\s+\?",
    re.I,
)
_SIMILAR_RE = re.compile(
    r"(?:what(?:'s|\s+is)\s+similar\s+to|words?\s+like|related\s+to)\s+[\"']?(\w+)[\"']?",
    re.I,
)

# Stop-words we skip when looking for meaningful topic words
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "about", "above", "after", "before",
    "between", "during", "for", "from", "in", "into", "of", "on", "over",
    "through", "to", "under", "up", "with", "tell", "know", "explain",
    "describe", "give", "show", "help", "please", "hey", "hi", "hello",
    "and", "or", "but", "if", "then", "because", "while", "although",
})


def classify_query(text: str) -> str:
    if _MATH_RE.search(text):
        return "math"
    if _ANALOGY_RE.search(text):
        return "analogy"
    if _SIMILAR_RE.search(text):
        return "similarity"
    return "generation"


def _extract_topic_words(message: str) -> list[str]:
    """Return meaningful (non-stop) words from the raw user message."""
    tokens = re.findall(r"[a-zA-Z0-9']+", message.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 2]


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    mode: str
    answer: str
    confidence: float
    details: dict


# ── reasoning engine ─────────────────────────────────────────────────────────

class ReasoningEngine:
    """Routes queries and produces clean, conversational responses."""

    def __init__(self, lm: "HDCLanguageModel") -> None:
        self.lm = lm

    def reason(
        self,
        prompt: str,
        *,
        user_message: str | None = None,
    ) -> ReasoningResult:
        """
        Parameters
        ----------
        prompt       Full context prompt (may include session history).
        user_message The raw user message only — used for topic extraction
                     and response framing. Falls back to prompt if not given.
        """
        raw = user_message or prompt
        mode = classify_query(raw)
        try:
            if mode == "math":
                return self._math_reason(raw)
            if mode == "analogy":
                return self._analogy_reason(raw)
            if mode == "similarity":
                return self._similarity_reason(raw)
            return self._generative_reason(prompt, raw)
        except Exception as exc:
            logger.warning("ReasoningEngine error (mode=%s): %s", mode, exc)
            return ReasoningResult(
                mode="error",
                answer="Something went wrong while processing your query.",
                confidence=0.0,
                details={"error": str(exc)},
            )

    # ── math ─────────────────────────────────────────────────────────────────

    def _math_reason(self, query: str) -> ReasoningResult:
        try:
            from super_agent.app.infrastructure.sympy_runner import run_symcode
            outcome = run_symcode(f"result = {query.strip()}")
            if outcome.ok:
                return ReasoningResult(
                    mode="math",
                    answer=f"= {outcome.result}",
                    confidence=1.0,
                    details={"sympy_result": str(outcome.result)},
                )
        except Exception as exc:
            logger.debug("SymPy path failed: %s", exc)
        return self._generative_reason(query, query)

    # ── analogy ───────────────────────────────────────────────────────────────

    def _analogy_reason(self, query: str) -> ReasoningResult:
        m = _ANALOGY_RE.search(query)
        if not m:
            return self._generative_reason(query, query)
        a, b, c = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
        candidates = self.lm.analogy(a, b, c, top_k=5)
        if not candidates:
            return ReasoningResult(
                mode="analogy",
                answer=f"I don't have enough vocabulary yet to solve \"{a} : {b} :: {c} : ?\".",
                confidence=0.0,
                details={"a": a, "b": b, "c": c},
            )
        best_word, best_conf = candidates[0]
        others = ", ".join(w for w, _ in candidates[1:3])
        answer = f"{a} is to {b} as {c} is to **{best_word}**."
        if others:
            answer += f"\nOther candidates: {others}."
        return ReasoningResult(
            mode="analogy",
            answer=answer,
            confidence=best_conf,
            details={"query": f"{a}:{b}::{c}:?", "candidates": candidates},
        )

    # ── similarity ────────────────────────────────────────────────────────────

    def _similarity_reason(self, query: str) -> ReasoningResult:
        m = _SIMILAR_RE.search(query)
        word = m.group(1).lower() if m else query.split()[-1].lower()
        similar = self.lm.most_similar(word, top_k=8)
        if not similar:
            return ReasoningResult(
                mode="similarity",
                answer=f"I don't have \"{word}\" in my vocabulary yet.",
                confidence=0.0,
                details={"word": word},
            )
        words_str = ", ".join(w for w, _ in similar)
        return ReasoningResult(
            mode="similarity",
            answer=f"Words closest to **{word}** in my memory: {words_str}.",
            confidence=similar[0][1],
            details={"word": word, "similar": similar},
        )

    # ── generative ────────────────────────────────────────────────────────────

    def _generative_reason(self, prompt: str, raw_message: str) -> ReasoningResult:
        """
        Generate a response using whatever the model currently knows.

        Strategy:
        1. Always attempt text generation from the full query.
        2. Also generate from each known topic word as a seed — topic-seeded
           generation is often more coherent than question-seeded generation.
        3. When generation produces nothing, describe the semantic neighbourhood
           in natural language ("X is associated with: a, b, c").
        4. Stats are a small footnote, never the main content.
        """
        vocab = self.lm.stats.vocab_size
        tokens_trained = self.lm.stats.training_tokens

        topic_words = _extract_topic_words(raw_message)

        raw_tokens = re.findall(r"[a-zA-Z0-9']+", raw_message.lower())
        ctx_tokens = raw_tokens[-self.lm.context_size:] if raw_tokens else []
        candidates = self.lm.predict_next(ctx_tokens, top_k=8) if ctx_tokens else []
        confidence = candidates[0][1] if candidates else 0.0

        # ── Always attempt generation from the full query ─────────────────────
        generated = ""
        if vocab >= 200:
            generated = self.lm.generate(
                raw_message, max_tokens=60, temperature=0.75
            ).strip()

        # Return clean generation if signal is strong enough
        if generated and confidence >= 0.08:
            return ReasoningResult(
                mode="generation",
                answer=generated,
                confidence=confidence,
                details={
                    "seed": raw_message,
                    "top_candidates": candidates[:3],
                    "vocab_size": vocab,
                    "training_tokens": tokens_trained,
                },
            )

        # ── Handle no topic words ─────────────────────────────────────────────
        if not topic_words:
            return ReasoningResult(
                mode="learning",
                answer=(
                    "I'm still building my vocabulary. "
                    "Keep the heartbeat running and I'll get smarter."
                ),
                confidence=confidence,
                details={"vocab_size": vocab, "training_tokens": tokens_trained},
            )

        known = [w for w in topic_words if self.lm.space.item_memory.get(w) is not None]
        unknown = [w for w in topic_words if w not in known]

        if not known:
            return ReasoningResult(
                mode="learning",
                answer=(
                    f"I haven't learned about **{', '.join(unknown[:4])}** yet.\n"
                    f"Use the Research tab to train me on that topic "
                    f"({vocab:,} words in vocabulary so far)."
                ),
                confidence=confidence,
                details={
                    "unknown_topic_words": unknown,
                    "vocab_size": vocab,
                    "training_tokens": tokens_trained,
                },
            )

        # ── Build a knowledge-first answer ────────────────────────────────────
        lines: list[str] = []

        # Lead with full-query generation when we got something (even low-conf)
        if generated:
            lines.append(generated)
            lines.append("")

        # Generate from each known topic word as its own seed — produces
        # tighter, more on-topic text than generating from the full question
        for word in known[:3]:
            word_gen = self.lm.generate(word, max_tokens=30, temperature=0.85).strip()
            if word_gen:
                lines.append(f"**{word.capitalize()}**: {word} {word_gen}")
            else:
                # Fallback: describe the word's neighbourhood in natural language
                similar = self.lm.most_similar(word, top_k=10)
                neighbours = [
                    w for w, _ in similar
                    if w not in _STOP and w != word and w.isalpha()
                ][:5]
                if neighbours:
                    lines.append(
                        f"**{word.capitalize()}** is associated with: "
                        f"{', '.join(neighbours)}."
                    )

        if unknown:
            lines.append(f"\nStill learning about: {', '.join(unknown[:4])}.")

        lines.append(
            f"\n*{vocab:,} words trained · {confidence * 100:.1f}% confidence*"
        )

        return ReasoningResult(
            mode="learning",
            answer="\n".join(lines),
            confidence=confidence,
            details={
                "known_topic_words": known,
                "unknown_topic_words": unknown,
                "vocab_size": vocab,
                "training_tokens": tokens_trained,
                "next_word_candidates": candidates[:5],
            },
        )
