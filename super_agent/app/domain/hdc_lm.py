"""
HDC Language Model — generative text model built on Vector Symbolic Architecture.

Architecture
============
Unlike transformers that learn weight matrices through backprop, this model:

1. Represents every word as a random (but reproducible) atomic hypervector
2. Encodes a context window using position-aware permutations + bundling
3. Predicts the next word by:
   a. Recall: unbind the context from the global associative memory
   b. Clean-up: nearest-neighbour lookup in the Item Memory
4. Learns new text in one shot: bind(context, next_word) → bundle into memory

Advantages over LLMs
====================
- One-shot learning: a single sentence updates the model instantly
- Transparency: you can decode any memory state algebraically
- O(D) inference instead of O(N²) attention
- Runs on CPU with NumPy — no GPU required

Limitations
===========
- Signal-to-noise degrades as the associative memory fills (~√assoc_count noise)
- No deep hierarchical feature learning (compensated by the reasoning module)
- Generation is probabilistic similarity search, not a distribution over tokens
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from super_agent.app.domain.hdc_engine import HDCSpace, ItemMemory, bundle, DEFAULT_DIM

logger = logging.getLogger(__name__)


# ── tokeniser ────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+|[.,!?;:\"()\[\]{}\-/\\]")
_SPECIAL = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}


def tokenise(text: str, lowercase: bool = True) -> list[str]:
    """
    Simple word + punctuation tokeniser.
    Keeps contractions ("don't") as single tokens.
    """
    tokens = _TOKEN_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def detokenise(tokens: list[str]) -> str:
    """Reconstruct approximate text from tokens (no-op punct spacing)."""
    if not tokens:
        return ""
    out = tokens[0]
    for t in tokens[1:]:
        if t in ".,!?;:)]}" or out.endswith("("):
            out += t
        else:
            out += " " + t
    return out


# ── n-gram extractor ──────────────────────────────────────────────────────────


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[list[str], str]]:
    """
    Yield (context_window, next_token) pairs for training.
    context_window has exactly n tokens, next_token is the token to predict.
    """
    pairs: list[tuple[list[str], str]] = []
    bos = ["<BOS>"] * n
    padded = bos + tokens
    for i in range(n, len(padded)):
        ctx = padded[i - n: i]
        nxt = padded[i]
        pairs.append((ctx, nxt))
    return pairs


# ── model stats ───────────────────────────────────────────────────────────────


@dataclass
class HDCLMStats:
    vocab_size: int = 0
    training_tokens: int = 0
    training_docs: int = 0
    queries: int = 0
    cache_hits: int = 0
    last_trained: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "training_tokens": self.training_tokens,
            "training_docs": self.training_docs,
            "queries": self.queries,
            "cache_hits": self.cache_hits,
            "last_trained": self.last_trained,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HDCLMStats":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ── HDC Language Model ────────────────────────────────────────────────────────


class HDCLanguageModel:
    """
    Generative HDC-based language model with one-shot online learning.

    Usage
    -----
    model = HDCLanguageModel.load(path)  # or HDCLanguageModel()
    model.train_text("The cat sat on the mat.")
    next_word, confidence = model.predict_next(["the", "cat", "sat"])
    generated = model.generate("The cat", max_tokens=20)
    model.save(path)
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        context_size: int = 5,
    ) -> None:
        self.dim = dim
        self.context_size = context_size
        self.space = HDCSpace(dim=dim)
        self.stats = HDCLMStats(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self._lock = threading.RLock()
        # Special tokens registered once
        for tok in _SPECIAL:
            self.space.symbol(tok)

    # ── training ──────────────────────────────────────────────────────────────

    def train_tokens(self, tokens: list[str]) -> int:
        """
        Train on a pre-tokenised sequence. Returns number of pairs learned.

        Lock strategy: build the batch superposition vector WITHOUT holding
        self._lock (the heavy loop), then acquire the lock once to merge the
        batch into the shared associative memory.  This prevents the long
        training loop from starving concurrent predict_next / generate calls.
        """
        if len(tokens) < 2:
            return 0
        pairs = extract_ngrams(tokens, self.context_size)

        # ── Phase 1: build hypervectors outside the lock ────────────────────
        # encode_ngram and symbol are safe to call without self._lock because
        # ItemMemory.add / .get have their own internal lock.
        batch_hv: np.ndarray | None = None
        from super_agent.app.domain.hdc_engine import bundle as _bundle
        for ctx, nxt in pairs:
            ctx_hv = self.space.encode_ngram(ctx)
            nxt_hv = self.space.symbol(nxt)
            pair_hv = self.space.bind(ctx_hv, nxt_hv)
            if batch_hv is None:
                batch_hv = pair_hv.copy()
            else:
                batch_hv = _bundle([batch_hv, pair_hv])

        # ── Phase 2: merge batch into shared memory with a brief lock ───────
        with self._lock:
            if batch_hv is not None:
                if self.space._assoc_memory is None:
                    self.space._assoc_memory = batch_hv
                else:
                    self.space._assoc_memory = _bundle(
                        [self.space._assoc_memory, batch_hv]
                    )
                self.space._assoc_count += len(pairs)
            self.stats.training_tokens += len(tokens)
            self.stats.vocab_size = len(self.space.item_memory)
            self.stats.last_trained = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return len(pairs)

    def train_text(self, text: str) -> int:
        """Tokenise and train on a raw text string."""
        tokens = tokenise(text)
        if not tokens:
            return 0
        pairs = self.train_tokens(tokens)
        with self._lock:
            self.stats.training_docs += 1
        logger.debug("Trained on %d tokens → %d pairs", len(tokens), pairs)
        return pairs

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_next(
        self, context: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Predict the most likely next tokens given a context window.
        Returns a list of (token, confidence) sorted descending.
        """
        if not context:
            return []
        ctx = context[-self.context_size:]
        if len(ctx) < self.context_size:
            ctx = ["<BOS>"] * (self.context_size - len(ctx)) + ctx

        # encode_ngram / recall are read-only on assoc_memory — safe without lock
        ctx_hv = self.space.encode_ngram(ctx)
        with self._lock:
            candidate = self.space.recall(ctx_hv)
            if candidate is None:
                return []
            self.stats.queries += 1

        # nearest() has its own internal lock; run it outside self._lock
        # to allow training to proceed concurrently
        results = self.space.item_memory.nearest(candidate, top_k=top_k)
        return [(tok, conf) for tok, conf in results if tok not in _SPECIAL]

    def predict_best(self, context: list[str]) -> tuple[str, float]:
        """
        Convenience: returns (best_token, confidence).
        Returns ("<UNK>", 0.0) when nothing is found.
        """
        results = self.predict_next(context, top_k=1)
        if not results:
            return "<UNK>", 0.0
        return results[0]

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        seed_text: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        min_confidence: float = 0.0,
        stop_tokens: tuple[str, ...] = ("<EOS>",),
    ) -> str:
        """
        Autoregressive text generation.

        temperature: > 1 → more random (samples from top-5), < 1 → greedier.
        min_confidence: stop if best prediction is below this threshold.
        """
        tokens = tokenise(seed_text)
        generated = list(tokens)

        for _ in range(max_tokens):
            ctx = generated[-self.context_size:]
            candidates = self.predict_next(ctx, top_k=5)
            if not candidates:
                break

            # Apply temperature sampling
            if temperature <= 0.01:
                chosen, conf = candidates[0]
            else:
                words = [c[0] for c in candidates]
                scores = np.array([c[1] for c in candidates], dtype=np.float64)
                # Shift scores to be positive before temperature scaling
                scores -= scores.min() - 1e-8
                scores = scores ** (1.0 / temperature)
                probs = scores / scores.sum()
                chosen = str(np.random.choice(words, p=probs))
                conf = dict(candidates).get(chosen, 0.0)

            if conf < min_confidence:
                break
            if chosen in stop_tokens:
                break
            generated.append(chosen)

        return detokenise(generated[len(tokens):])

    def generate_stream(
        self,
        seed_text: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
    ) -> Iterator[str]:
        """Streaming generator — yields one token at a time."""
        tokens = tokenise(seed_text)
        generated = list(tokens)

        for _ in range(max_tokens):
            ctx = generated[-self.context_size:]
            candidates = self.predict_next(ctx, top_k=5)
            if not candidates:
                return

            words = [c[0] for c in candidates]
            scores = np.array([c[1] for c in candidates], dtype=np.float64)
            scores -= scores.min() - 1e-8
            scores = scores ** (1.0 / max(temperature, 0.01))
            probs = scores / scores.sum()
            chosen = str(np.random.choice(words, p=probs))
            if chosen == "<EOS>":
                return

            generated.append(chosen)
            yield chosen

    # ── analogy query (built-in reasoning) ───────────────────────────────────

    def analogy(self, a: str, b: str, c: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Solve: A is to B as C is to ?
        Vector algebra: result ≈ bind(unbind(symbol(a), symbol(b)), symbol(c))
        Example: king - man + woman ≈ queen
        """
        with self._lock:
            a_hv = self.space.symbol(a)
            b_hv = self.space.symbol(b)
            c_hv = self.space.symbol(c)
            # unbind a from b, then bind with c
            diff = self.space.unbind(b_hv, a_hv)
            result = self.space.bind(diff, c_hv)
            return self.space.item_memory.nearest(result, top_k=top_k)

    # ── similarity query ──────────────────────────────────────────────────────

    def most_similar(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Find words most similar to a given word in hypervector space."""
        with self._lock:
            hv = self.space.symbol(word)
            results = self.space.item_memory.nearest(hv, top_k=top_k + 1)
        return [(w, s) for w, s in results if w != word][:top_k]

    # ── context answer (RAG-style) ────────────────────────────────────────────

    def answer_from_context(
        self, question: str, context_text: str, max_tokens: int = 80
    ) -> str:
        """
        Temporarily train on context_text, generate an answer for question,
        then returns the generated continuation.
        This is a primitive "in-context learning" — full one-shot style.
        """
        # Snapshot current assoc memory size
        pre_count = self.space.assoc_memory_size
        self.train_text(context_text)
        answer = self.generate(question, max_tokens=max_tokens)
        logger.debug(
            "answer_from_context: trained %d new pairs, generated %d chars",
            self.space.assoc_memory_size - pre_count,
            len(answer),
        )
        return answer

    # ── Convex persistence (Vercel-safe) ─────────────────────────────────────

    def to_convex_payload(self) -> dict:
        """
        Serialize the trained model state to a flat dict for Convex storage.

        Encoding strategy (avoids Convex's 8192-element array limit):
        - assocMemoryB64 : the ±1 bipolar weight vector packed as int8 bytes
                           then base64-encoded → ~13 KB for dim=10 000
        - vocabLabels    : newline-delimited word list; item-memory hypervectors
                           are deterministic and will be regenerated on load
        """
        import base64

        with self._lock:
            if self.space._assoc_memory is not None:
                arr_int8 = self.space._assoc_memory.astype(np.int8)
                assoc_b64 = base64.b64encode(arr_int8.tobytes()).decode("ascii")
            else:
                assoc_b64 = ""
            vocab_str = "\n".join(self.space.item_memory.labels())
            assoc_count = self.space._assoc_count
            training_tokens = self.stats.training_tokens
            training_docs = self.stats.training_docs
            last_trained = self.stats.last_trained or None
            created_at = self.stats.created_at

        return {
            "dim": self.dim,
            "context_size": self.context_size,
            "assoc_count": assoc_count,
            "assoc_memory_b64": assoc_b64,
            "vocab_labels": vocab_str,
            "training_tokens": training_tokens,
            "training_docs": training_docs,
            "last_trained": last_trained,
            "created_at": created_at,
        }

    @classmethod
    def from_convex_payload(cls, data: dict) -> "HDCLanguageModel":
        """
        Reconstruct a model from a Convex payload (result of load_model_weights).

        Item-memory hypervectors are regenerated deterministically from the
        stored vocabulary labels — no need to persist the full matrix.
        """
        import base64

        dim = int(data.get("dim", DEFAULT_DIM))
        context_size = int(data.get("contextSize", data.get("context_size", 5)))
        obj = cls(dim=dim, context_size=context_size)

        obj.space._assoc_count = int(data.get("assocCount", data.get("assoc_count", 0)))

        assoc_b64 = data.get("assocMemoryB64", data.get("assoc_memory_b64", ""))
        if assoc_b64:
            arr_bytes = base64.b64decode(assoc_b64)
            obj.space._assoc_memory = (
                np.frombuffer(arr_bytes, dtype=np.int8).astype(np.float32)
            )

        vocab_str = data.get("vocabLabels", data.get("vocab_labels", ""))
        if vocab_str:
            for label in vocab_str.split("\n"):
                label = label.strip()
                if label:
                    obj.space.symbol(label)   # deterministic rebuild of item memory

        obj.stats = HDCLMStats(
            vocab_size=len(obj.space.item_memory),
            training_tokens=int(data.get("trainingTokens", data.get("training_tokens", 0))),
            training_docs=int(data.get("trainingDocs", data.get("training_docs", 0))),
            last_trained=data.get("lastTrained", data.get("last_trained", "")),
            created_at=data.get("createdAt", data.get("created_at", "")),
        )
        logger.info(
            "HDCLanguageModel restored from Convex: vocab=%d tokens=%d",
            len(obj.space.item_memory), obj.stats.training_tokens,
        )
        return obj

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps({
                "dim": self.dim,
                "context_size": self.context_size,
                "stats": self.stats.to_dict(),
            }),
            encoding="utf-8",
        )
        self.space.save(path)
        logger.info(
            "HDCLanguageModel saved: vocab=%d tokens=%d → %s",
            self.stats.vocab_size, self.stats.training_tokens, path,
        )

    @classmethod
    def load(cls, path: Path) -> "HDCLanguageModel":
        meta_path = path.with_suffix(".meta.json")
        dim = DEFAULT_DIM
        context_size = 5
        stats_dict: dict = {}
        if meta_path.is_file():
            meta_text = meta_path.read_text(encoding="utf-8").strip()
            if meta_text:
                meta = json.loads(meta_text)
                dim = int(meta.get("dim", dim))
                context_size = int(meta.get("context_size", context_size))
                stats_dict = meta.get("stats", {})

        obj = cls(dim=dim, context_size=context_size)
        if path.is_file():
            obj.space = HDCSpace.load(path)
        if stats_dict:
            obj.stats = HDCLMStats.from_dict(stats_dict)
        logger.info(
            "HDCLanguageModel loaded: vocab=%d tokens=%d from %s",
            len(obj.space.item_memory), obj.stats.training_tokens, path,
        )
        return obj

    @classmethod
    def load_or_new(cls, path: Path, **kwargs) -> "HDCLanguageModel":
        if path.is_file():
            try:
                return cls.load(path)
            except (ValueError, KeyError, json.JSONDecodeError, Exception) as exc:
                logger.warning(
                    "Failed to load HDC model from %s (%s) — starting fresh.",
                    path, exc,
                )
        return cls(**kwargs)

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HDCLanguageModel(dim={self.dim}, ctx={self.context_size}, "
            f"vocab={self.stats.vocab_size}, tokens={self.stats.training_tokens})"
        )
