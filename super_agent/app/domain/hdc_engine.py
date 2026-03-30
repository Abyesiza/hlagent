"""
HDC Engine — Vector Symbolic Architecture (VSA) core.

Implements Fourier Holographic Reduced Representations (FHRR) using bipolar
hypervectors (±1) for maximum algebraic clarity and NumPy efficiency.

Core algebra:
  - bind(a, b)      : element-wise multiplication  (⊗)
  - bundle(vectors) : majority vote / thresholded sum (⊕)
  - permute(v, n)   : cyclic roll by n positions (Π)  — encodes position
  - similarity(a,b) : cosine distance ∈ [-1, 1]

The Law-of-Large-Numbers magic:
  Any two random D-dimensional bipolar vectors are nearly orthogonal with
  probability → 1 as D → ∞. This "background noise floor" shrinks as 1/√D,
  giving D=10 000 roughly 0.01 expected similarity between unrelated vectors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default dimensionality — increasing D gives more capacity but uses more RAM.
DEFAULT_DIM = 10_000


# ── helpers ──────────────────────────────────────────────────────────────────


def _stable_seed(label: str) -> int:
    """Deterministic 64-bit seed from a label string via SHA-256."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bipolar(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Random bipolar (±1) hypervector."""
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)


# ── core VSA operations ───────────────────────────────────────────────────────


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Binding (⊗): element-wise multiplication.

    Properties:
    - Commutative: bind(a,b) = bind(b,a)
    - Self-inverse: bind(bind(a,b), a) ≈ b  (exact for bipolar)
    - Dissimilar to both operands (perfect for key-value pairs)
    """
    return (a * b).astype(np.float32)


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Retrieve value: since bind is self-inverse, unbind == bind."""
    return bind(bound, key)


def bundle(vectors: list[np.ndarray], threshold: float = 0.0) -> np.ndarray:
    """
    Bundling (⊕): majority vote over stacked bipolar vectors.

    The result is similar to all inputs but identical to none —
    a holistic "superposition" of concepts.
    Tie-breaking uses a random ±1 flip per tied dimension.
    """
    if not vectors:
        raise ValueError("bundle() requires at least one vector")
    stacked = np.stack(vectors, axis=0)          # (N, D)
    summed = stacked.sum(axis=0, dtype=np.float32)

    # Tie resolution: randomly break zeros with a full-size random mask
    ties = summed == 0.0
    if np.any(ties):
        random_flips = np.random.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=len(summed)
        )
        summed = np.where(ties, random_flips, summed)

    return np.sign(summed).astype(np.float32)


def permute(v: np.ndarray, n: int) -> np.ndarray:
    """
    Permutation (Π): cyclic shift by n positions — encodes order/position.

    permute(permute(v, 1), 1) ≠ v  (unlike bind, which is self-inverse)
    This asymmetry is what makes sequences representable: "CAT" ≠ "ACT".
    """
    return np.roll(v, n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised Hamming similarity for bipolar vectors ∈ [0, 1].
    Faster than cosine for large batches.
    """
    return float(np.mean(a == b))


# ── Item Memory (vocabulary / clean-up memory) ───────────────────────────────


class ItemMemory:
    """
    A dictionary of label → hypervector pairs.

    Used as:
    1. Vocabulary lookup (word → hypervector)
    2. Nearest-neighbour clean-up (noisy vector → cleanest matching label)
    """

    # Initial capacity; doubles when full (amortised O(1) appends)
    _INIT_CAPACITY = 1024

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim
        self._labels: list[str] = []
        self._label_index: dict[str, int] = {}   # O(1) label→index lookup
        # Pre-allocated matrix: first `_size` rows are valid data
        self._buf: np.ndarray = np.empty((self._INIT_CAPACITY, dim), dtype=np.float32)
        self._size: int = 0                      # number of valid rows
        self._unit_mat: np.ndarray | None = None # cached row-normalised matrix (snapshot)
        self._unit_labels: list[str] = []        # labels matching _unit_mat snapshot
        self._unit_size: int = 0                 # vocab size when _unit_mat was built
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    def __contains__(self, label: str) -> bool:
        return label in self._label_index

    def add(self, label: str, hv: np.ndarray) -> None:
        """
        Add a new item. Silently skips if label already exists.
        Uses a pre-allocated buffer that doubles on overflow — O(1) amortised.
        """
        with self._lock:
            if label in self._label_index:
                return
            idx = self._size
            self._labels.append(label)
            self._label_index[label] = idx
            # Grow buffer by 2× when full (amortised O(1))
            if idx >= len(self._buf):
                new_cap = max(len(self._buf) * 2, idx + 1)
                new_buf = np.empty((new_cap, self.dim), dtype=np.float32)
                new_buf[:self._size] = self._buf[:self._size]
                self._buf = new_buf
            self._buf[idx] = hv.astype(np.float32)
            self._size += 1
            # unit_mat will be rebuilt lazily (size mismatch detected in nearest())

    def get(self, label: str) -> np.ndarray | None:
        with self._lock:
            idx = self._label_index.get(label)
            if idx is None:
                return None
            return self._buf[idx].copy()

    def nearest(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-k (label, similarity) pairs for a query hypervector.

        Lock strategy: take a *snapshot* of the data slice under a brief lock,
        then do all heavy math outside the lock so training is never blocked.
        The normalised matrix is cached and only rebuilt when the vocab grows.
        """
        # ── Step 1: grab references under a brief lock (no data copies) ─────
        with self._lock:
            current_size = self._size
            if current_size == 0:
                return []
            # Reuse cached unit_mat when vocab size hasn't changed
            if self._unit_mat is not None and self._unit_size == current_size:
                return self._do_search(self._unit_mat, self._unit_labels, query, top_k)
            # Take a reference to the buffer — the array data won't be freed
            # while we hold this reference even if add() swaps self._buf.
            buf_ref = self._buf
            labels_snap = list(self._labels)  # fast — just copies string refs

        # ── Step 2: copy + normalise OUTSIDE the lock ────────────────────────
        mat_snap = buf_ref[:current_size].copy()   # ~578 MB copy, no lock held
        norms = np.linalg.norm(mat_snap, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        unit_snap = (mat_snap / norms).astype(np.float32)
        del mat_snap  # free immediately

        # Cache for future calls
        with self._lock:
            if self._unit_size != current_size:   # only write if still valid
                self._unit_mat = unit_snap
                self._unit_labels = labels_snap
                self._unit_size = current_size

        return self._do_search(unit_snap, labels_snap, query, top_k)

    @staticmethod
    def _do_search(
        unit_mat: np.ndarray,
        labels: list[str],
        query: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Cosine similarity search against a pre-normalised matrix."""
        qn = np.linalg.norm(query)
        if qn < 1e-12:
            return []
        q_unit = (query / qn).astype(np.float32)
        sims = unit_mat @ q_unit   # (N,)
        k = min(top_k, len(labels))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(labels[i], float(sims[i])) for i in top_idx]

    def labels(self) -> list[str]:
        with self._lock:
            return list(self._labels)

    def to_dict(self) -> dict:
        with self._lock:
            if self._size == 0:
                return {"labels": [], "matrix": []}
            return {
                "labels": self._labels,
                "matrix": self._buf[:self._size].tolist(),
            }

    @classmethod
    def from_dict(cls, data: dict, dim: int = DEFAULT_DIM) -> "ItemMemory":
        obj = cls(dim=dim)
        labels = data.get("labels", [])
        matrix = data.get("matrix", [])
        if labels and matrix:
            n = len(labels)
            obj._labels = list(labels)
            obj._label_index = {lbl: i for i, lbl in enumerate(obj._labels)}
            # Allocate exactly the right capacity (+ 25% headroom for growth)
            capacity = max(cls._INIT_CAPACITY, int(n * 1.25))
            obj._buf = np.empty((capacity, dim), dtype=np.float32)
            obj._buf[:n] = np.array(matrix, dtype=np.float32)
            obj._size = n
            # _unit_mat will be built lazily on first nearest() call
        return obj


# ── HDC Space (hypervector factory) ──────────────────────────────────────────


@dataclass
class HDCSpace:
    """
    Factory + registry for hypervectors.

    Provides:
    - Deterministic symbol() generation per label
    - Bundled Associative Memory with incremental update
    - All core VSA operations as instance methods
    """

    dim: int = DEFAULT_DIM
    item_memory: ItemMemory = field(default_factory=lambda: ItemMemory(DEFAULT_DIM))

    def __post_init__(self) -> None:
        self.item_memory = ItemMemory(dim=self.dim)
        self._assoc_memory: np.ndarray | None = None
        self._assoc_count: int = 0

    # ── symbol generation ────────────────────────────────────────────────────

    def symbol(self, label: str) -> np.ndarray:
        """
        Reproducible random bipolar hypervector for a label.
        Auto-registers in the ItemMemory for nearest-neighbour lookup.
        """
        if label in self.item_memory:
            return self.item_memory.get(label)  # type: ignore[return-value]
        seed = _stable_seed(label)
        rng = np.random.default_rng(seed)
        hv = _bipolar(rng, self.dim)
        self.item_memory.add(label, hv)
        return hv

    # ── algebraic operations (delegates to module functions) ─────────────────

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return bind(a, b)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        return unbind(bound, key)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        return bundle(vectors)

    def permute(self, v: np.ndarray, n: int) -> np.ndarray:
        return permute(v, n)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return cosine_sim(a, b)

    # ── context encoding ─────────────────────────────────────────────────────

    def encode_ngram(self, tokens: list[str]) -> np.ndarray:
        """
        Encode a token sequence as a single context hypervector.

        Each token is permuted by its distance from the "current" slot
        (token[-1] has shift 1, token[-2] has shift 2, etc.), then bundled.
        This gives positional sensitivity without losing algebraic structure.
        """
        if not tokens:
            return _bipolar(np.random.default_rng(0), self.dim)
        hvs: list[np.ndarray] = []
        n = len(tokens)
        for i, tok in enumerate(tokens):
            shift = n - i        # tokens near the end have smaller shifts
            hv = self.permute(self.symbol(tok), shift)
            hvs.append(hv)
        return self.bundle(hvs)

    # ── associative memory ───────────────────────────────────────────────────

    def remember(self, key_hv: np.ndarray, value_hv: np.ndarray) -> None:
        """
        Bundle a key⊗value pair into the associative memory.
        This is the one-shot learning step.
        """
        pair = self.bind(key_hv, value_hv)
        if self._assoc_memory is None:
            self._assoc_memory = pair.copy()
        else:
            self._assoc_memory = bundle([self._assoc_memory, pair])
        self._assoc_count += 1

    def recall(self, key_hv: np.ndarray) -> np.ndarray | None:
        """
        Retrieve the (noisy) value for a key by unbinding from assoc memory.
        Use item_memory.nearest() to clean up the result.
        """
        if self._assoc_memory is None:
            return None
        return self.unbind(self._assoc_memory, key_hv)

    @property
    def assoc_memory_size(self) -> int:
        return self._assoc_count

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "assoc_count": self._assoc_count,
            "assoc_memory": self._assoc_memory.tolist() if self._assoc_memory is not None else None,
            "item_memory": self.item_memory.to_dict(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("HDCSpace saved: dim=%d items=%d assoc=%d → %s",
                    self.dim, len(self.item_memory), self._assoc_count, path)

    @classmethod
    def load(cls, path: Path) -> "HDCSpace":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"HDCSpace model file is empty: {path}")
        raw = json.loads(text)
        dim = int(raw.get("dim", DEFAULT_DIM))
        obj = cls(dim=dim)
        obj._assoc_count = int(raw.get("assoc_count", 0))
        am = raw.get("assoc_memory")
        obj._assoc_memory = np.array(am, dtype=np.float32) if am is not None else None
        im_data = raw.get("item_memory", {})
        obj.item_memory = ItemMemory.from_dict(im_data, dim=dim)
        logger.info("HDCSpace loaded: dim=%d items=%d assoc=%d from %s",
                    dim, len(obj.item_memory), obj._assoc_count, path)
        return obj
