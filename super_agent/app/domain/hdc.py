from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _seed_bytes(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


@dataclass
class HDCSpace:
    """
    Minimal Vector Symbolic Architecture in NumPy (D=10_000).

    Binding: element-wise multiply + normalize
    Bundling: sum + normalize
    """

    dim: int = 10_000
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(42)

    def random_hv(self) -> np.ndarray:
        v = self.rng.standard_normal(self.dim).astype(np.float64)
        return self._normalize(v)

    def symbol(self, label: str) -> np.ndarray:
        """Reproducible pseudo-random HDV per label."""
        seed = int.from_bytes(_seed_bytes(label)[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float64)
        return self._normalize(v)

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return HDCSpace._normalize(a * b)

    @staticmethod
    def bundle(vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("bundle requires at least one vector")
        s = np.sum(np.stack(vectors, axis=0), axis=0)
        return HDCSpace._normalize(s)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-12:
            return v
        return (v / n).astype(np.float64)


def associate_task_solution(task_label: str, solution_label: str, space: HDCSpace | None = None) -> np.ndarray:
    s = space or HDCSpace()
    t = s.symbol(task_label)
    sol = s.symbol(solution_label)
    return s.bind(t, sol)


def retrieve_best_match(query_label: str, memory_keys: list[str], space: HDCSpace | None = None) -> tuple[str, float]:
    s = space or HDCSpace()
    q = s.symbol(query_label)
    best_k = memory_keys[0]
    best_sim = -1.0
    for k in memory_keys:
        sim = s.cosine(q, s.symbol(k))
        if sim > best_sim:
            best_sim = sim
            best_k = k
    return best_k, best_sim
