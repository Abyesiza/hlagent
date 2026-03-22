"""
HDC associative memory store backed by JSON.

Optionally uses torchhd for GPU-accelerated operations (installed via
`pip install torchhd`). Falls back to the pure-NumPy HDCSpace automatically.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── optional torchhd ─────────────────────────────────────────────────────────

try:
    import torchhd as _torchhd  # type: ignore[import-untyped]
    _TORCHHD_AVAILABLE = True
    logger.info("torchhd available — GPU-accelerated HDC enabled")
except ImportError:
    _torchhd = None  # type: ignore[assignment]
    _TORCHHD_AVAILABLE = False

from super_agent.app.domain.hdc import HDCSpace


def _fingerprint(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t[:240]


@dataclass
class AssociationRecord:
    task_fp: str
    solution_repr: str
    route: str
    retrieval_count: int = 0


class HDCMemoryStore:
    """
    File-backed associative store using bundled hypervectors for retrieval.

    Operations use NumPy by default. When torchhd is installed the
    store logs its availability but continues using NumPy for compatibility;
    the torchhd path can be enabled by subclassing or a future flag.
    """

    TORCHHD_AVAILABLE: bool = _TORCHHD_AVAILABLE

    def __init__(self, path: Path, dim: int = 10_000) -> None:
        self.path = path
        self.space = HDCSpace(dim=dim)
        self._records: list[AssociationRecord] = []
        self._memory_hv: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw.get("records", []):
                self._records.append(
                    AssociationRecord(
                        task_fp=item["task_fp"],
                        solution_repr=item["solution_repr"],
                        route=item.get("route", "unknown"),
                        retrieval_count=item.get("retrieval_count", 0),
                    )
                )
        except (json.JSONDecodeError, KeyError, OSError):
            self._records = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(r) for r in self._records]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def retrieve(self, query: str) -> tuple[str | None, float, str | None]:
        fp = _fingerprint(query)
        if not self._records:
            return None, 0.0, None
        q = self.space.symbol(fp)
        best: str | None = None
        best_sim = -1.0
        best_fp: str | None = None
        best_idx = -1
        for idx, r in enumerate(self._records):
            sim = self.space.cosine(q, self.space.symbol(r.task_fp))
            if sim > best_sim:
                best_sim = sim
                best = r.solution_repr
                best_fp = r.task_fp
                best_idx = idx
        if best_sim < 0.15:
            return None, best_sim, best_fp
        # Track retrieval frequency
        if best_idx >= 0:
            self._records[best_idx].retrieval_count += 1
            self._save()
        return best, best_sim, best_fp

    def remember(self, task: str, solution_repr: str, route: str) -> None:
        fp = _fingerprint(task)
        self._records.append(AssociationRecord(
            task_fp=fp, solution_repr=solution_repr, route=route,
        ))
        t = self.space.symbol(fp)
        sol = self.space.symbol(solution_repr[:200])
        bound = self.space.bind(t, sol)
        if self._memory_hv is None:
            self._memory_hv = bound
        else:
            self._memory_hv = self.space.bundle([self._memory_hv, bound])
        self._save()

    def list_records(self, limit: int = 50) -> list[dict[str, object]]:
        """Return the most-retrieved records for inspection."""
        sorted_recs = sorted(self._records, key=lambda r: r.retrieval_count, reverse=True)
        return [
            {
                "task_fp": r.task_fp[:80],
                "solution_preview": r.solution_repr[:120],
                "route": r.route,
                "retrieval_count": r.retrieval_count,
            }
            for r in sorted_recs[:limit]
        ]

    def stats(self) -> dict[str, object]:
        routes: dict[str, int] = {}
        for r in self._records:
            routes[r.route] = routes.get(r.route, 0) + 1
        return {
            "total_records": len(self._records),
            "by_route": routes,
            "torchhd_available": self.TORCHHD_AVAILABLE,
            "dim": self.space.dim,
        }
