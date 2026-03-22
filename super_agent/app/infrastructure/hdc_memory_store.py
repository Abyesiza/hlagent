from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from super_agent.app.domain.hdc import HDCSpace


def _fingerprint(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t[:240]


@dataclass
class AssociationRecord:
    task_fp: str
    solution_repr: str
    route: str


class HDCMemoryStore:
    """
    File-backed associative store using bundled hypervectors for retrieval.
    """

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
        for r in self._records:
            sim = self.space.cosine(q, self.space.symbol(r.task_fp))
            if sim > best_sim:
                best_sim = sim
                best = r.solution_repr
                best_fp = r.task_fp
        if best_sim < 0.15:
            return None, best_sim, best_fp
        return best, best_sim, best_fp

    def remember(self, task: str, solution_repr: str, route: str) -> None:
        fp = _fingerprint(task)
        self._records.append(AssociationRecord(task_fp=fp, solution_repr=solution_repr, route=route))
        t = self.space.symbol(fp)
        sol = self.space.symbol(solution_repr[:200])
        bound = self.space.bind(t, sol)
        if self._memory_hv is None:
            self._memory_hv = bound
        else:
            self._memory_hv = self.space.bundle([self._memory_hv, bound])
        self._save()
