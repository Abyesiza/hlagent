from __future__ import annotations

import math
from typing import Sequence


def qaoa_style_scores(costs: Sequence[float], layers: int = 2) -> list[float]:
    """
    Quantum-inspired mixing: map classical costs through trigonometric interference.

    Not a physical QAOA; provides diverse weighting over candidates for self-improvement.
    """
    if not costs:
        return []
    n = len(costs)
    out: list[float] = []
    for i, c in enumerate(costs):
        theta = math.pi * (i + 1) / (n + 1)
        gamma = sum(math.cos((k + 1) * theta) * c for k in range(max(1, layers)))
        out.append(gamma)
    return out


def pick_best_candidate_index(costs: Sequence[float]) -> int:
    """Lower cost is better (minimization)."""
    mixed = qaoa_style_scores(costs)
    best_i = min(range(len(mixed)), key=lambda j: mixed[j])
    return best_i
