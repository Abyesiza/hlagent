"""Quantum-inspired optimization for SICA candidate selection."""
from __future__ import annotations

import math
from typing import Sequence


def qaoa_style_scores(costs: Sequence[float], layers: int = 2) -> list[float]:
    """
    Quantum-inspired mixing: map classical costs through trigonometric interference.

    Not a physical QAOA; provides diverse weighting over candidates for self-improvement.
    Lower output = better candidate.
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
    """Return the index of the best (lowest mixed score) candidate."""
    if not costs:
        return 0
    mixed = qaoa_style_scores(costs)
    return min(range(len(mixed)), key=lambda j: mixed[j])


def estimate_candidate_costs(
    benchmark_score: float,
    gap_count: int,
    hdc_record_count: int,
) -> list[float]:
    """
    Produce three real cost estimates for SICA candidate dimensions:
      [0] benchmark_cost   — how far the test suite is from 100% pass (0=perfect)
      [1] gap_density      — fraction of blueprint features still incomplete
      [2] memory_sparsity  — inverse of HDC memory richness (0=rich, 1=empty)

    These are fed to pick_best_candidate_index so SICA focuses on the weakest
    dimension when generating its next improvement candidate.
    """
    benchmark_cost = max(0.0, 1.0 - float(benchmark_score))
    gap_density = min(1.0, gap_count / 7.0)          # 7 total blueprint items
    memory_sparsity = max(0.0, 1.0 - min(1.0, hdc_record_count / 50.0))
    return [benchmark_cost, gap_density, memory_sparsity]


def score_summary(
    benchmark_score: float,
    gap_count: int,
    hdc_record_count: int,
) -> dict[str, object]:
    """Return a human-readable scoring dict for status display."""
    costs = estimate_candidate_costs(benchmark_score, gap_count, hdc_record_count)
    labels = ["benchmark", "blueprint_coverage", "hdc_richness"]
    mixed = qaoa_style_scores(costs)
    best = pick_best_candidate_index(costs)
    return {
        "dimensions": {labels[i]: {"cost": round(costs[i], 4), "mixed": round(mixed[i], 4)} for i in range(len(costs))},
        "focus_dimension": labels[best],
        "overall_health": round(1.0 - sum(costs) / len(costs), 4),
    }
