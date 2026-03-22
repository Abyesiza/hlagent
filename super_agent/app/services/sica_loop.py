from __future__ import annotations

import json
import logging
from pathlib import Path

from super_agent.app.domain.hdc import HDCSpace, associate_task_solution
from super_agent.app.domain.quantum_inspired import pick_best_candidate_index
from super_agent.app.infrastructure.ast_liveness import parse_ok
from super_agent.app.infrastructure.git_safe import current_head, git_commit_all

logger = logging.getLogger(__name__)


def local_benchmark_stub() -> float:
    """Placeholder score in [0,1]; replace with pytest subset / SWE-bench harness."""
    return 0.42


def plan_improvements(score: float) -> dict[str, object]:
    return {
        "score": score,
        "todos": [
            {"id": "t1", "task": "raise benchmark score", "priority": 1},
            {"id": "t2", "task": "tighten sympy sandbox", "priority": 2},
        ],
    }


def sica_step(repo: Path, target_file: Path, new_content: str, message: str) -> dict[str, object]:
    """
    Inner loop: write file → AST liveness → optional git commit → HDC success vector (stub).
    """
    stable_before = current_head(repo)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(new_content, encoding="utf-8")
    ok, err = parse_ok(target_file)
    if not ok:
        return {"ok": False, "phase": "ast", "error": err, "stable_hash": stable_before}

    committed, cmsg = git_commit_all(repo, message)
    hdc = HDCSpace()
    vec = associate_task_solution(str(target_file), message[:80], hdc)

    costs = [0.9, 0.5, 0.3]  # placeholder candidate costs
    best = pick_best_candidate_index(costs)

    return {
        "ok": True,
        "committed": committed,
        "commit_msg": cmsg,
        "stable_hash": current_head(repo),
        "previous_hash": stable_before,
        "hdc_vector_norm": float((vec * vec).sum() ** 0.5),
        "quantum_candidate_pick": best,
    }


def run_outer_loop_summary(repo: Path) -> str:
    score = local_benchmark_stub()
    plan = plan_improvements(score)
    return json.dumps(plan, indent=2)
