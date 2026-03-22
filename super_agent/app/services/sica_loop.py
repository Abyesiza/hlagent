"""SICA dual-loop: real pytest benchmark + HDC + quantum candidate selection."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from super_agent.app.domain.hdc import HDCSpace, associate_task_solution
from super_agent.app.domain.quantum_inspired import estimate_candidate_costs, pick_best_candidate_index
from super_agent.app.infrastructure.ast_liveness import parse_ok
from super_agent.app.infrastructure.git_safe import current_head, git_commit_all

logger = logging.getLogger(__name__)

_HISTORY_FILE = "benchmark_history.jsonl"


# ── real benchmark ────────────────────────────────────────────────────────────

def run_pytest_benchmark(repo: Path) -> dict[str, object]:
    """
    Run the test suite under `repo/tests/` and return a score dict.
    Score is the pass-rate in [0, 1].
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(repo / "tests"), "-q", "--tb=no", "--no-header"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout + result.stderr
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
        total = passed + failed + errors
        score = passed / max(total, 1)
        return {
            "score": round(score, 4),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": total,
            "output": out[:1500],
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "passed": 0, "failed": 0, "errors": 0, "total": 0, "error": "pytest timeout"}
    except Exception as e:
        logger.exception("benchmark run failed")
        return {"score": 0.0, "passed": 0, "failed": 0, "errors": 0, "total": 0, "error": str(e)}


def _save_benchmark(data_dir: Path, bench: dict[str, object]) -> None:
    out = data_dir / _HISTORY_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(bench) + "\n")


def load_benchmark_history(data_dir: Path, limit: int = 20) -> list[dict[str, object]]:
    path = data_dir / _HISTORY_FILE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return list(reversed(out))


# ── plan generation from real blueprint gaps ──────────────────────────────────

def plan_improvements(bench: dict[str, object]) -> dict[str, object]:
    """
    Build a real improvement plan from live blueprint gaps and benchmark score.
    Uses quantum-inspired scoring to rank candidates.
    """
    from super_agent.app.domain.blueprint_status import default_blueprint

    bp = default_blueprint()
    gaps = [
        {"id": i.id, "title": i.title, "status": i.status, "notes": i.notes}
        for i in bp.next_gaps()
    ]

    score = float(bench.get("score", 0.0))
    costs = estimate_candidate_costs(
        benchmark_score=score,
        gap_count=len(gaps),
        hdc_record_count=0,
    )
    best_idx = pick_best_candidate_index(costs) if costs else 0
    priority_dim = ["benchmark", "blueprint_coverage", "hdc_richness"][min(best_idx, 2)]

    return {
        "score": score,
        "benchmark": bench,
        "priority_dimension": priority_dim,
        "gaps": gaps,
        "todos": [
            {
                "id": g["id"],
                "task": f"Complete: {g['title']}",
                "priority": 1 if g["status"] == "todo" else 2,
                "notes": g["notes"],
            }
            for g in gaps
        ],
    }


# ── SICA inner step (write → AST → commit → HDC) ─────────────────────────────

def sica_step(repo: Path, target_file: Path, new_content: str, message: str) -> dict[str, object]:
    """
    Inner loop: write file → AST liveness → git commit → HDC vector → quantum pick.
    Returns a result dict with ok, committed, stable_hash, hdc_vector_norm.
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

    # Real costs from this step's context
    bench = run_pytest_benchmark(repo)
    bp = __import__(
        "super_agent.app.domain.blueprint_status", fromlist=["default_blueprint"]
    ).default_blueprint()
    costs = estimate_candidate_costs(
        benchmark_score=float(bench.get("score", 0.5)),
        gap_count=len(list(bp.next_gaps())),
        hdc_record_count=1,
    )
    best_candidate = pick_best_candidate_index(costs)

    return {
        "ok": True,
        "committed": committed,
        "commit_msg": cmsg,
        "stable_hash": current_head(repo),
        "previous_hash": stable_before,
        "hdc_vector_norm": float((vec * vec).sum() ** 0.5),
        "quantum_candidate_pick": best_candidate,
        "post_step_score": bench.get("score"),
    }


# ── outer loop summary ────────────────────────────────────────────────────────

def run_outer_loop_summary(repo: Path) -> str:
    """
    Run benchmark → plan improvements → save history → return JSON summary.
    This is the real SICA outer loop assessment.
    """
    bench = run_pytest_benchmark(repo)
    data_dir = repo / "data"
    _save_benchmark(data_dir, bench)
    plan = plan_improvements(bench)
    return json.dumps(plan, indent=2)
