"""SICA dual-loop: real pytest benchmark + HDC + quantum candidate selection."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from super_agent.app.domain.hdc import HDCSpace, associate_task_solution
from super_agent.app.domain.quantum_inspired import estimate_candidate_costs, pick_best_candidate_index
from super_agent.app.infrastructure.ast_liveness import parse_ok
from super_agent.app.infrastructure.git_safe import current_head, git_commit_all, git_revert_to, git_log

if TYPE_CHECKING:
    from super_agent.app.services.orchestrator import SuperAgentOrchestrator
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore

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

def sica_step(
    repo: Path,
    target_file: Path,
    new_content: str,
    message: str,
    skip_benchmark: bool = True,
) -> dict[str, object]:
    """
    Inner loop: write file → AST liveness → git commit → HDC vector.
    By default, skips the pytest benchmark (skip_benchmark=True) so the hot-path
    stays fast. Pass skip_benchmark=False for scheduled/explicit SICA runs.
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

    if skip_benchmark:
        return {
            "ok": True,
            "committed": committed,
            "commit_msg": cmsg,
            "stable_hash": current_head(repo),
            "previous_hash": stable_before,
            "hdc_vector_norm": float((vec * vec).sum() ** 0.5),
        }

    # Full benchmark path (used by scheduled SICA outer loop)
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


# ── non-Python file write (TypeScript, Markdown, etc.) ───────────────────────

def write_non_python_file(
    repo: Path,
    target_file: Path,
    new_content: str,
    message: str,
) -> dict[str, object]:
    """
    Write any non-Python file and commit it.
    Skips Python AST check — callers are responsible for basic content validation.
    Returns the same dict shape as sica_step for consistency.
    """
    stable_before = current_head(repo)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(new_content, encoding="utf-8")
    committed, cmsg = git_commit_all(repo, message)
    return {
        "ok": True,
        "committed": committed,
        "commit_msg": cmsg,
        "stable_hash": current_head(repo),
        "previous_hash": stable_before,
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


# ── full SICA improvement cycle ───────────────────────────────────────────────

_REGRESSION_THRESHOLD = 0.10  # rollback if score drops more than 10%


def run_improvement_cycle(
    orchestrator: "SuperAgentOrchestrator",
    data_dir: Path,
    *,
    gap_id: str | None = None,
    convex: "ConvexAgentStore | None" = None,
) -> dict[str, object]:
    """
    Full SICA cycle:
      1. Run benchmark to get baseline
      2. Plan improvements from blueprint gaps
      3. Pick the top-priority gap (or a specific gap_id)
      4. Apply the improvement via orchestrator.improve_self
      5. Re-run benchmark to verify no regression
      6. Roll back if tests regressed by more than _REGRESSION_THRESHOLD
      7. Record everything to data_dir/benchmark_history.jsonl

    Returns a rich status dict suitable for API responses and Convex storage.
    """
    repo = data_dir.parent
    started_at = datetime.now(UTC).isoformat()

    # 1. Baseline benchmark
    bench_before = run_pytest_benchmark(repo)
    _save_benchmark(data_dir, bench_before)

    stable_hash = current_head(repo)

    # 2. Plan
    plan = plan_improvements(bench_before)
    todos = plan.get("todos", [])
    if not todos:
        return {
            "status": "no_gaps",
            "message": "Blueprint has no open gaps — nothing to do.",
            "plan": plan,
            "started_at": started_at,
        }

    # 3. Pick target gap
    chosen: dict[str, object] | None = None
    if gap_id:
        chosen = next((t for t in todos if t.get("id") == gap_id), None)
    if chosen is None:
        chosen = todos[0]

    instruction = (
        f"Implement the following blueprint item: {chosen['task']}. "
        f"Notes: {chosen.get('notes', '')}"
    )

    logger.info("SICA cycle: applying gap '%s' — %s", chosen.get("id"), chosen.get("task"))

    # 4. Apply improvement
    try:
        result = orchestrator.improve_self(instruction)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SICA improve_self raised")
        return {
            "status": "error",
            "message": str(exc),
            "gap": chosen,
            "started_at": started_at,
            "bench_before": bench_before,
        }

    if not result.ok:
        return {
            "status": "improve_failed",
            "message": result.error or "improve_self returned ok=False",
            "gap": chosen,
            "result": result.model_dump(),
            "started_at": started_at,
            "bench_before": bench_before,
        }

    # 5. Post-improvement benchmark
    bench_after = run_pytest_benchmark(repo)
    _save_benchmark(data_dir, bench_after)

    score_before = float(bench_before.get("score", 1.0))
    score_after = float(bench_after.get("score", 1.0))
    regression = round(score_before - score_after, 4)

    # 6. Rollback if tests regressed
    reverted = False
    revert_msg = ""
    if regression > _REGRESSION_THRESHOLD and stable_hash:
        ok_rev, revert_msg = git_revert_to(repo, stable_hash)
        reverted = ok_rev
        logger.warning(
            "SICA: score dropped %.2f → %.2f (regression %.2f > threshold %.2f), reverted=%s",
            score_before, score_after, regression, _REGRESSION_THRESHOLD, reverted,
        )

    # 7. Persist to Convex if available
    cycle_record = {
        "status": "reverted" if reverted else "applied",
        "gap": chosen,
        "result": result.model_dump(),
        "bench_before": bench_before,
        "bench_after": bench_after,
        "score_delta": round(score_after - score_before, 4),
        "regression": regression,
        "reverted": reverted,
        "revert_msg": revert_msg,
        "stable_hash": stable_hash or "",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }

    if convex:
        try:
            convex.create_task(
                title=f"SICA: {chosen['task'][:80]}",
                task_type="sica_cycle",
                status="reverted" if reverted else "done",
                metadata=json.dumps({"gap_id": chosen.get("id"), "score_delta": cycle_record["score_delta"]}),
            )
        except Exception:  # noqa: BLE001
            logger.warning("SICA: failed to write cycle record to Convex", exc_info=True)

    return cycle_record


# ── feature request → improve ─────────────────────────────────────────────────

_FEATURE_REQUEST_RE = re.compile(
    r"\b(build|add|create|implement|make|develop|give me|i need|i want|can you add|"
    r"please add|feature|endpoint|button|tab|page|component|function|api)\b",
    re.IGNORECASE,
)


def looks_like_feature_request(text: str) -> bool:
    """Return True if the user's message appears to be requesting a new feature / code change."""
    return bool(_FEATURE_REQUEST_RE.search(text))


def describe_gap_for_chat(gap: dict[str, object]) -> str:
    """Format a gap dict into a readable one-liner for embedding in a chat response."""
    return f"**Blueprint gap [{gap.get('id')}]:** {gap.get('task', '')} — {gap.get('notes', '')}"
