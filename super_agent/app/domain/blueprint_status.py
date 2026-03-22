from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PhaseId(str, Enum):
    P1_ASYNC = "phase_1_async_backbone"
    P2_NEURO_SYMBOLIC = "phase_2_neuro_symbolic"
    P3_HDC = "phase_3_hdc_memory"
    P4_QUANTUM = "phase_4_quantum_inspired"
    P5_SICA = "phase_5_sica_sage"
    P6_HEARTBEAT = "phase_6_heartbeat_research"
    P7_DEPLOY = "phase_7_deploy_scale"


class WorkItem(BaseModel):
    id: str
    title: str
    status: Literal["done", "partial", "todo"]
    notes: str = ""


class BlueprintSnapshot(BaseModel):
    """
    Single source of truth for human + API: what is implemented vs still open.

    Bump `version` when phases change; clients compare to prior to drive evolution.
    """

    version: int = 2
    items: list[WorkItem] = Field(default_factory=list)

    def next_gaps(self) -> list[WorkItem]:
        return [i for i in self.items if i.status != "done"]


def default_blueprint() -> BlueprintSnapshot:
    return BlueprintSnapshot(
        version=2,
        items=[
            WorkItem(
                id=PhaseId.P1_ASYNC.value,
                title="Async backbone: FastAPI, BackgroundTasks, Gemini streaming",
                status="done",
                notes="POST /chat (sync), POST /agent/start (ThreadPoolExecutor jobs), "
                      "POST /chat/stream (SSE token streaming via generate_text_stream).",
            ),
            WorkItem(
                id=PhaseId.P2_NEURO_SYMBOLIC.value,
                title="Neuro-symbolic router + SymPy sandbox execution",
                status="done",
                notes="classify_intent → SYMBOLIC path: Gemini SymPy codegen → run_symcode "
                      "with retry + neural fallback. NEURAL path: plain or search-grounded Gemini.",
            ),
            WorkItem(
                id=PhaseId.P3_HDC.value,
                title="HDC memory (10k-dim bundling/binding, cosine retrieval)",
                status="done",
                notes="hdc_memory.json + orchestrator retrieve/remember + retrieval_count tracking. "
                      "torchhd GPU backend auto-detected (numpy fallback always active). "
                      "GET /api/v1/memory/list exposes stored records.",
            ),
            WorkItem(
                id=PhaseId.P4_QUANTUM.value,
                title="Quantum-inspired optimization (QAOA-style scoring)",
                status="done",
                notes="estimate_candidate_costs() produces real costs from benchmark score, "
                      "blueprint gap count, and HDC record count. Connected to SICA candidate "
                      "selection via pick_best_candidate_index(). score_summary() exposed on "
                      "GET /api/v1/sica/summary.",
            ),
            WorkItem(
                id=PhaseId.P5_SICA.value,
                title="SICA dual-loop: assess, plan JSON, AST liveness, git commit",
                status="done",
                notes="run_pytest_benchmark() runs the real test suite and returns pass-rate. "
                      "plan_improvements() reads live blueprint gaps. Benchmark history saved to "
                      "data/benchmark_history.jsonl. GET /api/v1/benchmark/history available.",
            ),
            WorkItem(
                id=PhaseId.P6_HEARTBEAT.value,
                title="Heartbeat + proactive research from HEARTBEAT.md",
                status="done",
                notes="APScheduler fires every 30 min (SUPER_AGENT_HEARTBEAT_INTERVAL_SECONDS). "
                      "Reads data/HEARTBEAT.md topics, writes findings to MEMORY.md. "
                      "GET/POST /api/v1/heartbeat/topics to read/update topics live.",
            ),
            WorkItem(
                id=PhaseId.P7_DEPLOY.value,
                title="Docker sandbox default + subprocess fallback + revert-to-stable",
                status="done",
                notes="run_sandbox() routes to Docker (--network=none, --memory=256m) when "
                      "SUPER_AGENT_ENABLE_DOCKER_SANDBOX=true, falls back to python -I subprocess. "
                      "POST /api/v1/sandbox/run. git revert via current_head() + git_commit_all().",
            ),
        ],
    )
