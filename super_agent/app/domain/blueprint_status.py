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

    version: int = 1
    items: list[WorkItem] = Field(default_factory=list)

    def next_gaps(self) -> list[WorkItem]:
        return [i for i in self.items if i.status != "done"]


def default_blueprint() -> BlueprintSnapshot:
    return BlueprintSnapshot(
        version=1,
        items=[
            WorkItem(
                id=PhaseId.P1_ASYNC.value,
                title="Async backbone: FastAPI, BackgroundTasks, Gemini wrapper",
                status="partial",
                notes="Orchestrator + /api/v1/chat wired; Interactions API polling + Redis/ARQ still open.",
            ),
            WorkItem(
                id=PhaseId.P2_NEURO_SYMBOLIC.value,
                title="Neuro-symbolic router + SymPy sandbox execution",
                status="partial",
                notes="End-to-end: intent → Gemini SymPy codegen → run_symcode; add Z3 / richer NL→SymPy.",
            ),
            WorkItem(
                id=PhaseId.P3_HDC.value,
                title="HDC memory (10k-dim bundling/binding, cosine retrieval)",
                status="partial",
                notes="hdc_memory.json + orchestrator retrieve/remember; optional torchhd extra for GPU.",
            ),
            WorkItem(
                id=PhaseId.P4_QUANTUM.value,
                title="Quantum-inspired optimization (QAOA-style scoring)",
                status="partial",
                notes="Classical simulation layer; plug into SICA candidate selection.",
            ),
            WorkItem(
                id=PhaseId.P5_SICA.value,
                title="SICA dual-loop: assess, plan JSON, AST liveness, git commit",
                status="partial",
                notes="Hooks stubbed; SWE-bench harness not included.",
            ),
            WorkItem(
                id=PhaseId.P6_HEARTBEAT.value,
                title="Heartbeat + proactive research from HEARTBEAT.md",
                status="partial",
                notes="APScheduler; shorten interval via SUPER_AGENT_HEARTBEAT_INTERVAL_SECONDS in dev.",
            ),
            WorkItem(
                id=PhaseId.P7_DEPLOY.value,
                title="Docker sandbox default + revert-to-stable-git-hash",
                status="todo",
                notes="Sandbox opt-in; auto-revert needs tagged stable + CI.",
            ),
        ],
    )
