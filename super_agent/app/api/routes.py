from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel

from super_agent.app.api.deps import AppContainer
from super_agent.app.domain.blueprint_status import BlueprintSnapshot, default_blueprint
from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.domain.math_schemas import SymCodeRequest
from super_agent.app.infrastructure.intent_router import classify_intent
from super_agent.app.infrastructure.sympy_runner import run_symcode

router = APIRouter(prefix="/api/v1")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


@router.get("/status", response_model=BlueprintSnapshot)
def blueprint_status() -> BlueprintSnapshot:
    return default_blueprint()


@router.get("/gaps")
def next_gaps() -> dict[str, object]:
    bp = default_blueprint()
    return {"version": bp.version, "gaps": [i.model_dump() for i in bp.next_gaps()]}


class ChatRequest(BaseModel):
    message: str = ""
    prompt: str = ""
    session_id: str | None = None


class ImproveRequestBody(BaseModel):
    instruction: str
    target_file: str | None = None


@router.post("/chat", response_model=ChatTurnResult)
def chat_sync(body: ChatRequest, c: ContainerDep) -> ChatTurnResult:
    message = body.message or body.prompt
    result = c.orchestrator.run(message, session_id=body.session_id)
    result.grounded = result.metadata.get("grounded", False)  # type: ignore[assignment]
    result.session_id = body.session_id
    return result


@router.post("/agent/start")
def start_agent(body: ChatRequest, c: ContainerDep) -> dict[str, str]:
    prompt = body.message or body.prompt
    job_id = c.agent_loop.start_turn(prompt, session_id=body.session_id)
    return {"job_id": job_id, "status": "accepted"}


@router.get("/agent/jobs/{job_id}")
def get_job(job_id: str, c: ContainerDep) -> dict[str, object]:
    job = c.agent_loop.get_job(job_id)
    if not job:
        return {"error": "not found"}
    out: dict[str, object] = {
        "job_id": job.job_id,
        "phase": job.state.phase.value,
        "result": job.turn.model_dump() if job.turn else None,
        "error": job.error,
    }
    return out


@router.post("/sympy/run")
def sympy_run(req: SymCodeRequest) -> dict[str, object]:
    res = run_symcode(req)
    return res.model_dump()


@router.post("/route-intent")
def route_intent(body: dict[str, str]) -> dict[str, str]:
    text = body.get("text") or ""
    return {"intent": classify_intent(text).value}


@router.get("/codebase/snapshot")
def codebase_snapshot(c: ContainerDep) -> dict[str, str]:
    from super_agent.app.services.workspace_context import load_codebase_snapshot
    content = load_codebase_snapshot(c.settings.data_dir)
    return {"content": content or "(not generated yet — POST /api/v1/codebase/refresh)"}


@router.post("/codebase/refresh")
def codebase_refresh(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services.codebase_scanner import refresh_codebase_md
    out = refresh_codebase_md(c.settings.data_dir)
    size = out.stat().st_size if out.is_file() else 0
    return {"status": "ok", "path": str(out), "bytes": size}


@router.post("/improve", response_model=None)
def request_improvement(body: ImproveRequestBody, c: ContainerDep) -> dict[str, object]:
    from super_agent.app.domain.chat_schemas import ImproveRequest
    req = ImproveRequest(instruction=body.instruction, target_file=body.target_file)
    result = c.orchestrator.improve_self(req.instruction, req.target_file)
    return result.model_dump()


@router.get("/improve/history")
def improvement_history(c: ContainerDep, limit: int = 20) -> dict[str, object]:
    from super_agent.app.domain.chat_schemas import ImproveResult
    history_file = c.settings.data_dir / "improvements.jsonl"
    if not history_file.is_file():
        return {"entries": []}
    import json
    lines = history_file.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"entries": list(reversed(entries))}


@router.post("/research/trigger")
def research_trigger(background_tasks: BackgroundTasks, c: ContainerDep) -> dict[str, str]:
    from super_agent.app.services import research_loop

    def _run() -> None:
        research_loop.run_proactive_research(c.gemini, c.settings.data_dir)

    background_tasks.add_task(_run)
    return {"status": "scheduled"}


@router.get("/sica/summary")
def sica_summary(c: ContainerDep) -> dict[str, str]:
    from super_agent.app.services import sica_loop

    summary = sica_loop.run_outer_loop_summary(c.settings.data_dir.parent)
    return {"summary": summary}
