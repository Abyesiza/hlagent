from __future__ import annotations

import asyncio
import json
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path

from super_agent.app.api.deps import AppContainer
from super_agent.app.domain.blueprint_status import BlueprintSnapshot, default_blueprint
from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.domain.math_schemas import SymCodeRequest
from super_agent.app.infrastructure.intent_router import classify_intent
from super_agent.app.infrastructure.sympy_runner import run_symcode

router = APIRouter(prefix="/api/v1")

_stream_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sse")


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


ContainerDep = Annotated[AppContainer, Depends(get_container)]


# ── blueprint ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=BlueprintSnapshot)
def blueprint_status() -> BlueprintSnapshot:
    return default_blueprint()


@router.get("/gaps")
def next_gaps() -> dict[str, object]:
    bp = default_blueprint()
    return {"version": bp.version, "gaps": [i.model_dump() for i in bp.next_gaps()]}


# ── chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = ""
    prompt: str = ""
    session_id: str | None = None
    auto_improve: bool = False


@router.post("/chat", response_model=ChatTurnResult)
def chat_sync(body: ChatRequest, c: ContainerDep) -> ChatTurnResult:
    message = body.message or body.prompt
    result = c.orchestrator.run(message, session_id=body.session_id, auto_improve=body.auto_improve)
    result.grounded = result.metadata.get("grounded", False)  # type: ignore[assignment]
    result.session_id = body.session_id
    return result


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, c: ContainerDep) -> StreamingResponse:
    """
    SSE endpoint: streams the agent response token-by-token.
    Events:
      {"type":"start"}
      {"type":"token","text":"..."}
      {"type":"done","result":{...ChatTurnResult...}}
    """
    orchestrator = c.orchestrator
    message = body.message or body.prompt

    async def event_gen():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        loop = asyncio.get_event_loop()
        # Run the full orchestrator in a thread so we don't block the event loop
        result: ChatTurnResult = await loop.run_in_executor(
            _stream_executor,
            lambda: orchestrator.run(
                message, session_id=body.session_id, auto_improve=body.auto_improve
            ),
        )
        # Stream the answer word-by-word
        words = result.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            await asyncio.sleep(0.015)
        result.grounded = result.metadata.get("grounded", False)  # type: ignore[assignment]
        result.session_id = body.session_id
        yield f"data: {json.dumps({'type': 'done', 'result': result.model_dump()})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── async jobs ────────────────────────────────────────────────────────────────

@router.post("/agent/start")
def start_agent(body: ChatRequest, c: ContainerDep) -> dict[str, str]:
    prompt = body.message or body.prompt
    job_id = c.agent_loop.start_turn(prompt, session_id=body.session_id, auto_improve=body.auto_improve)
    return {"job_id": job_id, "status": "accepted"}


@router.get("/agent/jobs/{job_id}")
def get_job(job_id: str, c: ContainerDep) -> dict[str, object]:
    job = c.agent_loop.get_job(job_id)
    if not job:
        return {"error": "not found"}
    return {
        "job_id": job.job_id,
        "phase": job.state.phase.value,
        "result": job.turn.model_dump() if job.turn else None,
        "error": job.error,
    }


# ── symbolic math ─────────────────────────────────────────────────────────────

@router.post("/sympy/run")
def sympy_run(req: SymCodeRequest) -> dict[str, object]:
    res = run_symcode(req)
    return res.model_dump()


@router.post("/route-intent")
def route_intent(body: dict[str, str]) -> dict[str, str]:
    text = body.get("text") or ""
    return {"intent": classify_intent(text).value}


# ── sandbox ───────────────────────────────────────────────────────────────────

class SandboxRunRequest(BaseModel):
    code: str
    timeout: int = 15


@router.post("/sandbox/run")
def sandbox_run(body: SandboxRunRequest, c: ContainerDep) -> dict[str, object]:
    """Execute Python code in the subprocess sandbox."""
    from super_agent.app.infrastructure.sandbox_docker import run_sandbox
    result = run_sandbox(c.settings, body.code, body.timeout)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "backend": result.backend,
        "timed_out": result.timed_out,
    }


# ── HDC memory ────────────────────────────────────────────────────────────────

@router.get("/memory/list")
def memory_list(c: ContainerDep) -> dict[str, object]:
    records = c.memory.list_records()
    stats = c.memory.stats()
    return {"records": records, "stats": stats}


# ── codebase ──────────────────────────────────────────────────────────────────

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


# ── self-improve ──────────────────────────────────────────────────────────────

class ImproveRequestBody(BaseModel):
    instruction: str
    target_file: str | None = None


@router.post("/improve", response_model=None)
def request_improvement(body: ImproveRequestBody, c: ContainerDep) -> dict[str, object]:
    from super_agent.app.domain.chat_schemas import ImproveRequest
    req = ImproveRequest(instruction=body.instruction, target_file=body.target_file)
    result = c.orchestrator.improve_self(req.instruction, req.target_file)
    return result.model_dump()


@router.post("/improve/fullstack", response_model=None)
def request_fullstack_improvement(body: ImproveRequestBody, c: ContainerDep) -> dict[str, object]:
    """
    Full-stack improvement: applies backend change then updates agent-api.ts and AgentTester.tsx.
    Returns a FullStackImproveResult with backend + frontend_api + frontend_ui sub-results.
    """
    result = c.orchestrator.improve_full_stack(body.instruction, body.target_file)
    return result.model_dump()


@router.get("/improve/history")
def improvement_history(c: ContainerDep, limit: int = 20) -> dict[str, object]:
    history_file = c.settings.data_dir / "improvements.jsonl"
    if not history_file.is_file():
        return {"entries": []}
    lines = history_file.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"entries": list(reversed(entries))}


# ── research & heartbeat ──────────────────────────────────────────────────────

@router.post("/research/trigger")
def research_trigger(background_tasks: BackgroundTasks, c: ContainerDep) -> dict[str, str]:
    from super_agent.app.services import research_loop

    def _run() -> None:
        research_loop.run_proactive_research(c.gemini, c.settings.data_dir)

    background_tasks.add_task(_run)
    return {"status": "scheduled"}


class HeartbeatTopicsRequest(BaseModel):
    topics: list[str]


@router.get("/heartbeat/topics")
def heartbeat_topics_get(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services.research_loop import read_heartbeat_topics
    topics = read_heartbeat_topics(c.settings.data_dir)
    return {"topics": topics}


@router.post("/heartbeat/topics")
def heartbeat_topics_set(body: HeartbeatTopicsRequest, c: ContainerDep) -> dict[str, object]:
    """Replace the HEARTBEAT.md topic list."""
    heartbeat_path = c.settings.data_dir / "HEARTBEAT.md"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Heartbeat topics", "", "The agent researches these topics in rotation on each heartbeat.", ""]
    for t in body.topics:
        lines.append(f"- {t}")
    heartbeat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "topics": body.topics}


@router.get("/heartbeat/status")
def heartbeat_status_get(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services.research_loop import heartbeat_status
    status = heartbeat_status(c.settings.data_dir)
    status["interval_seconds"] = c.settings.heartbeat_interval_seconds
    return status


# ── agent memory (MEMORY.md) ──────────────────────────────────────────────────

@router.get("/memory/research")
def memory_read(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services.research_loop import read_memory
    content = read_memory(c.settings.data_dir)
    return {"content": content, "chars": len(content)}


@router.delete("/memory/research")
def memory_clear(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services.research_loop import clear_memory
    clear_memory(c.settings.data_dir)
    return {"status": "cleared"}


# ── SICA ─────────────────────────────────────────────────────────────────────

@router.get("/sica/summary")
def sica_summary(c: ContainerDep) -> dict[str, object]:
    from super_agent.app.services import sica_loop
    from super_agent.app.domain.quantum_inspired import score_summary

    summary_json = sica_loop.run_outer_loop_summary(c.settings.data_dir.parent)
    plan = json.loads(summary_json)
    q_score = score_summary(
        benchmark_score=plan.get("score", 0.0),
        gap_count=len(plan.get("gaps", [])),
        hdc_record_count=len(c.memory.list_records()),
    )
    return {"summary": summary_json, "quantum_score": q_score}


@router.get("/benchmark/history")
def benchmark_history(c: ContainerDep, limit: int = 10) -> dict[str, object]:
    from super_agent.app.services.sica_loop import load_benchmark_history
    return {"entries": load_benchmark_history(c.settings.data_dir, limit)}


# ── system info & filesystem access ───────────────────────────────────────────

@router.get("/system/os")
def get_os_info() -> dict[str, str]:
    """
    Returns information about the operating system where the agent is running.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node_name": platform.node(),
    }


@router.get("/system/files")
def list_files(
    c: ContainerDep,
    path: str = Query(..., description="The path to list, relative to the agent's data directory.")
) -> dict[str, object]:
    """
    Lists files and directories within the agent's data directory.
    Access is restricted to prevent directory traversal outside of the designated data directory.
    """
    base_dir = c.settings.data_dir
    # Resolve the requested path to its canonical form
    # and ensure it's within the allowed base_dir.
    requested_path = (base_dir / path).resolve()

    # Security check: Ensure the resolved path is a child of the resolved base_dir
    try:
        requested_path.relative_to(base_dir.resolve())
    except ValueError:
        return {"error": "Access denied: Path is outside the allowed data directory.", "path": str(requested_path)}

    if not requested_path.exists():
        return {"error": "Path not found.", "path": str(requested_path)}
    if not requested_path.is_dir():
        return {"error": "Path is not a directory. Only directories can be listed.", "path": str(requested_path)}

    entries = []
    try:
        for entry in requested_path.iterdir():
            entry_info = {
                "name": entry.name,
                "path": str(entry.relative_to(base_dir)), # Path relative to data_dir for consistency
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
                "mtime": entry.stat().st_mtime, # Modification time
                "is_symlink": entry.is_symlink(),
                "is_readable": os.access(entry, os.R_OK)
            }
            entries.append(entry_info)
    except PermissionError:
        return {"error": "Permission denied to list directory.", "path": str(requested_path)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}", "path": str(requested_path)}

    # Return the path relative to base_dir in the response
    return {"current_path": str(requested_path.relative_to(base_dir)), "entries": entries}


@router.get("/system/desktop_files")
def list_desktop_files() -> dict[str, object]:
    """
    Lists files and directories directly on the user's desktop.
    Note: This attempts to locate the standard "Desktop" directory for the current user.
    """
    # Attempt to find the user's desktop directory.
    # This path is common for Windows, macOS, and many Linux distributions.
    desktop_path = Path.home() / "Desktop"

    if not desktop_path.exists():
        return {"error": "Desktop path not found. It might be named differently or not exist.", "path": str(desktop_path)}
    if not desktop_path.is_dir():
        return {"error": "Desktop path is not a directory.", "path": str(desktop_path)}

    entries = []
    try:
        for entry in desktop_path.iterdir():
            entry_info = {
                "name": entry.name,
                "path": str(entry), # Use absolute path for desktop files
                "type": "directory" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
                "mtime": entry.stat().st_mtime, # Modification time
                "is_symlink": entry.is_symlink(),
                "is_readable": os.access(entry, os.R_OK)
            }
            entries.append(entry_info)
    except PermissionError:
        return {"error": "Permission denied to list desktop directory.", "path": str(desktop_path)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}", "path": str(desktop_path)}

    return {"current_path": str(desktop_path), "entries": entries}
