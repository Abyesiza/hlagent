from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from super_agent.app.core.config import Settings
from super_agent.app.domain.agent_state import AgentPhase, AgentTurnState
from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.services.email_notify import notify_agent_job_finished, notify_improvement
from super_agent.app.services.orchestrator import SuperAgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class AgentJob:
    job_id: str
    prompt: str
    job_type: str = "chat"  # "chat" | "improve" | "sica"
    session_id: str | None = None
    auto_improve: bool = False
    project_id: str | None = None
    target_file: str | None = None
    state: AgentTurnState = field(default_factory=AgentTurnState)
    turn: ChatTurnResult | None = None
    improve_result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


class AgentLoopService:
    """
    Observe → Plan → Act → Reflect using the full SuperAgentOrchestrator stack.

    Supports three job types:
      - "chat"    — normal conversational turn (orchestrator.run)
      - "improve" — on-demand code improvement (orchestrator.improve_self)
      - "sica"    — full SICA gap-patching cycle (sica_loop.run_improvement_cycle)
    """

    def __init__(self, settings: Settings, orchestrator: SuperAgentOrchestrator) -> None:
        self._settings = settings
        self._orchestrator = orchestrator
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-loop")
        self._jobs: dict[str, AgentJob] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def start_turn(
        self,
        prompt: str,
        session_id: str | None = None,
        auto_improve: bool = False,
        project_id: str | None = None,
    ) -> str:
        job = self._make_job(
            prompt=prompt,
            job_type="chat",
            session_id=session_id,
            auto_improve=auto_improve,
            project_id=project_id,
        )
        self._executor.submit(self._run_chat, job.job_id)
        return job.job_id

    def start_improvement_job(
        self,
        instruction: str,
        target_file: str | None = None,
        fullstack: bool = False,
    ) -> str:
        """Kick off an async improvement job. Returns job_id immediately."""
        job = self._make_job(
            prompt=instruction,
            job_type="improve",
            target_file=target_file,
        )
        job.improve_result = {"pending": True, "fullstack": fullstack}
        self._executor.submit(self._run_improve, job.job_id, fullstack)
        return job.job_id

    def start_sica_cycle(self, gap_id: str | None = None) -> str:
        """Kick off an async SICA improvement cycle. Returns job_id immediately."""
        job = self._make_job(
            prompt=f"SICA cycle{f' for gap {gap_id}' if gap_id else ''}",
            job_type="sica",
        )
        self._executor.submit(self._run_sica, job.job_id, gap_id)
        return job.job_id

    def get_job(self, job_id: str) -> AgentJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        return [self._job_summary(j) for j in jobs[:limit]]

    # ── internals ────────────────────────────────────────────────────────────

    def _make_job(self, prompt: str, job_type: str = "chat", **kwargs: Any) -> AgentJob:
        job_id = str(uuid.uuid4())
        job = AgentJob(
            job_id=job_id,
            prompt=prompt,
            job_type=job_type,
            project_id=kwargs.get("project_id") or kwargs.get("session_id") or job_id,
            **{k: v for k, v in kwargs.items() if k not in ("project_id",)},
        )
        job.state.phase = AgentPhase.OBSERVE
        self._jobs[job_id] = job
        return job

    def _run_chat(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            logger.warning("Job %s not found", job_id)
            return
        try:
            # OBSERVE: note that a new turn is starting
            job.state.phase = AgentPhase.OBSERVE
            logger.debug("[%s] OBSERVE — prompt: %s…", job_id[:8], job.prompt[:60])

            # PLAN: classify intent so the ACT phase knows the route
            job.state.phase = AgentPhase.PLAN
            from super_agent.app.domain.intent import classify_intent
            intent = classify_intent(job.prompt)
            logger.debug("[%s] PLAN — classified intent: %s", job_id[:8], intent.value)

            # ACT: hand off to orchestrator
            job.state.phase = AgentPhase.ACT
            job.turn = self._orchestrator.run(
                job.prompt,
                session_id=job.session_id,
                auto_improve=job.auto_improve,
            )

            # REFLECT: evaluate quality and log
            job.state.phase = AgentPhase.REFLECT
            answer_len = len(job.turn.answer) if job.turn else 0
            route = job.turn.route if job.turn else "unknown"
            logger.info(
                "[%s] REFLECT — route=%s answer_len=%d",
                job_id[:8], route, answer_len,
            )

        except Exception as e:
            logger.exception("Error in chat job %s", job_id)
            job.error = str(e)
            job.turn = ChatTurnResult(route="error", answer=f"Orchestration failed: {e}", intent="error")
        finally:
            job.state.phase = AgentPhase.COMPLETE
            job.finished_at = datetime.now(UTC).isoformat()
            notify_agent_job_finished(
                self._settings,
                job_id=job.job_id,
                prompt=job.prompt,
                error=job.error,
                answer=job.turn.answer if job.turn else None,
            )

    def _run_improve(self, job_id: str, fullstack: bool) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            job.state.phase = AgentPhase.OBSERVE
            job.state.phase = AgentPhase.PLAN
            job.state.phase = AgentPhase.ACT

            if fullstack:
                result = self._orchestrator.improve_full_stack(job.prompt, job.target_file)
                job.improve_result = result.model_dump()
            else:
                result = self._orchestrator.improve_self(job.prompt, job.target_file)
                job.improve_result = result.model_dump()

            job.state.phase = AgentPhase.REFLECT
            logger.info(
                "[%s] IMPROVE done — ok=%s target=%s",
                job_id[:8],
                job.improve_result.get("ok"),
                job.improve_result.get("target_file"),
            )

        except Exception as e:
            logger.exception("Error in improve job %s", job_id)
            job.error = str(e)
            job.improve_result = {"ok": False, "error": str(e)}
        finally:
            job.state.phase = AgentPhase.COMPLETE
            job.finished_at = datetime.now(UTC).isoformat()
            r = job.improve_result or {}
            notify_improvement(
                self._settings,
                ok=bool(r.get("ok")),
                instruction=job.prompt,
                target_file=r.get("target_file"),
                error=r.get("error"),
            )

    def _run_sica(self, job_id: str, gap_id: str | None) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            job.state.phase = AgentPhase.OBSERVE
            from super_agent.app.services.sica_loop import run_improvement_cycle, plan_improvements, run_pytest_benchmark

            # Observe: assess current state
            repo = self._settings.data_dir.parent
            bench = run_pytest_benchmark(repo)
            plan = plan_improvements(bench)
            logger.info(
                "[%s] SICA OBSERVE — score=%.3f gaps=%d",
                job_id[:8], bench.get("score", 0), len(plan.get("todos", [])),
            )

            job.state.phase = AgentPhase.PLAN
            logger.debug("[%s] SICA PLAN — gap_id=%s", job_id[:8], gap_id)

            job.state.phase = AgentPhase.ACT
            convex = getattr(self._orchestrator, "_convex_store", None)
            cycle = run_improvement_cycle(
                self._orchestrator,
                self._settings.data_dir,
                gap_id=gap_id,
                convex=convex,
            )
            job.improve_result = cycle

            job.state.phase = AgentPhase.REFLECT
            logger.info(
                "[%s] SICA REFLECT — status=%s score_delta=%s reverted=%s",
                job_id[:8],
                cycle.get("status"),
                cycle.get("score_delta"),
                cycle.get("reverted"),
            )

        except Exception as e:
            logger.exception("Error in SICA job %s", job_id)
            job.error = str(e)
            job.improve_result = {"status": "error", "message": str(e)}
        finally:
            job.state.phase = AgentPhase.COMPLETE
            job.finished_at = datetime.now(UTC).isoformat()

    @staticmethod
    def _job_summary(job: AgentJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "prompt": job.prompt[:120],
            "phase": job.state.phase.value if job.state and job.state.phase else "unknown",
            "error": job.error,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "improve_result": job.improve_result,
            "turn_route": job.turn.route if job.turn else None,
        }
