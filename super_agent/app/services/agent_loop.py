from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from super_agent.app.core.config import Settings
from super_agent.app.domain.agent_state import AgentPhase, AgentTurnState
from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.services.email_notify import notify_agent_job_finished
from super_agent.app.services.orchestrator import SuperAgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class AgentJob:
    job_id: str
    prompt: str
    session_id: str | None = None
    auto_improve: bool = False
    project_id: str | None = None
    state: AgentTurnState = field(default_factory=AgentTurnState)
    turn: ChatTurnResult | None = None
    error: str | None = None


class AgentLoopService:
    """Observe → Plan → Act → Reflect using the full SuperAgentOrchestrator stack."""

    def __init__(self, settings: Settings, orchestrator: SuperAgentOrchestrator) -> None:
        self._settings = settings
        self._orchestrator = orchestrator
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-loop")
        self._jobs: dict[str, AgentJob] = {}

    def start_turn(
        self,
        prompt: str,
        session_id: str | None = None,
        auto_improve: bool = False,
        project_id: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = AgentJob(
            job_id=job_id,
            prompt=prompt,
            session_id=session_id,
            auto_improve=auto_improve,
            project_id=project_id or session_id or str(uuid.uuid4()),
        )
        job.state.phase = AgentPhase.OBSERVE
        self._jobs[job_id] = job
        self._executor.submit(self._run_sync, job_id)
        return job_id

    def _run_sync(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            logger.warning("Job %s not found", job_id)
            return
        try:
            job.state.phase = AgentPhase.PLAN
            job.state.phase = AgentPhase.ACT
            job.turn = self._orchestrator.run(
                job.prompt,
                session_id=job.session_id,
                auto_improve=job.auto_improve,
            )
            job.state.phase = AgentPhase.REFLECT
        except Exception as e:
            logger.exception("Error in agent loop for job %s", job_id)
            job.error = str(e)
            job.turn = ChatTurnResult(route="error", answer=f"Orchestration failed: {e}", intent="error")
        finally:
            job.state.phase = AgentPhase.COMPLETE
            notify_agent_job_finished(
                self._settings,
                job_id=job.job_id,
                prompt=job.prompt,
                error=job.error,
                answer=job.turn.answer if job.turn else None,
            )

    def get_job(self, job_id: str) -> AgentJob | None:
        return self._jobs.get(job_id)
