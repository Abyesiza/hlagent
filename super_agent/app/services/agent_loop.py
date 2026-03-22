from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from super_agent.app.core.config import Settings
from super_agent.app.domain.agent_state import AgentPhase, AgentTurnState
from super_agent.app.domain.chat_schemas import ChatTurnResult
from super_agent.app.services.orchestrator import SuperAgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class AgentJob:
    job_id: str
    prompt: str
    session_id: str | None = None
    state: AgentTurnState = field(default_factory=AgentTurnState)
    turn: ChatTurnResult | None = None
    error: str | None = None
    project_id: str | None = None  # Added for project tracking across multiple turns/jobs


class AgentLoopService:
    """Observe → Plan → Act → Reflect using the full SuperAgentOrchestrator stack."""

    def __init__(self, settings: Settings, orchestrator: SuperAgentOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-loop")
        self._jobs: dict[str, AgentJob] = {}

    def start_turn(self, prompt: str, session_id: str | None = None, project_id: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        
        # Determine the effective project ID for this turn.
        # This allows linking multiple turns to a broader project/session context.
        effective_project_id: str
        if project_id is not None:
            effective_project_id = project_id
        elif session_id is not None:
            # If a session_id exists and no specific project_id, treat the session as the project
            effective_project_id = session_id
        else:
            # If neither session_id nor project_id, this is a new, standalone project
            effective_project_id = str(uuid.uuid4())

        job = AgentJob(
            job_id=job_id,
            prompt=prompt,
            session_id=session_id,
            project_id=effective_project_id  # Pass the determined project ID
        )
        job.state.phase = AgentPhase.OBSERVE
        self._jobs[job_id] = job
        self._executor.submit(self._run_sync, job_id)
        return job_id

    def _run_sync(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found, cannot run sync.")
            return

        try:
            job.state.phase = AgentPhase.PLAN
            # The orchestrator is now called with project_id, allowing it to maintain context
            # and potentially inform future proactive steps.
            job.state.phase = AgentPhase.ACT
            job.turn = self._orchestrator.run(job.prompt, session_id=job.session_id, project_id=job.project_id)
            job.state.phase = AgentPhase.REFLECT

            # --- Proactive Logic Placeholder ---
            # At this point, the orchestrator (or a higher-level project manager service)
            # would analyze job.turn and the project_id to determine if more steps are needed
            # for the overall project.
            # If the orchestrator's output (e.g., in job.turn) indicates an ongoing project
            # with a clear next logical action, the AgentLoopService could:
            # 1. Update the project state (e.g., in a session store or persistent project store).
            # 2. Programmatically generate a 'next prompt'.
            # 3. Call self.start_turn(next_prompt, session_id=job.session_id, project_id=job.project_id)
            #    to enqueue the next proactive step, running it in the background.
            # This would allow the agent to continue working on a project without explicit
            # user prompts for every single sub-step, fostering proactive assistance.
            # For now, this specific auto-queuing logic is a conceptual placeholder,
            # pending further enhancements to the orchestrator's return type and project flow management.

        except Exception as e:
            logger.exception(f"Error in agent loop for job {job_id}")
            job.error = str(e)
            job.turn = ChatTurnResult(route="error", answer=f"Orchestration failed: {e}", intent="error")
        finally:
            # Mark the current turn as complete. If proactive steps are taken,
            # they will typically involve creating new AgentJobs linked by project_id.
            job.state.phase = AgentPhase.COMPLETE

    def get_job(self, job_id: str) -> AgentJob | None:
        return self._jobs.get(job_id)
