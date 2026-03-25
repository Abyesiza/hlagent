from __future__ import annotations

from dataclasses import dataclass

from super_agent.app.core.config import Settings, get_settings
from super_agent.app.core.gemini_client import GeminiClient
from super_agent.app.infrastructure.convex_store import ConvexAgentStore
from super_agent.app.infrastructure.hdc_memory_store import HDCMemoryStore
from super_agent.app.services.agent_loop import AgentLoopService
from super_agent.app.services.orchestrator import SuperAgentOrchestrator
from super_agent.app.services.session_store import SessionStore


@dataclass
class AppContainer:
    settings: Settings
    gemini: GeminiClient
    hdc_memory: HDCMemoryStore
    sessions: SessionStore
    convex_store: ConvexAgentStore | None
    orchestrator: SuperAgentOrchestrator
    agent_loop: AgentLoopService

    @property
    def memory(self) -> HDCMemoryStore:
        """Alias for routes that reference c.memory."""
        return self.hdc_memory


def build_container() -> AppContainer:
    settings = get_settings()
    gemini = GeminiClient(settings)
    hdc = HDCMemoryStore(settings.data_dir / "hdc_memory.json")
    sessions = SessionStore(settings.data_dir / "sessions")
    convex_store = (
        ConvexAgentStore(settings.convex_url)
        if settings.convex_url
        else None
    )
    orchestrator = SuperAgentOrchestrator(
        settings, gemini, hdc, sessions, convex_store=convex_store
    )
    agent_loop = AgentLoopService(settings, orchestrator)
    return AppContainer(
        settings=settings,
        gemini=gemini,
        hdc_memory=hdc,
        sessions=sessions,
        convex_store=convex_store,
        orchestrator=orchestrator,
        agent_loop=agent_loop,
    )
