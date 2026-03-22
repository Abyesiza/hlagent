from __future__ import annotations

from dataclasses import dataclass

from super_agent.app.core.config import Settings, get_settings
from super_agent.app.core.gemini_client import GeminiClient
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
    orchestrator: SuperAgentOrchestrator
    agent_loop: AgentLoopService


def build_container() -> AppContainer:
    settings = get_settings()
    gemini = GeminiClient(settings)
    hdc = HDCMemoryStore(settings.data_dir / "hdc_memory.json")
    sessions = SessionStore(settings.data_dir / "sessions")
    orchestrator = SuperAgentOrchestrator(settings, gemini, hdc, sessions)
    agent_loop = AgentLoopService(settings, orchestrator)
    return AppContainer(
        settings=settings,
        gemini=gemini,
        hdc_memory=hdc,
        sessions=sessions,
        orchestrator=orchestrator,
        agent_loop=agent_loop,
    )
