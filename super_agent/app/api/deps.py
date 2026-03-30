from __future__ import annotations

from dataclasses import dataclass

from super_agent.app.core.config import Settings, get_settings
from super_agent.app.domain.hdc_lm import HDCLanguageModel
from super_agent.app.infrastructure.convex_store import ConvexAgentStore
from super_agent.app.infrastructure.hdc_memory_store import HDCMemoryStore
from super_agent.app.services.orchestrator import SuperAgentOrchestrator
from super_agent.app.services.session_store import SessionStore
from super_agent.app.services.training_pipeline import TrainingPipeline


@dataclass
class AppContainer:
    settings: Settings
    lm: HDCLanguageModel
    hdc_memory: HDCMemoryStore
    sessions: SessionStore
    convex_store: ConvexAgentStore | None
    orchestrator: SuperAgentOrchestrator
    pipeline: TrainingPipeline

    @property
    def memory(self) -> HDCMemoryStore:
        return self.hdc_memory


def build_container() -> AppContainer:
    settings = get_settings()

    # Load or create the HDC Language Model
    model_path = settings.data_dir / "hdc_model.json"
    lm = HDCLanguageModel.load_or_new(
        model_path,
        dim=settings.hdc_dim,
        context_size=settings.hdc_context_size,
    )

    hdc = HDCMemoryStore(settings.data_dir / "hdc_memory.json")
    sessions = SessionStore(settings.data_dir / "sessions")

    convex_store = (
        ConvexAgentStore(settings.convex_url)
        if settings.convex_url
        else None
    )

    orchestrator = SuperAgentOrchestrator(
        settings, lm, hdc, sessions, convex_store=convex_store
    )

    pipeline = TrainingPipeline(
        lm,
        settings.data_dir,
        convex_store=convex_store,
    )

    return AppContainer(
        settings=settings,
        lm=lm,
        hdc_memory=hdc,
        sessions=sessions,
        convex_store=convex_store,
        orchestrator=orchestrator,
        pipeline=pipeline,
    )
