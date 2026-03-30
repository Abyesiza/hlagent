from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings
    from super_agent.app.domain.hdc_lm import HDCLanguageModel
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore
    from super_agent.app.services.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def attach_heartbeat(
    scheduler: AsyncIOScheduler,
    settings: "Settings",
    lm: "HDCLanguageModel",
    convex_store: "ConvexAgentStore | None" = None,
    pipeline: "TrainingPipeline | None" = None,
) -> None:
    """
    Register the continuous-learning heartbeat job with APScheduler.
    Every `research_interval_seconds`, one heartbeat topic is researched,
    scraped, and used to train the HDC model.
    """
    from super_agent.app.services import research_loop

    async def tick() -> None:
        import asyncio as _asyncio
        convex = convex_store
        task_id = convex.create_task("heartbeat_research", detail="APScheduler") if convex else None
        try:
            if convex and task_id:
                convex.set_task_status(task_id, "running")
            loop = _asyncio.get_event_loop()
            # Run the blocking research+train in a thread so the event loop stays free
            msg = await loop.run_in_executor(
                None,
                lambda: research_loop.run_proactive_research(
                    settings.data_dir,
                    lm,
                    convex=convex,
                    max_pages=settings.scraper_max_pages,
                    pipeline=pipeline,
                ),
            )
            logger.info("heartbeat: %s", msg)
            if convex and task_id:
                convex.set_task_status(task_id, "completed")
        except Exception as exc:
            logger.exception("heartbeat tick failed")
            if convex and task_id:
                convex.set_task_status(task_id, "failed", error=str(exc))

    interval = max(30, settings.research_interval_seconds)
    scheduler.add_job(
        tick,
        IntervalTrigger(seconds=interval),
        id="heartbeat_research",
        replace_existing=True,
    )
    logger.info("HDC continuous-learning heartbeat: interval=%ds", interval)
