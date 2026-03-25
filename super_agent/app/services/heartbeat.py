from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings
    from super_agent.app.core.gemini_client import GeminiClient
    from super_agent.app.infrastructure.convex_store import ConvexAgentStore

logger = logging.getLogger(__name__)


def attach_heartbeat(
    scheduler: AsyncIOScheduler,
    settings: "Settings",
    gemini: "GeminiClient",
    convex_store: "ConvexAgentStore | None" = None,
) -> None:
    from super_agent.app.services import research_loop

    async def tick() -> None:
        convex = convex_store
        task_id = convex.create_task("heartbeat_research", detail="APScheduler") if convex else None
        try:
            if convex and task_id:
                convex.set_task_status(task_id, "running")
            msg = research_loop.run_proactive_research(
                gemini,
                settings.data_dir,
                convex=convex,
            )
            logger.info("heartbeat: %s", msg)
        except Exception as e:
            if convex and task_id:
                convex.set_task_status(task_id, "failed", error=str(e))
            from super_agent.app.services.email_notify import notify_background_task

            notify_background_task(
                settings,
                kind="heartbeat_research",
                status="failed",
                detail="APScheduler",
                error=str(e),
            )
            logger.exception("heartbeat tick")
            return
        if convex and task_id:
            convex.set_task_status(task_id, "completed")

    scheduler.add_job(
        tick,
        IntervalTrigger(seconds=max(30, settings.heartbeat_interval_seconds)),
        id="heartbeat_research",
        replace_existing=True,
    )
