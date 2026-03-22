from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from super_agent.app.core.config import Settings
    from super_agent.app.core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


def attach_heartbeat(
    scheduler: AsyncIOScheduler,
    settings: "Settings",
    gemini: "GeminiClient",
) -> None:
    from super_agent.app.services import research_loop

    async def tick() -> None:
        try:
            msg = research_loop.run_proactive_research(gemini, settings.data_dir)
            logger.info("heartbeat: %s", msg)
        except Exception:
            logger.exception("heartbeat tick")

    scheduler.add_job(
        tick,
        IntervalTrigger(seconds=max(30, settings.heartbeat_interval_seconds)),
        id="heartbeat_research",
        replace_existing=True,
    )
