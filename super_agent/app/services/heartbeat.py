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
    orchestrator: "object | None" = None,
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

    # ── optional SICA auto-patch ──────────────────────────────────────────────
    if settings.sica_auto_patch_enabled and orchestrator is not None:

        async def sica_tick() -> None:
            from super_agent.app.services.sica_loop import run_improvement_cycle

            logger.info("SICA auto-patch: starting cycle")
            try:
                cycle = run_improvement_cycle(
                    orchestrator,
                    settings.data_dir,
                    convex=convex_store,
                )
                logger.info(
                    "SICA auto-patch: status=%s score_delta=%s reverted=%s",
                    cycle.get("status"), cycle.get("score_delta"), cycle.get("reverted"),
                )
            except Exception:
                logger.exception("SICA auto-patch tick failed")

        interval_seconds = max(3600, int(settings.sica_auto_patch_interval_hours * 3600))
        scheduler.add_job(
            sica_tick,
            IntervalTrigger(seconds=interval_seconds),
            id="sica_auto_patch",
            replace_existing=True,
        )
        logger.info(
            "SICA auto-patch enabled — interval %sh",
            settings.sica_auto_patch_interval_hours,
        )
