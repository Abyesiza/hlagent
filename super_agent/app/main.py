from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from super_agent.app.api.deps import build_container
from super_agent.app.core.config import get_settings
from super_agent.app.api.routes import router
from super_agent.app.services.email_notify import email_ready
from super_agent.app.services.heartbeat import attach_heartbeat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container()
    app.state.container = container

    # Seed Convex from existing disk data on first start
    if container.convex_store is not None:
        try:
            container.convex_store.ensure_seeded_from_disk(container.settings.data_dir)
            logger.info("Convex seed check complete")
        except Exception as exc:
            logger.warning("Convex seed failed (non-fatal): %s", exc)

    # Log email configuration status at startup
    settings = container.settings
    if settings.smtp_user and settings.smtp_password:
        logger.info(
            "Email notifications: READY  user=%s  notify_to=%s",
            settings.smtp_user,
            settings.notification_email or settings.smtp_user,
        )
    else:
        logger.warning(
            "Email notifications: NOT configured — set Email + EmailPassword in .env"
        )

    scheduler = AsyncIOScheduler()
    attach_heartbeat(
        scheduler,
        container.settings,
        container.gemini,
        convex_store=container.convex_store,
        orchestrator=container.orchestrator,
    )
    scheduler.start()
    logger.info("Super Agent API started (heartbeat scheduler on)")
    yield
    scheduler.shutdown(wait=False)
    logger.info("Super Agent API shutdown")


app = FastAPI(title="Super Agent", lifespan=lifespan)

_settings = get_settings()
_cors_list = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_origin_regex=_settings.cors_origin_regex.strip() or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, object]:
    s = get_settings()
    keys = s.all_api_keys()
    return {
        "status": "ok",
        "gemini_configured": len(keys) > 0,
        "gemini_key_count": len(keys),
        "convex_configured": bool(s.convex_url),
        "email_notifications_ready": email_ready(s),
    }
