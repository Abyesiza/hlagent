from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from super_agent.app.api.deps import build_container
from super_agent.app.core.config import get_settings
from super_agent.app.api.routes import router
from super_agent.app.services.heartbeat import attach_heartbeat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container()
    app.state.container = container
    scheduler = AsyncIOScheduler()
    attach_heartbeat(scheduler, container.settings, container.gemini)
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, object]:
    s = get_settings()
    return {
        "status": "ok",
        "gemini_configured": bool(s.gemini_api_key),
    }
