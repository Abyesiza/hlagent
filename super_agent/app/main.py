from __future__ import annotations

import asyncio
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
    settings = container.settings

    # Seed Convex from existing disk data on first start
    if container.convex_store is not None:
        try:
            container.convex_store.ensure_seeded_from_disk(settings.data_dir)
            logger.info("Convex seed check complete")
        except Exception as exc:
            logger.warning("Convex seed failed (non-fatal): %s", exc)

    logger.info(
        "HDC model loaded: vocab=%d tokens=%d dim=%d",
        container.lm.stats.vocab_size,
        container.lm.stats.training_tokens,
        container.lm.dim,
    )

    # Optional: seed model on first startup
    if settings.seed_on_startup and container.lm.stats.training_tokens == 0:
        logger.info("Seed-on-startup enabled — bootstrapping model in background")
        asyncio.create_task(container.pipeline.run_seed_topics())

    # Start heartbeat scheduler
    scheduler = AsyncIOScheduler()
    attach_heartbeat(
        scheduler,
        settings,
        container.lm,
        convex_store=container.convex_store,
        pipeline=container.pipeline,
    )
    scheduler.start()
    logger.info("HDC Super-Agent started (continuous learning heartbeat on)")
    yield
    scheduler.shutdown(wait=False)
    # Save model to disk
    try:
        container.lm.save(settings.data_dir / "hdc_model.json")
        logger.info("HDC model saved to disk on shutdown")
    except Exception as exc:
        logger.warning("Disk model save on shutdown failed: %s", exc)
    # Save model to Convex (ensures Vercel cold starts resume from latest state)
    if container.convex_store is not None:
        try:
            payload = container.lm.to_convex_payload()
            container.convex_store.save_model_weights(**payload)
            logger.info("HDC model saved to Convex on shutdown")
        except Exception as exc:
            logger.warning("Convex model save on shutdown failed: %s", exc)
    logger.info("HDC Super-Agent shutdown")


app = FastAPI(
    title="HDC Super-Agent",
    description="Hyperdimensional Computing language model with continuous web-based learning.",
    version="2.0.0",
    lifespan=lifespan,
)

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
    return {
        "status": "ok",
        "model": "HDC-LM",
        "convex_configured": bool(s.convex_url),
        "hdc_dim": s.hdc_dim,
        "hdc_context_size": s.hdc_context_size,
        "research_interval_seconds": s.research_interval_seconds,
    }
