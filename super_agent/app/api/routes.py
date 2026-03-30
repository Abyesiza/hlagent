"""
FastAPI routes for the HDC Super-Agent.

Endpoints
=========
POST /chat                 — synchronous chat turn (HDC reasoning)
POST /chat/stream          — streaming chat (SSE)
POST /train/text           — train model on a raw text snippet (one-shot)
POST /train/topic          — research a topic + train (async background)
GET  /train/status         — training pipeline status
POST /research             — research a topic and return a summary
POST /analogy              — solve A:B::C:? via HDC algebra
POST /similar              — find semantically similar words
GET  /model/stats          — HDC model statistics
GET  /model/vocab          — sample from the vocabulary
POST /model/generate       — generate text from a seed
GET  /memory               — list recent HDC memory records
DELETE /memory             — clear memory
GET  /heartbeat/status     — heartbeat/research topic cursor
POST /heartbeat/topics     — update research topics
GET  /health               — system health
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

# Dedicated 2-thread executor for HDC inference so heavy training tasks
# don't block the chat/generate endpoints via GIL contention.
_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hdc_infer")
from pydantic import BaseModel, Field

from super_agent.app.api.deps import AppContainer
from super_agent.app.services.research_loop import (
    heartbeat_status,
    read_heartbeat_topics,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── dependency ────────────────────────────────────────────────────────────────


def get_container(request: Request) -> AppContainer:
    return request.app.state.container  # type: ignore[no-any-return]


# ── request / response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    confidence: float
    session_id: str | None = None
    details: dict = Field(default_factory=dict)


class TrainTextRequest(BaseModel):
    text: str


class TrainTopicRequest(BaseModel):
    topic: str
    max_pages: int = 5
    question: str | None = None


class GenerateRequest(BaseModel):
    seed: str
    max_tokens: int = 50
    temperature: float = 0.8


class AnalogyRequest(BaseModel):
    a: str
    b: str
    c: str


class SimilarRequest(BaseModel):
    word: str
    top_k: int = 8


class HeartbeatTopicsRequest(BaseModel):
    topics: list[str]


class ResearchRequest(BaseModel):
    topic: str
    max_pages: int = 5


# ── chat ──────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, c: AppContainer = Depends(get_container)) -> ChatResponse:
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _INFERENCE_POOL,
                lambda: c.orchestrator.chat(body.message, session_id=body.session_id),
            ),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        return ChatResponse(
            answer="Still learning — the model is currently training. Try again in a moment.",
            mode="busy",
            confidence=0.0,
            session_id=body.session_id,
            details={"reason": "training_in_progress"},
        )
    return ChatResponse(
        answer=result.answer,
        mode=result.mode,
        confidence=result.confidence,
        session_id=result.session_id,
        details=result.details,
    )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest, c: AppContainer = Depends(get_container)
) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        async for token in c.orchestrator.chat_stream(
            body.message, session_id=body.session_id
        ):
            yield f"data: {token}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── training ──────────────────────────────────────────────────────────────────


@router.post("/train/text")
def train_text(body: TrainTextRequest, c: AppContainer = Depends(get_container)) -> dict:
    pairs = c.lm.train_text(body.text)
    c.lm.save(c.settings.data_dir / "hdc_model.json")
    return {
        "pairs_trained": pairs,
        "vocab_size": c.lm.stats.vocab_size,
        "total_tokens": c.lm.stats.training_tokens,
    }


@router.post("/train/topic")
async def train_topic(
    body: TrainTopicRequest,
    background_tasks: BackgroundTasks,
    c: AppContainer = Depends(get_container),
) -> dict:
    """
    Research a topic on the internet and train the HDC model.
    Long-running — runs in the background; poll /train/status for progress.
    """
    async def _run() -> None:
        await c.pipeline.run_topic(body.topic, max_pages=body.max_pages)

    background_tasks.add_task(_run)
    return {
        "status": "started",
        "topic": body.topic,
        "message": "Research + training running in background. Poll /train/status.",
    }


@router.get("/train/status")
def train_status(c: AppContainer = Depends(get_container)) -> dict:
    return c.pipeline.status()


# ── research ──────────────────────────────────────────────────────────────────


@router.post("/research")
async def research(
    body: ResearchRequest, c: AppContainer = Depends(get_container)
) -> dict:
    result = await c.orchestrator.research_and_answer(
        topic=body.topic,
    )
    return result.to_dict()


# ── generation ────────────────────────────────────────────────────────────────


@router.post("/model/generate")
async def generate(body: GenerateRequest, c: AppContainer = Depends(get_container)) -> dict:
    loop = asyncio.get_event_loop()
    try:
        text = await asyncio.wait_for(
            loop.run_in_executor(
                _INFERENCE_POOL,
                lambda: c.lm.generate(
                    body.seed,
                    max_tokens=min(body.max_tokens, 200),
                    temperature=body.temperature,
                ),
            ),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        text = ""
    return {
        "seed": body.seed,
        "generated": text,
        "full_text": body.seed + (" " if text else "") + text,
    }


# ── analogy + similarity ──────────────────────────────────────────────────────


@router.post("/analogy")
def analogy(body: AnalogyRequest, c: AppContainer = Depends(get_container)) -> dict:
    candidates = c.lm.analogy(body.a, body.b, body.c, top_k=5)
    return {
        "query": f"{body.a}:{body.b}::{body.c}:?",
        "candidates": [{"word": w, "similarity": s} for w, s in candidates],
        "best": candidates[0][0] if candidates else None,
    }


@router.post("/similar")
def similar(body: SimilarRequest, c: AppContainer = Depends(get_container)) -> dict:
    results = c.lm.most_similar(body.word, top_k=body.top_k)
    return {
        "word": body.word,
        "similar": [{"word": w, "similarity": s} for w, s in results],
    }


# ── model introspection ───────────────────────────────────────────────────────


@router.get("/model/stats")
def model_stats(c: AppContainer = Depends(get_container)) -> dict:
    return c.orchestrator.status()


@router.get("/model/vocab")
def model_vocab(
    limit: int = 100,
    c: AppContainer = Depends(get_container),
) -> dict:
    vocab = c.lm.space.item_memory.labels()
    return {
        "vocab_size": len(vocab),
        "sample": vocab[:limit],
    }


# ── memory ────────────────────────────────────────────────────────────────────


@router.get("/memory")
def list_memory(
    limit: int = 50,
    c: AppContainer = Depends(get_container),
) -> dict:
    return {
        "records": c.hdc_memory.list_records(limit=limit),
        "stats": c.hdc_memory.stats(),
    }


@router.delete("/memory")
def clear_memory(c: AppContainer = Depends(get_container)) -> dict:
    from super_agent.app.services.research_loop import clear_memory as _clear

    _clear(c.settings.data_dir, convex=c.convex_store)
    c.hdc_memory._records.clear()
    c.hdc_memory._memory_hv = None
    c.hdc_memory._save()
    return {"status": "cleared"}


# ── heartbeat / research config ───────────────────────────────────────────────


@router.get("/heartbeat/status")
def hb_status(c: AppContainer = Depends(get_container)) -> dict:
    return heartbeat_status(c.settings.data_dir, convex=c.convex_store)  # type: ignore[return-value]


@router.post("/heartbeat/topics")
def set_topics(
    body: HeartbeatTopicsRequest,
    c: AppContainer = Depends(get_container),
) -> dict:
    if c.convex_store is not None:
        c.convex_store.set_topics(body.topics)
    else:
        from pathlib import Path

        hb_path = c.settings.data_dir / "HEARTBEAT.md"
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# HDC Research Topics\n"] + [f"- {t}" for t in body.topics]
        hb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "updated", "topic_count": len(body.topics)}


@router.post("/heartbeat/run")
async def run_heartbeat_now(
    background_tasks: BackgroundTasks,
    c: AppContainer = Depends(get_container),
) -> dict:
    """Trigger one heartbeat research cycle immediately."""
    from super_agent.app.services.research_loop import run_proactive_research

    def _run() -> None:
        run_proactive_research(
            c.settings.data_dir,
            c.lm,
            convex=c.convex_store,
            max_pages=c.settings.scraper_max_pages,
        )

    background_tasks.add_task(_run)
    return {"status": "triggered", "message": "Heartbeat research started in background."}
