from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatTurnResult(BaseModel):
    """Single end-to-end chat turn result."""

    mode: str = "generation"    # "generation" | "math" | "analogy" | "similarity" | "research" | "error"
    answer: str = ""
    confidence: float = 0.0
    session_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
