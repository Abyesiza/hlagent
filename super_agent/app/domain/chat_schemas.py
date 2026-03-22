from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from super_agent.app.domain.math_schemas import SymCodeResult


class ChatTurnResult(BaseModel):
    """Single end-to-end turn: routing + symbolic or neural output + optional memory hit."""

    route: Literal["symbolic", "neural", "error"] = "neural"
    intent: str = ""
    answer: str = ""
    sympy: SymCodeResult | None = None
    hdc_similarity: float | None = None
    hdc_matched_task: str | None = None
    context_snippet: str = ""
    grounded: bool = False
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImproveRequest(BaseModel):
    instruction: str
    target_file: str | None = None


class ImproveResult(BaseModel):
    ok: bool
    target_file: str
    instruction: str
    old_code: str = ""
    new_code: str = ""
    ast_ok: bool = False
    committed: bool = False
    commit_hash: str | None = None
    error: str | None = None
    timestamp: str = ""
