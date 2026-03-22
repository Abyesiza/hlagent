from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SymbolicIntent(str, Enum):
    NEURAL = "neural"
    SYMBOLIC = "symbolic"
    UNKNOWN = "unknown"


class SymCodeRequest(BaseModel):
    """LLM-generated SymPy script (deterministic execution path)."""

    source: str = Field(..., description="Python using sympy as sp; set `result`")
    verify_with_diff: bool = Field(default=True, description="Differentiate integral checks etc.")


class SymCodeResult(BaseModel):
    ok: bool
    stdout: str = ""
    result_repr: str = ""
    simplified: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
