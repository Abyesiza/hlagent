from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"


class AgentTurnState(BaseModel):
    phase: AgentPhase = AgentPhase.OBSERVE
    interaction_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
