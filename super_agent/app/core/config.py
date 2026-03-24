from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPER_AGENT_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API keys (key_0 is the primary; _1 … _5 are rotation slots) ──────────
    gemini_api_key: str | None = Field(default=None)
    gemini_api_key_1: str | None = Field(default=None)
    gemini_api_key_2: str | None = Field(default=None)
    gemini_api_key_3: str | None = Field(default=None)
    gemini_api_key_4: str | None = Field(default=None)
    gemini_api_key_5: str | None = Field(default=None)

    # ── models ────────────────────────────────────────────────────────────────
    gemini_model_flash: str = Field(default="gemini-2.5-flash")
    gemini_model_pro: str = Field(default="gemini-2.5-flash")
    gemini_fallback_models: str = Field(
        default="gemini-1.5-flash",
        description="Comma-separated; tried after all keys are exhausted for the primary model",
    )
    enable_google_search: bool = Field(default=True)

    # ── paths & runtime ───────────────────────────────────────────────────────
    data_dir: Path = Field(default_factory=lambda: Path("data").resolve())
    sandbox_dir: Path = Field(default_factory=lambda: Path("sandbox").resolve())

    heartbeat_interval_seconds: int = Field(default=300)
    agent_loop_timeout_seconds: float = Field(default=3600.0)

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
    )
    cors_origin_regex: str = Field(
        default="",
        description="Optional regex for CORS origins, e.g. https://.*\\.vercel\\.app",
    )

    # ── validators ────────────────────────────────────────────────────────────
    @field_validator(
        "gemini_api_key", "gemini_api_key_1", "gemini_api_key_2",
        "gemini_api_key_3", "gemini_api_key_4", "gemini_api_key_5",
        mode="before",
    )
    @classmethod
    def _strip_key(cls, v: object) -> object:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @model_validator(mode="after")
    def _fallback_unprefixed_env(self) -> "Settings":
        """Accept GEMINI_API_KEY (no SUPER_AGENT_ prefix) as the primary key."""
        if not self.gemini_api_key:
            plain = os.environ.get("GEMINI_API_KEY", "").strip()
            if plain:
                return self.model_copy(update={"gemini_api_key": plain})
        return self

    # ── helpers ───────────────────────────────────────────────────────────────
    def all_api_keys(self) -> list[str]:
        """Return all distinct, non-empty API keys in priority order."""
        candidates = [
            self.gemini_api_key,
            self.gemini_api_key_1,
            self.gemini_api_key_2,
            self.gemini_api_key_3,
            self.gemini_api_key_4,
            self.gemini_api_key_5,
        ]
        seen: set[str] = set()
        out: list[str] = []
        for k in candidates:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
