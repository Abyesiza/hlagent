from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (hlagent/), so `.env` loads even when cwd is not the project directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPER_AGENT_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = Field(
        default=None,
        description="Google AI API key (or set env GEMINI_API_KEY without prefix)",
    )
    gemini_model_flash: str = Field(
        default="gemini-2.5-flash",
        description="Primary fast model; if quota hits 429, gemini_fallback_models are tried",
    )
    gemini_model_pro: str = Field(
        default="gemini-2.5-flash",
        description="Used for higher-quality fallback; override via SUPER_AGENT_GEMINI_MODEL_PRO",
    )
    gemini_fallback_models: str = Field(
        default="gemini-1.5-flash",
        description="Comma-separated; tried in order on 429/404 from generateContent",
    )
    enable_google_search: bool = Field(
        default=True,
        description="Use Gemini Google Search grounding for current-events queries",
    )

    data_dir: Path = Field(default_factory=lambda: Path("data").resolve())
    sandbox_dir: Path = Field(default_factory=lambda: Path("sandbox").resolve())

    heartbeat_interval_seconds: int = Field(
        default=1800,
        description="Proactive heartbeat interval (default 30 minutes; lower in dev)",
    )
    agent_loop_timeout_seconds: float = Field(default=3600.0)

    docker_sandbox_image: str = Field(default="python:3.12-slim")
    enable_docker_sandbox: bool = Field(default=False)

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated origins for browser clients (Next.js tester)",
    )

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: object) -> object:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @model_validator(mode="after")
    def gemini_key_from_unprefixed_env(self) -> "Settings":
        if self.gemini_api_key:
            return self
        plain = os.environ.get("GEMINI_API_KEY")
        if plain and plain.strip():
            return self.model_copy(update={"gemini_api_key": plain.strip()})
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
