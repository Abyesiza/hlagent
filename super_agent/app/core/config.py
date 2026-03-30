from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPER_AGENT_",
        env_file=(
            _REPO_ROOT / ".env",
            _REPO_ROOT / ".env.local",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── HDC model ─────────────────────────────────────────────────────────────
    hdc_dim: int = Field(default=10_000, description="Hypervector dimensionality.")
    hdc_context_size: int = Field(default=5, description="N-gram context window.")

    # ── research / scraper ────────────────────────────────────────────────────
    scraper_max_pages: int = Field(
        default=5,
        description="Max web pages to scrape per research topic.",
    )
    scraper_rate_limit_seconds: float = Field(
        default=1.5,
        description="Polite delay between HTTP requests.",
    )
    include_wikipedia: bool = Field(
        default=True,
        description="Fetch Wikipedia articles as primary training data.",
    )
    research_interval_seconds: int = Field(
        default=60,
        description="How often the continuous learning scheduler runs (seconds).",
    )
    seed_on_startup: bool = Field(
        default=True,
        description="Run seed-topic training on first startup (slow, set True for a fresh model).",
    )

    # ── paths & runtime ───────────────────────────────────────────────────────
    data_dir: Path = Field(default_factory=lambda: Path("data").resolve())

    # ── Convex ────────────────────────────────────────────────────────────────
    convex_url: str | None = Field(
        default=None,
        description="Convex deployment URL for memory, tasks, and training logs.",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
    )
    cors_origin_regex: str = Field(
        default=r"https://.*\.vercel\.app",
        description="Regex for allowed browser origins.",
    )

    # ── validators ────────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _vercel_writable_data_dir(self) -> "Settings":
        if os.environ.get("VERCEL", "").strip() != "1":
            return self
        raw = os.environ.get("SUPER_AGENT_DATA_DIR", "").strip()
        base = Path(raw) if raw else Path("/tmp/hlagent-data")
        base.mkdir(parents=True, exist_ok=True)
        # Use object.__setattr__ so Pydantic v2 doesn't reject the in-place
        # mutation (returning model_copy() from a mode="after" validator is
        # silently ignored when called via __init__).
        object.__setattr__(self, "data_dir", base.resolve())
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
