from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
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
    google_search_on_vercel: bool = Field(
        default=False,
        description=(
            "On Vercel (VERCEL=1), Google Search grounding is OFF by default because it is slow "
            "(AFC + tool calls) and often exceeds serverless timeouts → browser 'failed to fetch'. "
            "Set SUPER_AGENT_GOOGLE_SEARCH_ON_VERCEL=true to enable (use a plan with higher maxDuration)."
        ),
    )

    # ── paths & runtime ───────────────────────────────────────────────────────
    data_dir: Path = Field(default_factory=lambda: Path("data").resolve())
    sandbox_dir: Path = Field(default_factory=lambda: Path("sandbox").resolve())

    heartbeat_interval_seconds: int = Field(default=300)
    agent_loop_timeout_seconds: float = Field(default=3600.0)

    # ── SICA auto-patch ───────────────────────────────────────────────────────
    sica_auto_patch_enabled: bool = Field(
        default=False,
        description=(
            "When True, the heartbeat scheduler runs a SICA improvement cycle "
            "every sica_auto_patch_interval_hours, picking the top blueprint gap, "
            "applying a code change, running tests, and rolling back if they regress."
        ),
    )
    sica_auto_patch_interval_hours: float = Field(
        default=24.0,
        description="How often the automatic SICA patch cycle runs (in hours).",
    )
    sica_regression_threshold: float = Field(
        default=0.10,
        description="Roll back an improvement if test pass-rate drops by more than this fraction.",
    )

    convex_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPER_AGENT_CONVEX_URL", "CONVEX_URL",
        ),
        description=(
            "Convex deployment URL: research memory, tasks, heartbeat topics/cursor/persona. "
            "Also read from CONVEX_URL. Frontend uses the same URL as NEXT_PUBLIC_CONVEX_URL."
        ),
    )

    # ── email (SMTP; use app password for Gmail — never commit secrets) ───────
    email_notifications_enabled: bool = Field(default=True)
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPER_AGENT_SMTP_USER", "SMTP_USER", "Email", "EMAIL",
        ),
        description="SMTP login, usually your Gmail address.",
    )
    smtp_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPER_AGENT_SMTP_PASSWORD", "SMTP_PASSWORD", "EmailPassword", "EMAIL_PASSWORD",
        ),
        description="App password or SMTP secret.",
    )
    notification_email: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPER_AGENT_NOTIFICATION_EMAIL", "NOTIFICATION_EMAIL", "Email", "EMAIL",
        ),
        description="Where to send alerts. Defaults to smtp_user if unset.",
    )
    email_from: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SUPER_AGENT_EMAIL_FROM", "EMAIL_FROM",
        ),
        description="From address; defaults to smtp_user.",
    )
    notify_on_research: bool = Field(default=True, description="Email when new proactive research is saved.")
    notify_on_background_tasks: bool = Field(
        default=True,
        description="Email when background tasks fail (e.g. research trigger).",
    )
    notify_on_user_requested_email: bool = Field(
        default=True,
        description='When the user says "email me" / "notify me by email", email after that reply.',
    )
    notify_on_agent_job: bool = Field(default=True, description="Email when /agent/start async job finishes.")
    notify_on_improve: bool = Field(default=True, description="Email after /improve or /improve/fullstack.")

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
    )
    cors_origin_regex: str = Field(
        default=r"https://.*\.vercel\.app",
        description="Regex for allowed browser origins (e.g. any *.vercel.app frontend). Set empty to disable.",
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

    @model_validator(mode="after")
    def _smtp_host_port_unprefixed(self) -> "Settings":
        """Pick up SMTP_HOST / SMTP_PORT without SUPER_AGENT_ prefix."""
        updates: dict[str, object] = {}
        host = os.environ.get("SMTP_HOST", "").strip()
        if host:
            updates["smtp_host"] = host
        port_raw = os.environ.get("SMTP_PORT", "").strip()
        if port_raw.isdigit():
            updates["smtp_port"] = int(port_raw)
        return self.model_copy(update=updates) if updates else self

    @field_validator("smtp_password", mode="before")
    @classmethod
    def _smtp_password_strip(cls, v: object) -> object:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.replace(" ", "").strip()
            return s if s else None
        return v

    @model_validator(mode="after")
    def _vercel_writable_data_dir(self) -> "Settings":
        """
        Vercel serverless: /var/task is read-only. HDC, sessions, MEMORY.md, etc. must use /tmp.
        Override with SUPER_AGENT_DATA_DIR=/path (must be writable).
        """
        if os.environ.get("VERCEL", "").strip() != "1":
            return self
        raw = os.environ.get("SUPER_AGENT_DATA_DIR", "").strip()
        base = Path(raw) if raw else Path("/tmp/hlagent-data")
        base.mkdir(parents=True, exist_ok=True)
        sandbox = base / "sandbox"
        sandbox.mkdir(parents=True, exist_ok=True)
        return self.model_copy(
            update={"data_dir": base.resolve(), "sandbox_dir": sandbox.resolve()},
        )

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

    def use_google_search_grounding(self) -> bool:
        """Whether to call Gemini with the Google Search tool (slow; often times out on Vercel)."""
        if not self.enable_google_search:
            return False
        if os.environ.get("VERCEL", "").strip() == "1":
            return bool(self.google_search_on_vercel)
        return True


@lru_cache
def get_settings() -> Settings:
    return Settings()
