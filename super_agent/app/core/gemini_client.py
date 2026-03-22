from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from super_agent.app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiInteractionHandle:
    """Opaque handle for a long-running Gemini Interactions / background job."""

    interaction_id: str
    background: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def _model_chain(settings: Settings, preferred: str | None) -> list[str]:
    primary = (preferred or settings.gemini_model_flash).strip()
    extra = [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for m in [primary, *extra]:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _friendly_gemini_failure(models_tried: list[str], last: BaseException) -> str:
    code = getattr(last, "code", None)
    if code == 429 or "429" in str(last) or "RESOURCE_EXHAUSTED" in str(last):
        return (
            "[Gemini quota exhausted for all tried models: "
            + ", ".join(models_tried)
            + ". Wait a few minutes, or add more models via SUPER_AGENT_GEMINI_FALLBACK_MODELS. "
            "See https://ai.google.dev/gemini-api/docs/rate-limits ]"
        )
    return f"[gemini error: {last}]"


def _extract_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        c0 = candidates[0]
        fin = getattr(c0, "finish_reason", None)
        if fin and str(fin).upper() not in ("STOP", "FINISH_REASON_STOP", "1"):
            logger.warning("Gemini finish_reason=%s", fin)
        parts = getattr(c0.content, "parts", [])
        joined = "".join(getattr(p, "text", "") for p in parts)
        if joined:
            return joined
    return None


class GeminiClient:
    """
    Thin wrapper around google.genai.

    - `generate_text`: plain generation, model fallback chain on 429/404.
    - `generate_with_search`: same but enables Google Search grounding so
      the model can retrieve live web results.
    - Multi-turn: pass `history` (list of {role, text} dicts) to either method.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any = None
        key = settings.gemini_api_key
        if key:
            try:
                from google import genai  # type: ignore[import-untyped]

                self._client = genai.Client(api_key=key)
                logger.info(
                    "Gemini client ready (primary=%s, search=%s)",
                    settings.gemini_model_flash,
                    settings.enable_google_search,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("Gemini client init failed: %s", e)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Internal: build the contents list for multi-turn
    # ------------------------------------------------------------------
    @staticmethod
    def _build_contents(
        prompt: str,
        history: list[dict[str, str]] | None,
    ) -> Any:
        """Return a `list[Content]` for multi-turn, or a plain string for single-turn."""
        from google.genai.types import Content, Part  # type: ignore[import-untyped]

        if not history:
            return prompt
        turns: list[Content] = []
        for h in history:
            turns.append(Content(role=h["role"], parts=[Part(text=h["text"])]))
        turns.append(Content(role="user", parts=[Part(text=prompt)]))
        return turns

    # ------------------------------------------------------------------
    # Internal: _call_model with fallback chain
    # ------------------------------------------------------------------
    def _call(
        self,
        prompt: str,
        model: str | None,
        history: list[dict[str, str]] | None,
        config: Any,
    ) -> str:
        if not self._client:
            return (
                "[gemini disabled: set SUPER_AGENT_GEMINI_API_KEY or GEMINI_API_KEY in "
                f"{Path(__file__).resolve().parents[3] / '.env'}]"
            )

        from google.genai import errors as genai_errors  # type: ignore[import-untyped]

        contents = self._build_contents(prompt, history)
        chain = _model_chain(self._settings, model)
        models_tried: list[str] = []
        last_exc: BaseException | None = None

        for idx, m in enumerate(chain):
            models_tried.append(m)
            try:
                if config is not None:
                    response = self._client.models.generate_content(
                        model=m, contents=contents, config=config
                    )
                else:
                    response = self._client.models.generate_content(model=m, contents=contents)

                text = _extract_text(response)
                if text:
                    if idx > 0:
                        logger.info("Gemini succeeded on fallback model %s", m)
                    return text
                return str(response)

            except genai_errors.ClientError as e:
                last_exc = e
                code = getattr(e, "code", None)
                if code in (429, 404):
                    logger.warning("Gemini model=%s code=%s — trying next", m, code)
                    continue
                logger.exception("Gemini call failed (non-retryable)")
                return f"[gemini error: {e}]"
            except Exception as e:
                last_exc = e
                logger.exception("Gemini call failed")
                return f"[gemini error: {e}]"

        if last_exc is not None:
            return _friendly_gemini_failure(models_tried, last_exc)
        return "[gemini error: no models in chain]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        history: list[dict[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> str:
        config = None
        if system_instruction:
            from google.genai.types import GenerateContentConfig  # type: ignore[import-untyped]

            config = GenerateContentConfig(system_instruction=system_instruction)
        return self._call(prompt, model, history, config)

    def generate_with_search(
        self,
        prompt: str,
        model: str | None = None,
        history: list[dict[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> tuple[str, bool]:
        """
        Generate with Google Search grounding enabled.

        Returns (text, grounded) where grounded=True means the model used web results.
        Falls back to plain generation when the setting is off or the tool is unavailable.
        """
        if not self._settings.enable_google_search:
            return self.generate_text(prompt, model, history, system_instruction), False

        try:
            from google.genai.types import (  # type: ignore[import-untyped]
                GenerateContentConfig,
                GoogleSearch,
                Tool,
            )

            tool = Tool(google_search=GoogleSearch())
            config = GenerateContentConfig(
                tools=[tool],
                system_instruction=system_instruction or None,
            )
            text = self._call(prompt, model, history, config)
            return text, True
        except Exception as e:
            logger.warning("generate_with_search failed, falling back to plain: %s", e)
            return self.generate_text(prompt, model, history, system_instruction), False

    def start_background_interaction(self, prompt: str, model: str | None = None) -> GeminiInteractionHandle:
        if not self._client:
            return GeminiInteractionHandle(interaction_id="disabled", background=True)
        _ = self.generate_text(prompt, model=model)
        return GeminiInteractionHandle(
            interaction_id="sync-fallback-1",
            background=True,
            metadata={"note": "Replace with real Interactions API poll loop"},
        )

    def poll_interaction(self, handle: GeminiInteractionHandle) -> dict[str, Any]:
        return {"id": handle.interaction_id, "status": "completed", "background": handle.background}
