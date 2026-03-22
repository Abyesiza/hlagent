"""
Gemini client with multi-key rotation.

On every 429 the client moves to the next API key (same model).
Only after exhausting all keys for a model does it try the next fallback model.
A 404 skips the current model entirely (model not available for any key).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from super_agent.app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiInteractionHandle:
    interaction_id: str
    background: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _friendly_failure(tried: list[str], last: BaseException) -> str:
    code = getattr(last, "code", None)
    if code == 429 or "429" in str(last) or "RESOURCE_EXHAUSTED" in str(last):
        n = len(tried)
        sample = ", ".join(tried[:4]) + ("…" if n > 4 else "")
        return (
            f"[Quota exhausted on all {n} key×model combinations tried ({sample}). "
            "Add more keys via SUPER_AGENT_GEMINI_API_KEY_1 … _5 in .env, "
            "or wait a few minutes. https://ai.google.dev/gemini-api/docs/rate-limits]"
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


# ── client ────────────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Wraps google.genai with automatic key rotation on 429.

    Call order for each request:
      primary_model + key[0]
      primary_model + key[1]
      … (all keys for primary model)
      fallback_model[0] + key[0]
      fallback_model[0] + key[1]
      … (all keys for first fallback model)
      … and so on
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._clients: list[Any] = []

        keys = settings.all_api_keys()
        if not keys:
            logger.warning(
                "No Gemini API key found. Set SUPER_AGENT_GEMINI_API_KEY (or GEMINI_API_KEY) in .env"
            )
            return

        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("google-genai not installed — Gemini disabled")
            return

        for k in keys:
            try:
                self._clients.append(genai.Client(api_key=k))
            except Exception as e:
                logger.warning("Could not init Gemini client for key …%s: %s", k[-4:], e)

        logger.info(
            "Gemini ready: %d key(s), primary model=%s, fallbacks=%s, search=%s",
            len(self._clients),
            settings.gemini_model_flash,
            settings.gemini_fallback_models,
            settings.enable_google_search,
        )

    @property
    def enabled(self) -> bool:
        return len(self._clients) > 0

    # ── content builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_contents(prompt: str, history: list[dict[str, str]] | None) -> Any:
        from google.genai.types import Content, Part  # type: ignore[import-untyped]
        if not history:
            return prompt
        turns: list[Content] = []
        for h in history:
            turns.append(Content(role=h["role"], parts=[Part(text=h["text"])]))
        turns.append(Content(role="user", parts=[Part(text=prompt)]))
        return turns

    # ── core call with key × model rotation ──────────────────────────────────

    def _call(
        self,
        prompt: str,
        model: str | None,
        history: list[dict[str, str]] | None,
        config: Any,
    ) -> str:
        if not self._clients:
            return (
                "[gemini disabled: set SUPER_AGENT_GEMINI_API_KEY in "
                f"{Path(__file__).resolve().parents[3] / '.env'}]"
            )

        from google.genai import errors as genai_errors  # type: ignore[import-untyped]

        contents = self._build_contents(prompt, history)
        models = _model_chain(self._settings, model)
        tried: list[str] = []
        last_exc: BaseException | None = None

        for m in models:
            for ki, client in enumerate(self._clients):
                label = f"{m}[key{ki}]"
                tried.append(label)
                try:
                    if config is not None:
                        response = client.models.generate_content(
                            model=m, contents=contents, config=config
                        )
                    else:
                        response = client.models.generate_content(
                            model=m, contents=contents
                        )
                    text = _extract_text(response)
                    if text:
                        if len(tried) > 1:
                            logger.info("Gemini succeeded on %s", label)
                        return text
                    return str(response)

                except genai_errors.ClientError as e:
                    last_exc = e
                    code = getattr(e, "code", None)
                    if code == 429:
                        logger.warning("Gemini %s → 429, rotating to next key", label)
                        continue          # try next key, same model
                    if code == 404:
                        logger.warning("Gemini %s → 404 (model not found), skipping model", label)
                        break             # skip remaining keys for this model
                    logger.exception("Gemini non-retryable error on %s", label)
                    return f"[gemini error: {e}]"
                except Exception as e:
                    last_exc = e
                    logger.exception("Gemini unexpected error on %s", label)
                    return f"[gemini error: {e}]"

        return _friendly_failure(tried, last_exc) if last_exc else "[gemini error: all options exhausted]"

    # ── public API ────────────────────────────────────────────────────────────

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
        if not self._settings.enable_google_search:
            return self.generate_text(prompt, model, history, system_instruction), False
        try:
            from google.genai.types import (  # type: ignore[import-untyped]
                GenerateContentConfig, GoogleSearch, Tool,
            )
            config = GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
                system_instruction=system_instruction or None,
            )
            return self._call(prompt, model, history, config), True
        except Exception as e:
            logger.warning("generate_with_search failed, falling back: %s", e)
            return self.generate_text(prompt, model, history, system_instruction), False

    def generate_text_stream(
        self,
        prompt: str,
        model: str | None = None,
        history: list[dict[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks from Gemini streaming, rotating keys on 429."""
        if not self._clients:
            yield "[gemini disabled]"
            return

        from google.genai import errors as genai_errors  # type: ignore[import-untyped]

        config = None
        if system_instruction:
            from google.genai.types import GenerateContentConfig  # type: ignore[import-untyped]
            config = GenerateContentConfig(system_instruction=system_instruction)

        contents = self._build_contents(prompt, history)
        models = _model_chain(self._settings, model)

        for m in models:
            for ki, client in enumerate(self._clients):
                try:
                    streamed_any = False
                    for chunk in client.models.generate_content_stream(
                        model=m, contents=contents, config=config
                    ):
                        text = _extract_text(chunk)
                        if text:
                            streamed_any = True
                            yield text
                    if streamed_any:
                        return
                except genai_errors.ClientError as e:
                    code = getattr(e, "code", None)
                    if code == 429:
                        logger.warning("Gemini stream %s[key%d] → 429, rotating key", m, ki)
                        continue
                    if code == 404:
                        logger.warning("Gemini stream %s → 404, skipping model", m)
                        break
                    yield f"[gemini streaming error: {e}]"
                    return
                except Exception as e:
                    logger.warning("Streaming failed %s[key%d]: %s — trying next", m, ki, e)
                    continue

        # All streaming failed — fall back to full blocking call
        yield self._call(prompt, model, history, config)

    def start_background_interaction(self, prompt: str, model: str | None = None) -> GeminiInteractionHandle:
        if not self._clients:
            return GeminiInteractionHandle(interaction_id="disabled", background=True)
        _ = self.generate_text(prompt, model=model)
        return GeminiInteractionHandle(
            interaction_id="sync-fallback-1",
            background=True,
            metadata={"note": "Replace with real Interactions API poll loop"},
        )

    def poll_interaction(self, handle: GeminiInteractionHandle) -> dict[str, Any]:
        return {"id": handle.interaction_id, "status": "completed", "background": handle.background}
