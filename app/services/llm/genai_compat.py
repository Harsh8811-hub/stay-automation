"""A `google.generativeai`-shaped facade that dispatches by `LLM_PROVIDER`.

The rest of the codebase already speaks this shape — `GenerativeModel(model)`,
`generate_content` / `generate_content_async(contents, generation_config=...)`,
`response.text`, `response.usage_metadata`. Keeping the shape means the provider
switch is one import line per file and no prompt, parser or error handler changes.

Two providers are supported:

* `gemini` — delegates to the real `google.generativeai`. Everything behaves
  exactly as before, including the token-usage logger and debug-capture patches
  in `app.observability`, which patch the real class.
* `deepseek` — posts to DeepSeek's OpenAI-compatible `/chat/completions` with
  plain HTTP. `deepseek-flash` carries vision natively, so images travel in the
  same request as text; there is no second model to route to.

DELIBERATE DIFFERENCES FROM GEMINI (all consequences of the provider, documented
so nobody "fixes" them back):

1. Thinking mode is disabled by default. DeepSeek enables it at effort `high`,
   and while it is on the API ignores `temperature` — which several extractors
   rely on being 0.0. `DEEPSEEK_THINKING=enabled` turns it back on for a
   deliberate quality comparison.
2. DeepSeek's JSON mode may return an empty body (their docs say so). Empty is
   retried once and then raised — never accepted as a valid answer.
3. Images are sent as base64 data URLs, capped at 32 MiB each by the provider.
   Anything larger is rejected here with an explicit error rather than a 400.

Errors are raised as `LLMUnavailable` so a provider outage can be told apart from
a guest's message not being understood. Callers that catch broadly keep working;
the ones that can say something better catch this type specifically.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from app.config.settings import get_settings
from app.services.llm_errors import LLMUnavailableError

logger = logging.getLogger(__name__)


class LLMUnavailable(LLMUnavailableError, RuntimeError):
    """The model could not be reached or refused the request.

    Subclasses `app.services.llm_errors.LLMUnavailableError` so that the circuit
    breaker, the escalation suppressor and every `except LLMUnavailableError`
    written against the Gemini path also catch a DeepSeek outage — one outage
    concept, two providers.

    Distinct from "the model ran and said nothing useful" on purpose: the first is
    the bot's problem (say so, hand off), the second is the guest's (ask again).
    """


class UnsupportedInput(LLMUnavailable):
    """The selected provider cannot accept this input at all (e.g. audio)."""


class _Usage:
    """The handful of `usage_metadata` attributes this codebase reads."""

    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
        cached_content_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.cached_content_token_count = cached_content_token_count


class _Response:
    def __init__(self, text: str, usage: _Usage | None) -> None:
        self.text = text
        self.usage_metadata = usage


class GenerationConfig:
    """Accept-and-ignore is wrong here: an unrecognised knob would silently stop
    taking effect and look like a model regression. Only the knobs this codebase
    actually passes are accepted; the rest raise at construction."""

    _ALLOWED = {
        "temperature",
        "max_output_tokens",
        "response_mime_type",
        "response_schema",
        "top_p",
        "top_k",
        "candidate_count",
    }

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - self._ALLOWED
        if unknown:
            raise TypeError(f"GenerationConfig does not support: {sorted(unknown)}")
        self.temperature: float | None = kwargs.get("temperature")
        self.max_output_tokens: int | None = kwargs.get("max_output_tokens")
        self.response_mime_type: str | None = kwargs.get("response_mime_type")
        self.response_schema: Any = kwargs.get("response_schema")
        self.top_p: float | None = kwargs.get("top_p")
        self.top_k: int | None = kwargs.get("top_k")
        self.candidate_count: int | None = kwargs.get("candidate_count")
        # Kept verbatim for the Gemini path: the real SDK takes a dict and knows
        # what to do with a pydantic `response_schema`. Re-serialising it here
        # would be this module re-implementing their conversion.
        self._raw: dict[str, Any] = dict(kwargs)

    @property
    def wants_json(self) -> bool:
        return self.response_mime_type == "application/json" or self.response_schema is not None


def _as_config(cfg: "GenerationConfig | dict | None") -> GenerationConfig:
    """One call site passes a plain dict where the rest pass a GenerationConfig
    (`vision/id_card.py`). The real SDK accepts both, so the facade must too —
    otherwise that one path crashes the moment the provider is DeepSeek."""
    if isinstance(cfg, dict):
        return GenerationConfig(**cfg)
    return cfg or GenerationConfig()


def _gemini_config(cfg: GenerationConfig | None) -> dict[str, Any] | None:
    """The real SDK accepts a dict or one of *its* GenerationConfig objects. This
    module's object is neither, and handing it straight through raised
    `TypeError: Invalid input type` on every Gemini call that passed a config —
    which is most of them, silently, until the tests caught it."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(cfg._raw)


def provider_name() -> str:
    return get_settings().LLM_PROVIDER


def configure(api_key: str | None = None) -> None:
    """Kept for call-site compatibility. A no-op on DeepSeek, which takes its key
    per request from settings."""
    if provider_name() == "gemini":
        import google.generativeai as real

        real.configure(api_key=api_key)


def _split_contents(contents: Any) -> tuple[str, list[tuple[str, bytes]]]:
    """Flatten Gemini's `contents` shapes into (text, [(mime, bytes)]).

    Accepted shapes, all of which exist in this codebase today: a bare string, a
    list of strings, and a list mixing strings with `{"mime_type", "data"}` parts.
    """
    texts: list[str] = []
    images: list[tuple[str, bytes]] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            texts.append(item)
            return
        if isinstance(item, (list, tuple)):
            for entry in item:
                walk(entry)
            return
        if isinstance(item, dict):
            data = item.get("data", item.get("image"))
            if isinstance(data, (bytes, bytearray)):
                mime = item.get("mime_type") or item.get("mimeType") or "image/jpeg"
                images.append((str(mime), bytes(data)))
                return
            if "text" in item:
                texts.append(str(item["text"]))
                return
            texts.append(str(item))
            return
        texts.append(str(item))

    walk(contents)
    return "\n\n".join(t for t in texts if t), images


def _caller_label() -> str:
    """`module.function` of the code that asked for the completion, best-effort."""
    import inspect

    try:
        frame = inspect.currentframe()
        # _caller_label <- generate_content_async <- caller
        caller = frame.f_back.f_back.f_back if frame is not None else None
        if caller is None:
            return "unknown"
        module = caller.f_globals.get("__name__", "?")
        return f"{module.rsplit('.', 1)[-1]}.{caller.f_code.co_name}"
    except Exception:
        return "unknown"


def _deepseek_payload(prompt: str, images: list[tuple[str, bytes]], cfg: GenerationConfig) -> dict:
    settings = get_settings()
    if not settings.DEEPSEEK_API_KEY:
        raise LLMUnavailable("DEEPSEEK_API_KEY is not set")

    content: list[dict] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for mime, data in images:
        if not mime.startswith("image/"):
            # DeepSeek has no audio/document input. Saying so beats a 400 the
            # caller turns into "I didn't understand your message".
            raise UnsupportedInput(f"DeepSeek cannot read {mime} input")
        if len(data) > 32 * 1024 * 1024:
            raise UnsupportedInput(f"image is {len(data)} bytes, over the 32 MiB limit")
        b64 = base64.b64encode(data).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        })
    if not content:
        raise LLMUnavailable("empty prompt")

    if cfg.wants_json and "json" not in prompt.lower():
        # DeepSeek refuses `response_format: json_object` with HTTP 400 —
        # "Prompt must contain the word 'json' in some form" — and Gemini never
        # asked. Every JSON-mode call site crosses this function, and only the
        # prompt decides whether it passes, so this is the one place that can
        # fix all of them. The guard keeps it a no-op for prompts that already
        # ask for JSON in their own words (prod 2026-10-23 06:13:38 — the
        # room-detail reply).
        #
        # After the emptiness check, deliberately: an image-only call with no
        # prompt must still be rejected as empty rather than becoming a request
        # that contains nothing but this sentence.
        if content[0]["type"] == "text":
            content[0]["text"] = f"{prompt}\n\nReturn the reply as a JSON object."
        else:
            content.insert(0, {"type": "text", "text": "Return the reply as a JSON object."})

    body: dict[str, Any] = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": content if len(content) > 1 else content[0]["text"]}],
        "stream": False,
        # Sent top-level: `extra_body` is an SDK convenience, this is raw HTTP.
        "thinking": {"type": settings.DEEPSEEK_THINKING},
    }
    if settings.DEEPSEEK_THINKING == "enabled":
        body["reasoning_effort"] = settings.DEEPSEEK_REASONING_EFFORT
    elif cfg.temperature is not None:
        # In thinking mode the provider ignores this; sending it would look like a
        # no-op and hide the reason.
        body["temperature"] = cfg.temperature
    if cfg.max_output_tokens:
        body["max_tokens"] = cfg.max_output_tokens
    if cfg.wants_json:
        # A `response_schema` cannot be enforced by this provider; the prompt
        # already carries the shape, and the caller validates the parsed object.
        body["response_format"] = {"type": "json_object"}
    return body


def _post_deepseek(body: dict) -> tuple[str, _Usage | None]:
    settings = get_settings()
    url = settings.DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json=body, headers=headers)
    except httpx.TransportError as exc:
        # Seen 2 times in 101 real guest images (a dropped connection mid-
        # response). One retry, because the alternative is telling a guest who
        # just sent a payment receipt that it could not be read.
        logger.warning("DEEPSEEK_TRANSPORT_RETRY | %s", type(exc).__name__)
        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                resp = client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc2:
            raise LLMUnavailable(f"deepseek transport error: {type(exc2).__name__}") from exc2
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"deepseek transport error: {type(exc).__name__}") from exc
    return _read_deepseek(resp)


async def _post_deepseek_async(body: dict) -> tuple[str, _Usage | None]:
    settings = get_settings()
    url = settings.DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=body, headers=headers)
    except httpx.TransportError as exc:
        logger.warning("DEEPSEEK_TRANSPORT_RETRY | %s", type(exc).__name__)
        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                resp = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc2:
            raise LLMUnavailable(f"deepseek transport error: {type(exc2).__name__}") from exc2
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"deepseek transport error: {type(exc).__name__}") from exc
    return _read_deepseek(resp)


def _read_deepseek(resp: httpx.Response) -> tuple[str, _Usage | None]:
    if resp.status_code >= 400:
        # Body excerpt only; it never contains the key. Truncated because an
        # HTML error page is unreadable in a log line.
        raise LLMUnavailable(f"deepseek HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"deepseek returned an unreadable body: {type(exc).__name__}") from exc
    usage = data.get("usage") or {}
    return text, _Usage(
        prompt_token_count=usage.get("prompt_tokens"),
        candidates_token_count=usage.get("completion_tokens"),
        total_token_count=usage.get("total_tokens"),
        # DeepSeek reports cache hits separately; this is the closest analogue to
        # Gemini's cached_content_token_count, and it is the number that tells us
        # whether the shared system prompt is actually hitting the cache.
        cached_content_token_count=usage.get("prompt_cache_hit_tokens"),
    )


def _log_usage(where: str, provider: str, model: str, usage: _Usage | None) -> None:
    logger.info(
        "LLM_USAGE | provider=%s model=%s at=%s prompt_tokens=%s cached_tokens=%s output_tokens=%s",
        provider,
        model,
        where,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "cached_content_token_count", None),
        getattr(usage, "candidates_token_count", None),
    )


class GenerativeModel:
    """The subset of `google.generativeai.GenerativeModel` this codebase uses."""

    def __init__(self, model_name: str | None = None, **_ignored: Any) -> None:
        self.model_name = model_name or get_settings().GEMINI_MODEL
        self._real = None

    # ── provider dispatch ────────────────────────────────────────
    def _gemini(self):
        if self._real is None:
            import google.generativeai as real

            settings = get_settings()
            if not settings.GEMINI_API_KEY:
                raise LLMUnavailable("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
            real.configure(api_key=settings.GEMINI_API_KEY)
            self._real = real.GenerativeModel(self.model_name)
        return self._real

    def _deepseek(self, contents: Any, generation_config: GenerationConfig | None) -> _Response:
        import re
        cfg = _as_config(generation_config)
        prompt, images = _split_contents(contents)
        body = _deepseek_payload(prompt, images, cfg)
        text, usage = _post_deepseek(body)
        if cfg.wants_json:
            cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            valid = False
            if cleaned:
                try:
                    json.loads(cleaned)
                    text = cleaned
                    valid = True
                except Exception:
                    valid = False
            if not valid:
                logger.warning("DEEPSEEK_INVALID_OR_EMPTY_JSON | at=%s raw=%s — retrying once", _caller_label(), text[:150])
                body = json.loads(json.dumps(body))
                if body.get("messages") and isinstance(body["messages"][0]["content"], str):
                    body["messages"][0]["content"] += "\n\nReturn ONLY a valid JSON object without markdown formatting."
                text, usage = _post_deepseek(body)
                cleaned2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
                cleaned2 = re.sub(r"\s*```$", "", cleaned2).strip()
                try:
                    json.loads(cleaned2)
                    text = cleaned2
                except Exception as exc:
                    logger.error(f"DEEPSEEK_JSON_RETRY_FAILED | at={_caller_label()} | error={exc} | raw={text[:200]}")
        return _Response(text, usage)

    async def _deepseek_async(self, contents: Any, generation_config: GenerationConfig | None) -> _Response:
        import re
        cfg = _as_config(generation_config)
        prompt, images = _split_contents(contents)
        body = _deepseek_payload(prompt, images, cfg)
        text, usage = await _post_deepseek_async(body)
        if cfg.wants_json:
            cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            valid = False
            if cleaned:
                try:
                    json.loads(cleaned)
                    text = cleaned
                    valid = True
                except Exception:
                    valid = False
            if not valid:
                logger.warning("DEEPSEEK_INVALID_OR_EMPTY_JSON | at=%s raw=%s — retrying once", _caller_label(), text[:150])
                body = json.loads(json.dumps(body))
                if body.get("messages") and isinstance(body["messages"][0]["content"], str):
                    body["messages"][0]["content"] += "\n\nReturn ONLY a valid JSON object without markdown formatting."
                text, usage = await _post_deepseek_async(body)
                cleaned2 = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
                cleaned2 = re.sub(r"\s*```$", "", cleaned2).strip()
                try:
                    json.loads(cleaned2)
                    text = cleaned2
                except Exception as exc:
                    logger.error(f"DEEPSEEK_JSON_RETRY_FAILED | at={_caller_label()} | error={exc} | raw={text[:200]}")
        return _Response(text, usage)

    # ── public API ───────────────────────────────────────────────
    def generate_content(self, contents: Any, generation_config: GenerationConfig | None = None, **_kw: Any):
        if provider_name() == "deepseek":
            response = self._deepseek(contents, generation_config)
            _log_usage(_caller_label(), "deepseek", get_settings().DEEPSEEK_MODEL, response.usage_metadata)
            return response
        return self._gemini().generate_content(contents, generation_config=_gemini_config(generation_config))

    async def generate_content_async(self, contents: Any, generation_config: GenerationConfig | None = None, **_kw: Any):
        if provider_name() == "deepseek":
            response = await self._deepseek_async(contents, generation_config)
            _log_usage(_caller_label(), "deepseek", get_settings().DEEPSEEK_MODEL, response.usage_metadata)
            return response
        return await self._gemini().generate_content_async(contents, generation_config=_gemini_config(generation_config))


class _Types:
    """The slice of `google.generativeai.types` this codebase actually reaches for.

    The facade's whole promise is that call sites keep their Gemini-era spelling,
    and `genai.types.GenerationConfig(...)` is a legal spelling against the real
    SDK — so it has to stay legal here. Without this, `summarize_human_chat` died
    with `AttributeError: module ... has no attribute 'types'` on every takeover
    and returned the "(Summary failed)" placeholder, which the caller stores as
    if it were a real summary (prod 2026-10-23 07:14:36). Add a name here when a
    new call site needs another member, rather than changing that call site.
    """

    GenerationConfig = GenerationConfig


types = _Types()


__all__ = [
    "GenerationConfig",
    "GenerativeModel",
    "LLMUnavailable",
    "UnsupportedInput",
    "configure",
    "provider_name",
    "types",
]
