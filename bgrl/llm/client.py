"""OpenRouter chat client behind a network-free :class:`ChatClient` seam.

Every consumer (the agent, the refinement harness, the tests) depends only on the
:class:`ChatClient` protocol — ``complete(messages, params) -> ChatResponse`` — so
the real HTTP client, a scripted :class:`FakeChatClient`, and a
:class:`CachingChatClient` decorator are interchangeable. **Tests never touch the
network**: they inject a fake; the only live path is :class:`OpenRouterClient`,
exercised manually behind the harness's ``--live`` flag.

OpenRouter speaks the OpenAI-compatible ``POST /chat/completions`` shape. The
request carries ``model`` + ``messages`` and optional ``response_format`` (a JSON
schema, see :mod:`bgrl.llm.parse`), ``reasoning`` (``{"effort": ...}``), and
``seed``. The response carries ``choices[0].message.content`` and a ``usage`` block
(token counts plus an optional USD ``cost``) that the harness sums to enforce its
budget. Models that do not support ``response_format`` reject the request with a
``400``; that surfaces as :class:`StructuredOutputUnsupported` so the agent can
fall back to plain-text index parsing rather than counting it as a model failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message; ``role`` is ``"system"``/``"user"``/``"assistant"``."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ChatParams:
    """The per-request knobs the sweep varies.

    ``response_format`` and ``reasoning`` are raw JSON blocks (or ``None``); because
    they are dicts this object is *not* hashable — never put it in a set/dict key,
    use :func:`cache_key` instead, which canonicalises it for the response cache.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    response_format: dict | None = None
    reasoning: dict | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token + cost accounting from one response (``cost`` is USD, 0.0 if absent)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


ZERO_USAGE = Usage(0, 0, 0, 0.0)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A completed model turn: the assistant text plus usage and metadata."""

    text: str
    usage: Usage
    model: str
    finish_reason: str | None = None


@runtime_checkable
class ChatClient(Protocol):
    """A thing that turns a chat request into a :class:`ChatResponse`."""

    def complete(self, messages: list[ChatMessage], params: ChatParams) -> ChatResponse: ...


# --------------------------------------------------------------------------- errors


class OpenRouterError(RuntimeError):
    """An OpenRouter request failed. ``status`` is the HTTP code (``None`` if no response)."""

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        super().__init__(f"OpenRouter error (status={status}): {message}")


class RateLimitError(OpenRouterError):
    """A ``429`` that survived the client's bounded retries."""


class StructuredOutputUnsupported(OpenRouterError):
    """A ``400`` indicating the model/provider rejected ``response_format``.

    The agent catches this and retries the same prompt without structured output,
    so an unsupported model degrades to text-index parsing instead of failing.
    """


# ----------------------------------------------------------------- request/response


def build_request_body(messages: list[ChatMessage], params: ChatParams) -> dict:
    """Build the OpenAI-compatible request body, omitting unset optional fields."""
    body: dict = {
        "model": params.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
    }
    if params.seed is not None:
        body["seed"] = params.seed
    if params.response_format is not None:
        body["response_format"] = params.response_format
    if params.reasoning is not None:
        body["reasoning"] = params.reasoning
    return body


def _coerce_content(content: object) -> str:
    """Normalise ``message.content`` (string, ``None``, or a list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts)
    return str(content)


def parse_chat_response(data: dict) -> ChatResponse:
    """Parse an OpenRouter JSON response into a :class:`ChatResponse`.

    Tolerant of a missing ``cost`` (some providers omit it) and of an absent
    ``finish_reason``; a missing ``choices`` list raises :class:`OpenRouterError`
    since there is no answer to parse.
    """
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError(None, f"response had no choices: {data!r}")
    message = choices[0].get("message") or {}
    usage_raw = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
        completion_tokens=int(usage_raw.get("completion_tokens", 0)),
        total_tokens=int(usage_raw.get("total_tokens", 0)),
        cost=float(usage_raw.get("cost", 0.0)),
    )
    return ChatResponse(
        text=_coerce_content(message.get("content")),
        usage=usage,
        model=str(data.get("model", "")),
        finish_reason=choices[0].get("finish_reason"),
    )


_STRUCTURED_MARKERS = ("response_format", "json_schema", "json schema", "structured output")


def _structured_output_rejected(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _STRUCTURED_MARKERS)


# ------------------------------------------------------------------- live HTTP client


class OpenRouterClient:
    """Synchronous OpenRouter client with bounded retries and usage accounting.

    Synchronous on purpose: the WP0 :meth:`Agent.act <bgrl.agents.base.Agent.act>`
    contract and the harness loop are both synchronous, so a blocking client is the
    zero-friction fit. An async web server (WP3) calls it from a worker thread
    (``run_in_threadpool``); that wiring is a post-merge seam, not this module's job.

    Retries ``429``/``5xx``/timeouts with exponential backoff (honouring
    ``Retry-After``); other ``4xx`` fail fast, with ``response_format`` rejections
    mapped to :class:`StructuredOutputUnsupported`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        app_title: str = "bgrl-wp4",
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.environ.get(_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{_API_KEY_ENV} is not set; export your OpenRouter API key or pass api_key="
            )
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._headers = {
            "Authorization": f"Bearer {key}",
            "X-Title": app_title,
            "Content-Type": "application/json",
        }
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def complete(self, messages: list[ChatMessage], params: ChatParams) -> ChatResponse:
        body = build_request_body(messages, params)
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=body, headers=self._headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self._max_retries:
                    self._backoff(attempt, None)
                    continue
                raise OpenRouterError(None, f"transport failure: {exc}") from exc

            status = resp.status_code
            if status == 200:
                return parse_chat_response(resp.json())
            if status == 429 or status >= 500:
                if attempt < self._max_retries:
                    self._backoff(attempt, resp.headers.get("retry-after"))
                    continue
                text = _error_text(resp)
                if status == 429:
                    raise RateLimitError(status, text)
                raise OpenRouterError(status, text)

            text = _error_text(resp)
            # A 400 on a request that carried ``response_format`` means the model/provider
            # can't honour structured output; surface it as StructuredOutputUnsupported so
            # the agent downgrades to text instead of crashing. Providers vary — some return
            # a descriptive message, others a bare "Provider returned error" — so we key on
            # the request carrying a schema, not only on the error wording. A genuine non-
            # structured 400 resurfaces on the text retry (which sends no response_format).
            structured_attempt = params.response_format is not None
            if status == 400 and (structured_attempt or _structured_output_rejected(text)):
                raise StructuredOutputUnsupported(status, text)
            raise OpenRouterError(status, text)

        # The loop always returns or raises above; this is an unreachable safety net.
        raise OpenRouterError(None, "retries exhausted")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after is not None:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
        time.sleep(min(self._backoff_base * (2**attempt), self._backoff_cap))

    def close(self) -> None:
        self._client.close()


def _error_text(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return resp.text
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and "message" in err:
            return str(err["message"])
    return resp.text


# ------------------------------------------------------------------------ fake client


class FakeChatClient:
    """A scripted, network-free :class:`ChatClient` for tests and offline runs.

    Each queued item is consumed per call, in order, and may be a ``str`` (becomes
    the response text), a :class:`ChatResponse`, or an ``Exception`` (raised — to
    exercise error paths such as :class:`StructuredOutputUnsupported`). Alternatively
    pass ``responder=fn`` to compute a response from ``(messages, params)``. Every
    call is recorded in :attr:`calls` for assertions about what was sent.
    """

    def __init__(
        self,
        responses: list[str | ChatResponse | Exception] | None = None,
        *,
        responder: object = None,
        default_usage: Usage = ZERO_USAGE,
        model: str = "fake-model",
    ) -> None:
        self._queue: list[str | ChatResponse | Exception] = list(responses or [])
        self._responder = responder
        self._default_usage = default_usage
        self._model = model
        self.calls: list[tuple[list[ChatMessage], ChatParams]] = []

    def complete(self, messages: list[ChatMessage], params: ChatParams) -> ChatResponse:
        self.calls.append((messages, params))
        if self._responder is not None:
            result = self._responder(messages, params)  # type: ignore[operator]
            return self._coerce(result)
        if not self._queue:
            raise AssertionError("FakeChatClient ran out of scripted responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return self._coerce(item)

    def _coerce(self, item: str | ChatResponse) -> ChatResponse:
        if isinstance(item, ChatResponse):
            return item
        return ChatResponse(text=item, usage=self._default_usage, model=self._model)


# --------------------------------------------------------------------- caching layer


def cache_key(messages: list[ChatMessage], params: ChatParams) -> str:
    """Stable hash of a request for the response cache.

    Sensitive to everything that changes the answer — model, full message text,
    temperature, max_tokens, seed, reasoning, response_format — so a cached hit is
    only reused for an identical request. Cache hits are only *trustworthy* for
    deterministic calls (``temperature=0`` + fixed ``seed``); the sweep pins both.
    """
    payload = {
        "model": params.model,
        "messages": [(m.role, m.content) for m in messages],
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "seed": params.seed,
        "reasoning": params.reasoning,
        "response_format": params.response_format,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _response_to_dict(resp: ChatResponse) -> dict:
    return {
        "text": resp.text,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "cost": resp.usage.cost,
        },
        "model": resp.model,
        "finish_reason": resp.finish_reason,
    }


def _response_from_dict(data: dict) -> ChatResponse:
    return ChatResponse(
        text=data["text"],
        usage=Usage(**data["usage"]),
        model=data["model"],
        finish_reason=data.get("finish_reason"),
    )


class ResponseCache:
    """An in-memory request→response cache, optionally persisted as JSON-lines.

    With a ``path`` the cache is loaded on construction and each ``put`` appends one
    line, so an interrupted sweep keeps every paid response and a re-run skips them.
    Later lines override earlier ones for the same key.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._store: dict[str, ChatResponse] = {}
        if self._path is not None and self._path.exists():
            self._load()

    def get(self, key: str) -> ChatResponse | None:
        return self._store.get(key)

    def put(self, key: str, resp: ChatResponse) -> None:
        self._store[key] = resp
        if self._path is not None:
            with self._path.open("a") as fh:
                fh.write(json.dumps({"key": key, "response": _response_to_dict(resp)}) + "\n")

    def _load(self) -> None:
        assert self._path is not None
        with self._path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._store[record["key"]] = _response_from_dict(record["response"])


class CachingChatClient:
    """Wraps any :class:`ChatClient`, serving identical requests from a cache.

    Orthogonal to the underlying client, so it caches the real client during a sweep
    and is trivially verified against a :class:`FakeChatClient` (a hit must not reach
    the inner client).
    """

    def __init__(self, inner: ChatClient, cache: ResponseCache | None = None) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else ResponseCache()
        self.hits = 0
        self.misses = 0

    def complete(self, messages: list[ChatMessage], params: ChatParams) -> ChatResponse:
        key = cache_key(messages, params)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        resp = self._inner.complete(messages, params)
        self._cache.put(key, resp)
        return resp


# --------------------------------------------------------------------- budget guard


class BudgetExceededError(RuntimeError):
    """Raised by :class:`BudgetGuardClient` once a run hits its cost or call cap.

    Carries the spend and call count reached so a caller (e.g. ``scripts/eval_llm.py``)
    can stop a long match mid-flight and still report the partial stats. ``cap_usd`` is
    the configured ceiling (``inf`` when only a call cap is set).
    """

    def __init__(self, spent: float, cap_usd: float, calls: int) -> None:
        self.spent = spent
        self.cap_usd = cap_usd
        self.calls = calls
        super().__init__(
            f"LLM budget cap reached: spent ${spent:.4f} in {calls} calls (cap ${cap_usd:.4f})"
        )


class BudgetGuardClient:
    """Wraps a :class:`ChatClient` and aborts once spend or call count crosses a cap.

    The cap is checked *before* each call, so every response it returns is fully paid
    for and accounted; at most the one call that crosses the line is allowed through
    before the next call raises :class:`BudgetExceededError`. Wrap the *real* client
    with this and keep the cache *outside* it
    (``CachingChatClient(BudgetGuardClient(real))``) so cache hits — which cost nothing
    new — never count against the budget.

    ``cap_usd`` relies on the provider reporting per-response ``cost``; ``max_calls`` is
    a provider-independent backstop for when ``cost`` is absent (it stays ``0.0``).
    """

    def __init__(
        self,
        inner: ChatClient,
        *,
        cap_usd: float = float("inf"),
        max_calls: int | None = None,
    ) -> None:
        self._inner = inner
        self._cap_usd = cap_usd
        self._max_calls = max_calls
        self.spent = 0.0
        self.calls = 0

    def complete(self, messages: list[ChatMessage], params: ChatParams) -> ChatResponse:
        if self.spent >= self._cap_usd or (
            self._max_calls is not None and self.calls >= self._max_calls
        ):
            raise BudgetExceededError(self.spent, self._cap_usd, self.calls)
        resp = self._inner.complete(messages, params)
        self.calls += 1
        self.spent += resp.usage.cost
        return resp
