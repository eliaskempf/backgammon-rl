"""Unit tests for the OpenRouter client seam — pure pieces only, no network.

The live ``OpenRouterClient.complete`` HTTP path is intentionally not exercised
here; we test the body builder, the response parser, the cache, and the fake.
"""

from __future__ import annotations

import pytest

from bgrl.llm.client import (
    BudgetExceededError,
    BudgetGuardClient,
    CachingChatClient,
    ChatMessage,
    ChatParams,
    ChatResponse,
    FakeChatClient,
    OpenRouterError,
    ResponseCache,
    StructuredOutputUnsupported,
    Usage,
    _structured_output_rejected,
    build_request_body,
    cache_key,
    parse_chat_response,
)

MSGS = [ChatMessage("system", "you are a bot"), ChatMessage("user", "pick a move")]


def test_build_body_drops_unset_optionals():
    body = build_request_body(MSGS, ChatParams(model="m"))
    assert body["model"] == "m"
    assert body["messages"] == [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "pick a move"},
    ]
    assert "seed" not in body
    assert "response_format" not in body
    assert "reasoning" not in body


def test_build_body_includes_set_optionals():
    rf = {"type": "json_schema", "json_schema": {"name": "choice"}}
    body = build_request_body(
        MSGS, ChatParams(model="m", seed=7, response_format=rf, reasoning={"effort": "low"})
    )
    assert body["seed"] == 7
    assert body["response_format"] == rf
    assert body["reasoning"] == {"effort": "low"}


def test_parse_response_full():
    data = {
        "model": "served/model",
        "choices": [{"message": {"content": "3"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.001},
    }
    resp = parse_chat_response(data)
    assert resp.text == "3"
    assert resp.model == "served/model"
    assert resp.finish_reason == "stop"
    assert resp.usage == Usage(10, 2, 12, 0.001)


def test_parse_response_tolerates_missing_cost_and_finish_reason():
    data = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1}}
    resp = parse_chat_response(data)
    assert resp.usage.cost == 0.0
    assert resp.usage.completion_tokens == 0
    assert resp.finish_reason is None


def test_parse_response_coerces_none_and_list_content():
    assert parse_chat_response({"choices": [{"message": {"content": None}}]}).text == ""
    listed = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    assert parse_chat_response(listed).text == "ab"


def test_parse_response_without_choices_raises():
    with pytest.raises(OpenRouterError):
        parse_chat_response({"choices": []})


def test_cache_key_stable_and_sensitive():
    p = ChatParams(model="m", seed=1)
    assert cache_key(MSGS, p) == cache_key(MSGS, p)
    assert cache_key(MSGS, p) != cache_key(MSGS, ChatParams(model="other", seed=1))
    assert cache_key(MSGS, p) != cache_key(MSGS, ChatParams(model="m", seed=2))
    other_msgs = [*MSGS, ChatMessage("user", "extra")]
    assert cache_key(MSGS, p) != cache_key(other_msgs, p)
    rf = {"type": "json_schema"}
    assert cache_key(MSGS, p) != cache_key(MSGS, ChatParams(model="m", seed=1, response_format=rf))


def test_caching_client_hit_skips_inner():
    inner = FakeChatClient(["first", "second"])
    cached = CachingChatClient(inner)
    p = ChatParams(model="m", seed=1)
    first = cached.complete(MSGS, p)
    again = cached.complete(MSGS, p)  # identical request -> served from cache
    assert first.text == "first"
    assert again.text == "first"
    assert len(inner.calls) == 1
    assert cached.hits == 1 and cached.misses == 1


def test_caching_client_persists_to_disk(tmp_path):
    path = tmp_path / "cache.jsonl"
    inner = FakeChatClient([ChatResponse("answer", Usage(5, 1, 6, 0.002), "m")])
    CachingChatClient(inner, ResponseCache(path)).complete(MSGS, ChatParams(model="m"))
    # A fresh cache loaded from disk replays the stored response without the inner client.
    reloaded = CachingChatClient(FakeChatClient([]), ResponseCache(path))
    resp = reloaded.complete(MSGS, ChatParams(model="m"))
    assert resp.text == "answer"
    assert resp.usage == Usage(5, 1, 6, 0.002)
    assert reloaded.hits == 1


def test_fake_client_scripts_text_response_and_exception():
    fake = FakeChatClient(["one", StructuredOutputUnsupported(400, "no response_format")])
    assert fake.complete(MSGS, ChatParams(model="m")).text == "one"
    with pytest.raises(StructuredOutputUnsupported):
        fake.complete(MSGS, ChatParams(model="m"))
    assert len(fake.calls) == 2


def test_fake_client_responder_mode_sees_request():
    seen = {}

    def responder(messages, params):
        seen["model"] = params.model
        return f"echo:{messages[-1].content}"

    fake = FakeChatClient(responder=responder)
    resp = fake.complete(MSGS, ChatParams(model="claude"))
    assert resp.text == "echo:pick a move"
    assert seen["model"] == "claude"


def test_structured_output_rejection_detection():
    assert _structured_output_rejected("Model does not support response_format")
    assert _structured_output_rejected("json_schema is not available for this provider")
    assert not _structured_output_rejected("rate limit exceeded")


def _priced(text, cost):
    return ChatResponse(text, Usage(1, 1, 2, cost), "m")


def test_budget_guard_blocks_after_cost_cap():
    # Cap checked before each call: the call that crosses the line still returns; the
    # next one raises. So with a $0.5 cap and $0.4/call, calls 1-2 pass, call 3 blocks.
    inner = FakeChatClient([_priced("a", 0.4), _priced("b", 0.4), _priced("c", 0.4)])
    guard = BudgetGuardClient(inner, cap_usd=0.5)
    p = ChatParams(model="m")
    assert guard.complete(MSGS, p).text == "a"
    assert guard.complete(MSGS, p).text == "b"
    with pytest.raises(BudgetExceededError):
        guard.complete(MSGS, p)
    assert guard.calls == 2
    assert guard.spent == pytest.approx(0.8)


def test_budget_guard_blocks_after_max_calls_when_cost_absent():
    # Providers may omit cost (stays 0.0); max_calls is the backstop.
    inner = FakeChatClient(responder=lambda messages, params: "ok")
    guard = BudgetGuardClient(inner, max_calls=2)
    p = ChatParams(model="m")
    guard.complete(MSGS, p)
    guard.complete(MSGS, p)
    with pytest.raises(BudgetExceededError):
        guard.complete(MSGS, p)
    assert guard.spent == 0.0


def test_caching_outside_guard_keeps_hits_free():
    # Caching(BudgetGuard(real)): a cache hit must not reach the guard, so repeated
    # identical requests neither re-spend nor count toward the call cap.
    inner = FakeChatClient([_priced("first", 0.3)])
    guard = BudgetGuardClient(inner, cap_usd=1.0)
    cached = CachingChatClient(guard)
    p = ChatParams(model="m", seed=1)
    assert cached.complete(MSGS, p).text == "first"
    assert cached.complete(MSGS, p).text == "first"  # served from cache, no new spend
    assert guard.calls == 1
    assert guard.spent == pytest.approx(0.3)
