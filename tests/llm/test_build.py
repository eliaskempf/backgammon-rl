"""Unit tests for the shared LLM agent/client builders (no network)."""

from __future__ import annotations

import pytest

from bgrl.agents.llm_agent import AgentStats, LLMAgent
from bgrl.llm.build import build_chat_client, build_llm_agent, reasoning_option
from bgrl.llm.client import ChatMessage, ChatParams


def test_reasoning_option():
    assert reasoning_option("off") is None
    assert reasoning_option("medium") == {"effort": "medium"}


def test_build_chat_client_offline_has_no_guard_and_answers_first_legal():
    client, guard = build_chat_client(live=False)
    assert guard is None
    resp = client.complete([ChatMessage("user", "pick")], ChatParams(model="m"))
    assert resp.text == "0"  # the fake always plays index 0


def test_build_llm_agent_resolves_names_and_wires_stats():
    client, _ = build_chat_client(live=False)
    stats = AgentStats()
    agent = build_llm_agent(client, model="m", renderer="ascii", template="terse", stats=stats)
    assert isinstance(agent, LLMAgent)
    assert agent.stats is stats


def test_build_llm_agent_rejects_unknown_names():
    client, _ = build_chat_client(live=False)
    with pytest.raises(KeyError):
        build_llm_agent(client, model="m", renderer="does-not-exist")
    with pytest.raises(ValueError):
        build_llm_agent(client, model="m", output_format="nonsense")
