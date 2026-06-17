"""Tests for the LLM agent's parse/validate/re-prompt/fallback pipeline.

Everything runs against a scripted FakeChatClient — no network, no live API calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from bgrl.agents.base import Agent
from bgrl.agents.llm_agent import Fallback, LLMAgent
from bgrl.env import Env, RandomDiceSource
from bgrl.game import play_game
from bgrl.llm.client import ChatResponse, FakeChatClient, StructuredOutputUnsupported, Usage
from bgrl.llm.parse import OutputFormat
from bgrl.llm.prompt import TERSE
from bgrl.llm.render import PipListRenderer

STATE = Env.initial_state()
DICE = (3, 1)
LEGAL = Env.legal_moves(STATE, DICE)


def _agent(client, **kwargs):
    return LLMAgent(client, model="m", renderer=PipListRenderer(), template=TERSE, **kwargs)


def test_implements_agent_protocol():
    assert isinstance(_agent(FakeChatClient(["0"])), Agent)


def test_valid_first_try_returns_indexed_move():
    agent = _agent(FakeChatClient([str(len(LEGAL) - 1)]))
    chosen = agent.act(STATE, DICE, LEGAL)
    assert chosen is LEGAL[-1][0]  # exact object from the list -> passes play_game's guard
    assert agent.stats.decisions == 1
    assert agent.stats.api_calls == 1
    assert agent.stats.invalid_responses == 0


def test_out_of_range_then_valid_reprompts_once():
    agent = _agent(FakeChatClient(["999", "1"]))
    chosen = agent.act(STATE, DICE, LEGAL)
    assert chosen is LEGAL[1][0]
    assert agent.stats.api_calls == 2
    assert agent.stats.invalid_responses == 1
    assert agent.stats.reprompts == 1


def test_exhausted_reprompts_falls_back_to_first_legal():
    agent = _agent(FakeChatClient(["nope", "still nope", "prose only"]), max_reprompts=2)
    chosen = agent.act(STATE, DICE, LEGAL)
    assert chosen is LEGAL[0][0]
    assert agent.stats.api_calls == 3
    assert agent.stats.invalid_responses == 3
    assert agent.stats.fallbacks == 1


def test_random_legal_fallback_is_seeded_and_legal():
    rng = np.random.default_rng(0)
    agent = _agent(
        FakeChatClient(["x", "y", "z"]),
        max_reprompts=2,
        fallback=Fallback.RANDOM_LEGAL,
        rng=rng,
    )
    chosen = agent.act(STATE, DICE, LEGAL)
    assert chosen in {move for move, _ in LEGAL}
    assert agent.stats.fallbacks == 1


def test_random_legal_fallback_requires_rng():
    with pytest.raises(ValueError, match="rng"):
        _agent(FakeChatClient([]), fallback=Fallback.RANDOM_LEGAL)


def test_structured_format_sends_response_format():
    client = FakeChatClient(["0"])
    _agent(client, output_format=OutputFormat.STRUCTURED).act(STATE, DICE, LEGAL)
    _, params = client.calls[0]
    assert params.response_format is not None
    assert params.response_format["type"] == "json_schema"


def test_index_text_sends_no_response_format():
    client = FakeChatClient(["0"])
    _agent(client, output_format=OutputFormat.INDEX_TEXT).act(STATE, DICE, LEGAL)
    assert client.calls[0][1].response_format is None


def test_structured_unsupported_downgrades_to_text():
    client = FakeChatClient([StructuredOutputUnsupported(400, "no response_format"), "0"])
    agent = _agent(client, output_format=OutputFormat.STRUCTURED)
    chosen = agent.act(STATE, DICE, LEGAL)
    assert chosen is LEGAL[0][0]
    assert agent.stats.structured_unsupported is True
    # the downgrade discovery call is not counted as a model answer
    assert agent.stats.api_calls == 1
    # first attempt asked for structured output; the retry asked for plain text
    assert client.calls[0][1].response_format is not None
    assert client.calls[1][1].response_format is None


def test_text_mode_does_not_swallow_unexpected_structured_error():
    client = FakeChatClient([StructuredOutputUnsupported(400, "weird")])
    agent = _agent(client, output_format=OutputFormat.INDEX_TEXT)
    with pytest.raises(StructuredOutputUnsupported):
        agent.act(STATE, DICE, LEGAL)


def test_stats_accumulate_usage():
    client = FakeChatClient([ChatResponse("0", Usage(10, 2, 12, 0.001), "m")])
    agent = _agent(client)
    agent.act(STATE, DICE, LEGAL)
    assert agent.stats.total_cost == 0.001
    assert agent.stats.total_prompt_tokens == 10
    assert agent.stats.total_completion_tokens == 2


def test_full_game_only_plays_legal_moves():
    # A responder that always picks the first candidate drives a full, legal game via
    # play_game; the driver raises if any returned move is not in the legal set.
    def first_legal(messages, params):
        return "0"

    white = _agent(FakeChatClient(responder=first_legal))
    black = _agent(FakeChatClient(responder=first_legal))
    result = play_game(white, black, RandomDiceSource(np.random.default_rng(7)))
    assert result.outcome is not None
    assert result.plies > 0
    # Every act decision is a ply; forced passes are plies without an act call, so the
    # decision total is positive and never exceeds the ply count.
    decisions = white.stats.decisions + black.stats.decisions
    assert 0 < decisions <= result.plies
