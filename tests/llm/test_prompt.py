"""Tests for prompt composition."""

from __future__ import annotations

from bgrl.env import Env
from bgrl.llm.parse import OutputFormat
from bgrl.llm.prompt import ALL_TEMPLATES, TERSE
from bgrl.llm.render import PipListRenderer

STATE = Env.initial_state()
DICE = (3, 1)
LEGAL = Env.legal_moves(STATE, DICE)


def test_build_produces_system_and_user_messages():
    msgs = TERSE.build(STATE, DICE, LEGAL, PipListRenderer(), OutputFormat.INDEX_TEXT)
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == TERSE.system


def test_user_message_contains_board_dice_enumerated_moves_and_format():
    user = TERSE.build(STATE, DICE, LEGAL, PipListRenderer(), OutputFormat.INDEX_TEXT)[1].content
    assert "You rolled 3-1." in user
    assert "Pip count - you: 167" in user  # board rendering is embedded
    # moves are enumerated from 0; index n-1 is the last candidate
    assert "0: " in user
    assert f"{len(LEGAL) - 1}: " in user
    assert f"integer index (0 to {len(LEGAL) - 1})" in user  # format instruction embedded


def test_structured_format_changes_the_instruction():
    user = TERSE.build(STATE, DICE, LEGAL, PipListRenderer(), OutputFormat.STRUCTURED)[1].content
    assert '{"choice": <index>}' in user


def test_registry_has_templates():
    assert set(ALL_TEMPLATES) == {"terse", "coach"}
