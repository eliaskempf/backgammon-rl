"""Tests for the board renderers, move description, and mover-relative perspective."""

from __future__ import annotations

from bgrl.env import BAR, OFF, Env, Move, Player, SubMove
from bgrl.llm.render import (
    ALL_RENDERERS,
    AsciiBoardRenderer,
    MoveListRenderer,
    PipListRenderer,
    PositionIdRenderer,
    _cell,
    describe_move,
    pip_count,
    point_number,
)

INITIAL = Env.initial_state()
DICE = (3, 1)


def test_point_number_mover_relative():
    assert point_number(0, Player.WHITE) == 1
    assert point_number(23, Player.WHITE) == 24
    assert point_number(23, Player.BLACK) == 1
    assert point_number(0, Player.BLACK) == 24


def test_describe_move_white_and_black():
    assert describe_move(Move((SubMove(23, 17),)), Player.WHITE) == "24/18"
    assert describe_move(Move((SubMove(0, 6),)), Player.BLACK) == "24/18"
    assert describe_move(Move((SubMove(BAR, 20),)), Player.WHITE) == "bar/21"
    assert describe_move(Move((SubMove(3, OFF),)), Player.WHITE) == "4/off"
    assert describe_move(Move((SubMove(23, 20), SubMove(12, 10))), Player.WHITE) == "24/21 13/11"
    assert describe_move(Move(()), Player.WHITE) == "(pass)"


def test_pip_count_initial():
    assert pip_count(INITIAL, Player.WHITE) == 167
    assert pip_count(INITIAL, Player.BLACK) == 167


def test_pip_list_golden():
    expected = (
        "Your checkers (X), moving 24->1 toward home: 24:2, 13:5, 8:3, 6:5\n"
        "Opponent checkers (O): 19:5, 17:3, 12:5, 1:2\n"
        "Your bar: 0, off: 0\n"
        "Opponent bar: 0, off: 0\n"
        "Pip count - you: 167, opponent: 167"
    )
    assert PipListRenderer().render(INITIAL, DICE, Player.WHITE) == expected


def test_position_id_golden():
    expected = "BGR:2,0,0,0,0,-5,0,-3,0,0,0,5,-5,0,0,0,3,0,5,0,0,0,0,-2|bar:0,0|off:0,0"
    assert PositionIdRenderer().render(INITIAL, DICE, Player.WHITE) == expected


def test_ascii_board_structure():
    out = AsciiBoardRenderer().render(INITIAL, DICE, Player.WHITE)
    assert "13 14 15 16 17 18 19 20 21 22 23 24" in out
    assert "BAR" in out
    assert "pip: you 167, opp 167" in out
    assert "X" in out and "O" in out


def test_ascii_cell_marks_and_overflow():
    assert _cell(6, 0, {6: 7}, {}) == " X"  # your checker mark
    assert _cell(6, 4, {6: 7}, {}) == " 7"  # 6+ checkers collapse the outer row to a count
    assert _cell(6, 0, {}, {6: 3}) == " O"  # opponent mark
    assert _cell(6, 4, {}, {6: 3}) == "  "  # above the stack -> blank
    assert _cell(6, 0, {}, {}) == " ."  # empty point


def test_renderers_are_mover_relative():
    # The opening position is symmetric under colour+axis swap, so each renderer
    # produces the identical string from either side's point of view.
    for renderer in (
        AsciiBoardRenderer(),
        PipListRenderer(),
        MoveListRenderer(),
        PositionIdRenderer(),
    ):
        white = renderer.render(INITIAL, DICE, Player.WHITE)
        black = renderer.render(INITIAL, DICE, Player.BLACK)
        assert white == black, renderer.name


def test_registry_has_all_renderers():
    assert set(ALL_RENDERERS) == {"ascii", "pip_list", "moves_only", "position_id"}
    assert ALL_RENDERERS["pip_list"].name == "pip_list"
