"""Tests for ``move_dice``: which die each submove of a legal play consumes."""

import pytest

from bgrl.env import Env, EnvState, Move, Player, legal_moves, move_dice
from bgrl.env.apply import apply_submove
from bgrl.env.movegen import _submoves_for_die
from bgrl.env.types import OFF, SubMove

WHITE, BLACK = Player.WHITE, Player.BLACK


def _labels_are_consistent(state: EnvState, move: Move, labels: tuple[int, ...]) -> bool:
    """Every label genuinely produces its submove from the replayed interim state."""
    if len(labels) != len(move.submoves):
        return False
    cur = state
    mover = state.turn
    for sm, die in zip(move.submoves, labels, strict=True):
        if sm not in _submoves_for_die(cur, mover, die):
            return False
        cur = apply_submove(cur, mover, sm)
    return True


def test_non_double_two_submoves_use_each_die_once():
    s = Env.initial_state()
    move = next(m for m, _ in legal_moves(s, (3, 1)) if len(m.submoves) == 2)
    labels = move_dice(s, (3, 1), move)
    assert sorted(labels) == [1, 3]
    assert _labels_are_consistent(s, move, labels)


def test_doubles_all_entries_equal_the_doubled_value():
    s = Env.initial_state()
    move = next(m for m, _ in legal_moves(s, (2, 2)) if len(m.submoves) == 4)
    assert move_dice(s, (2, 2), move) == (2, 2, 2, 2)
    assert _labels_are_consistent(s, move, move_dice(s, (2, 2), move))


def test_bear_off_overshoot_ambiguity_is_resolved_consistently():
    # Lone WHITE checker on pip-2 (index 1) with nothing higher: the bear-off
    # SubMove(1, OFF) is producible by BOTH dice (exact 2 and overshoot 6). A
    # second checker on index 10 must consume the other die, so a greedy
    # left-die-first labeller would strand a die; backtracking must not.
    board = [0] * 24
    board[1] = 1
    board[10] = 1
    board[18] = -13
    s = EnvState(board=tuple(board), bar=(0, 0), off=(13, 0), turn=WHITE)
    move = next(
        m
        for m, _ in legal_moves(s, (2, 6))
        if len(m.submoves) == 2 and any(sm.dst == OFF for sm in m.submoves)
    )
    labels = move_dice(s, (2, 6), move)
    assert sorted(labels) == [2, 6]
    assert _labels_are_consistent(s, move, labels)


def test_forced_higher_die_single_submove():
    # One WHITE checker on index 12; index 1 is blocked, so neither die's
    # continuation (12->6->1 or 12->7->1) is playable -> only one die can be used,
    # and the forced-higher-die rule keeps the 6 (12 -> 6).
    board = [0] * 24
    board[12] = 1
    board[1] = -2  # blocks the shared landing point for the second die
    board[18] = -13
    s = EnvState(board=tuple(board), bar=(0, 0), off=(14, 0), turn=WHITE)
    moves = legal_moves(s, (6, 5))
    assert len(moves) == 1
    (move, _), = moves
    assert move.submoves == (SubMove(12, 6),)
    assert move_dice(s, (6, 5), move) == (6,)


def test_empty_move_yields_empty_labels():
    s = Env.initial_state()
    assert move_dice(s, (3, 1), Move(())) == ()


def test_rejects_move_not_legal_for_roll():
    s = Env.initial_state()
    bogus = Move((SubMove(0, 5),))  # not reachable with (3, 1)
    with pytest.raises(ValueError):
        move_dice(s, (3, 1), bogus)
