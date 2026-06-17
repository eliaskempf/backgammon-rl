"""Server-side game-session mechanics (the legality boundary)."""

import numpy as np
import pytest

from bgrl.agents import RandomAgent
from bgrl.env import Env, EnvState, Move, Player, ReplayDiceSource, SubMove
from bgrl.web.session import GameError, GameSession, IllegalMove

WHITE = Player.WHITE
BLACK = Player.BLACK


def _session(dice_source, *, human_seat=WHITE):
    return GameSession(
        "test",
        opponent=RandomAgent(np.random.default_rng(0)),
        opponent_name="random",
        human_seat=human_seat,
        dice_source=dice_source,
    )


def _white_bar_blocked_state():
    """WHITE has a checker on the bar; BLACK owns all entry points (18..23)."""
    board = [0] * 24
    board[0] = 14  # WHITE's other 14 checkers, parked out of the way
    for i in range(18, 24):
        board[i] = -2  # BLACK blocks every bar-entry point (24 - die for die 1..6)
    board[17] = -3  # the remaining 3 BLACK checkers
    return EnvState(board=tuple(board), bar=(1, 0), off=(0, 0), turn=WHITE)


def test_forced_pass_flips_turn_and_records_a_pass():
    session = _session(ReplayDiceSource([(2, 4)]))
    session.state = _white_bar_blocked_state()

    session.roll()
    assert session.legal == []  # genuinely blocked, not a mock

    session.apply_pass()
    assert session.to_act is BLACK
    assert session.dice is None
    assert len(session.steps) == 1
    assert session.steps[0].move == Move(submoves=())  # the recorded PASS


def test_roll_then_legal_move_advances_and_records():
    session = _session(ReplayDiceSource([(3, 1)]))
    session.roll()
    assert session.dice == (3, 1)
    assert session.legal  # the opening has legal moves

    move = session.move_for_id(0)
    before = session.state
    session.apply_move(move)

    assert session.to_act is BLACK  # turn handed to the opponent
    assert session.dice is None
    assert len(session.steps) == 1
    step = session.steps[0]
    assert step.state == before and step.move == move
    assert step.afterstate == session.state


def test_illegal_move_and_out_of_range_id_raise():
    session = _session(ReplayDiceSource([(3, 1)]))
    session.roll()
    with pytest.raises(IllegalMove):
        session.move_for_id(999)
    with pytest.raises(IllegalMove):
        session.apply_move(Move(submoves=(SubMove(99, 88),)))


def test_cannot_roll_twice_or_move_before_rolling():
    session = _session(ReplayDiceSource([(3, 1), (5, 2)]))
    with pytest.raises(GameError):
        session.apply_move(Move(submoves=()))  # no dice rolled yet
    session.roll()
    with pytest.raises(GameError):
        session.roll()  # already rolled


def test_agent_play_uses_the_env_for_legality():
    # Drive a short opponent-only sequence and confirm steps chain correctly.
    session = _session(ReplayDiceSource([(6, 5), (4, 2), (3, 1)]), human_seat=BLACK)
    for _ in range(3):
        if session.terminal:
            break
        session.play_agent()
    assert session.steps
    for earlier, later in zip(session.steps, session.steps[1:], strict=False):
        assert earlier.afterstate == later.state  # afterstate of one ply is the next state


def test_terminal_outcome_is_set_when_game_ends():
    # A position where WHITE bears off its last checker this turn (BLACK still on board).
    board = [0] * 24
    board[0] = 1  # WHITE's last checker, on the ace point
    for idx, cnt in [(18, 2), (19, 2), (20, 2), (21, 3), (22, 3), (23, 3)]:
        board[idx] = -cnt  # 15 BLACK checkers still in play
    state = EnvState(board=tuple(board), bar=(0, 0), off=(14, 0), turn=WHITE)
    assert not Env.is_terminal(state)
    session = _session(ReplayDiceSource([(1, 1)]))
    session.state = state
    session.roll()
    session.apply_move(session.move_for_id(0))
    assert session.terminal
    assert session.outcome is not None and session.outcome.winner is WHITE
