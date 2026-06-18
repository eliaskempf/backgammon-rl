"""Server-side game-session mechanics (the legality boundary)."""

import numpy as np
import pytest

from bgrl.agents import RandomAgent
from bgrl.env import Env, EnvState, ManualDiceSource, Move, Player, ReplayDiceSource, SubMove
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


def test_manual_session_consumes_supplied_dice():
    session = _session(ManualDiceSource())
    assert session.is_manual
    session.supply_dice((3, 1))
    assert session.roll() == (3, 1)


def test_supply_dice_rejected_for_auto_session():
    session = _session(ReplayDiceSource([(3, 1)]))
    assert not session.is_manual
    with pytest.raises(GameError, match="manual-dice"):
        session.supply_dice((3, 1))


def test_manual_roll_without_supplied_dice_raises():
    session = _session(ManualDiceSource())
    with pytest.raises(RuntimeError, match="empty"):
        session.roll()  # nothing queued yet


def test_undo_returns_to_last_human_decision_with_same_dice():
    session = _session(ReplayDiceSource([(3, 1), (5, 2)]))
    session.roll()
    pre_state, pre_dice = session.state, session.dice
    session.apply_move(session.move_for_id(0))  # human ply
    session.play_agent()  # agent replies; turn back to WHITE
    assert session.to_act is WHITE
    assert session.can_undo

    session.undo()
    assert session.state == pre_state
    assert session.dice == pre_dice == (3, 1)
    assert session.legal == Env.legal_moves(session.state, session.dice)
    assert session.steps == []  # both plies dropped
    assert not session.can_undo  # nothing left to revert


def test_undo_repeats_across_turns():
    session = _session(ReplayDiceSource([(3, 1), (5, 2), (6, 4), (2, 3)]))
    session.roll()
    h1_state = session.state
    session.apply_move(session.move_for_id(0))
    session.play_agent()
    session.roll()
    h2_state = session.state
    session.apply_move(session.move_for_id(0))
    session.play_agent()
    assert session.to_act is WHITE

    session.undo()  # back to the second human decision
    assert session.state == h2_state
    assert session.dice == (6, 4)

    session.undo()  # back to the first human decision
    assert session.state == h1_state
    assert session.dice == (3, 1)
    assert not session.can_undo


def test_undo_with_no_history_raises():
    session = _session(ReplayDiceSource([(3, 1)]))
    assert not session.can_undo
    with pytest.raises(GameError, match="nothing to undo"):
        session.undo()


def test_undo_skips_a_forced_human_pass():
    # A lone forced human pass is not a redoable decision: can_undo is False.
    session = _session(ReplayDiceSource([(2, 4)]))
    session.state = _white_bar_blocked_state()
    session.roll()
    session.apply_pass()
    assert session.steps[0].move == Move(submoves=())
    assert not session.can_undo
    with pytest.raises(GameError, match="nothing to undo"):
        session.undo()


def test_undo_in_manual_mode_restores_dice_without_requeueing():
    session = _session(ManualDiceSource())
    session.supply_dice((3, 1))
    session.roll()
    session.apply_move(session.move_for_id(0))
    session.supply_dice((5, 2))
    session.play_agent()
    assert session.to_act is WHITE

    session.undo()
    assert session.dice == (3, 1)
    assert session.legal
    # The restored roll is live, so a fresh roll is neither needed nor allowed.
    with pytest.raises(GameError, match="already rolled"):
        session.roll()


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
