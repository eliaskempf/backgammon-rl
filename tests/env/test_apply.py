"""Unit tests for the single-movement primitive and checker conservation."""

from bgrl.env import BAR, OFF, EnvState, Player, SubMove, apply_submove, initial_board
from bgrl.env.board import CHECKERS_PER_SIDE, total_checkers

WHITE = Player.WHITE
BLACK = Player.BLACK


def _state(board, bar=(0, 0), off=(0, 0), turn=WHITE):
    return EnvState(board=tuple(board), bar=bar, off=off, turn=turn)


def test_initial_conservation():
    s = _state(initial_board())
    assert total_checkers(s, WHITE) == CHECKERS_PER_SIDE
    assert total_checkers(s, BLACK) == CHECKERS_PER_SIDE


def test_normal_move_white_decreases_index():
    s = _state(initial_board(), turn=WHITE)
    s2 = apply_submove(s, WHITE, SubMove(23, 20))  # die 3 toward home
    assert s2.board[23] == 1
    assert s2.board[20] == 1
    assert total_checkers(s2, WHITE) == CHECKERS_PER_SIDE
    assert s2.turn == WHITE  # apply never flips the turn


def test_hit_sends_blot_to_bar():
    board = [0] * 24
    board[1] = -1  # BLACK checker that will do the hitting
    board[4] = 1  # WHITE blot on point 4 (to be hit)
    board[0] = -14  # rest of BLACK parked so BLACK conserves to 15
    s = _state(board, off=(14, 0), turn=BLACK)
    s2 = apply_submove(s, BLACK, SubMove(1, 4))  # BLACK die 3 lands on the blot
    assert s2.board[4] == -1  # now a single BLACK checker
    assert s2.bar[WHITE] == 1  # the WHITE blot went to the bar
    assert s2.board[1] == 0
    assert total_checkers(s2, WHITE) == CHECKERS_PER_SIDE
    assert total_checkers(s2, BLACK) == CHECKERS_PER_SIDE


def test_bar_entry_white():
    board = [0] * 24
    board[5] = 14  # 14 WHITE home
    board[18] = -15  # BLACK parked
    s = _state(board, bar=(1, 0), turn=WHITE)
    s2 = apply_submove(s, WHITE, SubMove(BAR, 23))  # enter with die 1 -> point 23
    assert s2.bar[WHITE] == 0
    assert s2.board[23] == 1
    assert total_checkers(s2, WHITE) == CHECKERS_PER_SIDE


def test_bear_off_white():
    board = [0] * 24
    board[5] = 15  # all WHITE on the 6-point
    board[18] = -15  # BLACK parked
    s = _state(board, turn=WHITE)
    s2 = apply_submove(s, WHITE, SubMove(5, OFF))  # bear off with a 6
    assert s2.off[WHITE] == 1
    assert s2.board[5] == 14
    assert total_checkers(s2, WHITE) == CHECKERS_PER_SIDE
