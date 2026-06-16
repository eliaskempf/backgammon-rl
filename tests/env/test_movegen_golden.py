"""Golden move-gen tests: exact counts and crafted edge cases."""

from bgrl.env import Env, EnvState, Player, legal_moves
from bgrl.env.movegen import _submoves_for_die
from bgrl.env.types import BAR, OFF, SubMove

WHITE, BLACK = Player.WHITE, Player.BLACK


def _after_keys(state, dice):
    return {(a.board, a.bar, a.off) for _m, a in legal_moves(state, dice)}


def _start(turn):
    s = Env.initial_state()
    return EnvState(board=s.board, bar=s.bar, off=s.off, turn=turn)


def test_opening_counts():
    # Cross-checked against the vendored reference.
    assert len(legal_moves(_start(WHITE), (3, 1))) == 16
    assert len(legal_moves(_start(BLACK), (3, 1))) == 16
    assert len(legal_moves(_start(WHITE), (6, 6))) == 11
    assert len(legal_moves(_start(BLACK), (5, 5))) == 4


def test_no_legal_move_is_pass():
    # WHITE closed out: one checker on the bar, every entry point (18..23) blocked.
    board = [0] * 24
    for p in range(18, 24):
        board[p] = -2  # 12 BLACK
    board[0] = -3  # 15 BLACK total
    s = EnvState(board=tuple(board), bar=(1, 0), off=(14, 0), turn=WHITE)
    assert legal_moves(s, (3, 5)) == []
    assert legal_moves(s, (6, 6)) == []


def test_bear_off_exact():
    board = [0] * 24
    board[2] = 1  # lone WHITE checker on the 3-point (pip 3)
    board[18] = -15
    s = EnvState(board=tuple(board), bar=(0, 0), off=(14, 0), turn=WHITE)
    assert _submoves_for_die(s, WHITE, 3) == [SubMove(2, OFF)]


def test_bear_off_overshoot_allowed_without_higher_checker():
    board = [0] * 24
    board[1] = 1  # lone checker on pip-2 point, nothing higher
    board[18] = -15
    s = EnvState(board=tuple(board), bar=(0, 0), off=(14, 0), turn=WHITE)
    assert _submoves_for_die(s, WHITE, 6) == [SubMove(1, OFF)]  # overshoot ok


def test_bear_off_overshoot_blocked_by_higher_checker():
    board = [0] * 24
    board[1] = 1  # pip 2
    board[5] = 1  # pip 6 (higher) -> blocks overshoot from index 1
    board[18] = -15
    s = EnvState(board=tuple(board), bar=(0, 0), off=(13, 0), turn=WHITE)
    assert _submoves_for_die(s, WHITE, 6) == [SubMove(5, OFF)]  # must clear the 6-point


def test_bar_entry_forced_and_blocked():
    board = [0] * 24
    board[20] = -2  # blocks WHITE entry for die 4 (enters at 24-4 == 20)
    board[18] = -13  # 15 BLACK
    s = EnvState(board=tuple(board), bar=(1, 0), off=(14, 0), turn=WHITE)
    assert _submoves_for_die(s, WHITE, 1) == [SubMove(BAR, 23)]  # 24-1 == 23 open
    assert _submoves_for_die(s, WHITE, 4) == []  # blocked


def test_forced_higher_die():
    # One WHITE checker on index 12; a BLACK wall on index 1 blocks every two-die
    # combination (both routes run through index 1). Only single moves remain, so
    # the higher die (6) is forced: 12->6 is kept, the lower-die 12->7 is dropped.
    board = [0] * 24
    board[12] = 1
    board[1] = -2
    board[18] = -13  # 15 BLACK
    s = EnvState(board=tuple(board), bar=(0, 0), off=(14, 0), turn=WHITE)
    keys = _after_keys(s, (6, 5))
    assert len(keys) == 1
    (after,) = keys
    board_after = after[0]
    assert board_after[6] == 1
    assert board_after[12] == 0
    assert board_after[7] == 0  # the lower-die move 12->7 is not offered


def test_doubles_bear_off_superset_of_oracle_gap():
    # Position where the reference's bear-off-doubles enumerator misses legal
    # four-dice plays; verify our DFS DOES produce them (a correct superset).
    board = (7, 0, 0, 3, 2, 2, 0, 0, -1, 0, 0, 0, 0, -4, 0, 0, -1, -1, 0, 0, -2, -1, -3, -2)
    s = EnvState(board=board, bar=(0, 0), off=(1, 0), turn=WHITE)
    keys = _after_keys(s, (3, 3))
    a1 = (
        (8, 0, 1, 2, 2, 0, 0, 0, -1, 0, 0, 0, 0, -4, 0, 0, -1, -1, 0, 0, -2, -1, -3, -2),
        (0, 0),
        (2, 0),
    )
    a2 = (
        (7, 1, 1, 3, 1, 0, 0, 0, -1, 0, 0, 0, 0, -4, 0, 0, -1, -1, 0, 0, -2, -1, -3, -2),
        (0, 0),
        (2, 0),
    )
    assert a1 in keys
    assert a2 in keys
