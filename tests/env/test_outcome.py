"""Terminal detection and single / gammon / backgammon classification."""

from bgrl.env import Env, EnvState, Player, WinKind, is_terminal, outcome

WHITE, BLACK = Player.WHITE, Player.BLACK


def test_initial_not_terminal():
    assert not is_terminal(Env.initial_state())
    assert outcome(Env.initial_state()) is None


def test_single_win():
    board = [0] * 24
    board[18] = -5  # BLACK still has 5 on the board
    s = EnvState(board=tuple(board), bar=(0, 0), off=(15, 10), turn=BLACK)
    o = outcome(s)
    assert o.winner == WHITE
    assert o.kind == WinKind.SINGLE


def test_gammon():
    board = [0] * 24
    board[18] = -15  # BLACK borne off none, but only in BLACK's own quadrant
    s = EnvState(board=tuple(board), bar=(0, 0), off=(15, 0), turn=BLACK)
    o = outcome(s)
    assert o.winner == WHITE
    assert o.kind == WinKind.GAMMON


def test_backgammon_checker_in_winner_home():
    board = [0] * 24
    board[3] = -15  # BLACK checker stuck in WHITE's home board (index 3)
    s = EnvState(board=tuple(board), bar=(0, 0), off=(15, 0), turn=BLACK)
    o = outcome(s)
    assert o.winner == WHITE
    assert o.kind == WinKind.BACKGAMMON


def test_backgammon_checker_on_bar():
    board = [0] * 24
    board[18] = -14
    s = EnvState(board=tuple(board), bar=(0, 1), off=(15, 0), turn=BLACK)
    o = outcome(s)
    assert o.winner == WHITE
    assert o.kind == WinKind.BACKGAMMON
