"""Tests for ``legal_orderings`` — every legal submove order per afterstate.

These back the web-UI fix where a human builds a play one click at a time and may enter
its submoves in any legal order (not just the backend's canonical one). The env stays the
source of truth: ``legal_orderings`` must agree with ``legal_moves`` on the afterstate set
and the canonical ordering, and every ordering it returns must be executable.
"""

from bgrl.env import Env, EnvState, Player, apply_submove, legal_moves, legal_orderings
from bgrl.env.movegen import _seq_key
from bgrl.env.types import OFF, SubMove

WHITE, BLACK = Player.WHITE, Player.BLACK


def _start(turn):
    s = Env.initial_state()
    return EnvState(board=s.board, bar=s.bar, off=s.off, turn=turn)


def _replay(state, mover, ordering):
    cur = state
    for sm in ordering:
        cur = apply_submove(cur, mover, sm)
    return (cur.board, cur.bar, cur.off)


# A spread of positions: openings, doubles, bear-off, mid-game, and a forced bar entry.
def _bearoff_white():
    board = [0] * 24
    board[3], board[4], board[5] = 1, 12, 1
    board[23] = -15
    return EnvState(board=tuple(board), bar=(0, 0), off=(1, 0), turn=WHITE)


def _midgame_white():
    board = [0] * 24
    board[12], board[7], board[6], board[5], board[14], board[4] = 5, 3, 1, 4, 1, 1
    board[16], board[18], board[11], board[22] = -3, -5, -5, -2
    return EnvState(board=tuple(board), bar=(0, 0), off=(0, 0), turn=WHITE)


CASES = [
    (_start(WHITE), (3, 1)),
    (_start(WHITE), (6, 6)),
    (_start(BLACK), (5, 5)),
    (_start(BLACK), (4, 2)),
    (_bearoff_white(), (6, 3)),
    (_midgame_white(), (6, 1)),
]


def test_orderings_agree_with_legal_moves():
    """Same afterstates; the canonical Move is the lexicographically smallest ordering."""
    for state, dice in CASES:
        lm = legal_moves(state, dice)
        lo = legal_orderings(state, dice)
        assert {(a.board, a.bar, a.off) for _m, a in lm} == set(lo.keys())
        for move, after in lm:
            orderings = lo[(after.board, after.bar, after.off)]
            assert move.submoves == orderings[0]  # canonical == lex-min ordering
            assert orderings == sorted(orderings, key=_seq_key)


def test_every_ordering_is_executable_maximal_and_distinct():
    for state, dice in CASES:
        for key, orderings in legal_orderings(state, dice).items():
            lengths = {len(o) for o in orderings}
            assert len(lengths) == 1  # all maximal -> same length
            assert len(orderings) == len({tuple(o) for o in orderings})  # no duplicates
            for o in orderings:
                assert _replay(state, state.turn, o) == key  # reaches its afterstate


def test_bearoff_point6_can_bear_off_in_any_order():
    # Regression for bug-examples/3.png: with a single checker on the 6-point (idx 5) and
    # dice 6&3, clicking it must be able to bear off (6/off) regardless of click order, so
    # 6/off (SubMove(5, OFF)) must appear as a *first* submove of some ordering.
    state, dice = _bearoff_white(), (6, 3)
    first_submoves = {
        o[0] for orderings in legal_orderings(state, dice).values() for o in orderings
    }
    assert SubMove(5, OFF) in first_submoves


def test_midgame_play_offers_both_submove_orders():
    # Regression for bug-examples/1.png: the play moving both 13/7 (SubMove(12, 6)) and
    # 8/7 (SubMove(7, 6)) must be reachable in either order, so neither submove is blocked
    # by having played the other first.
    state, dice = _midgame_white(), (6, 1)
    a, b = SubMove(12, 6), SubMove(7, 6)
    both = [
        orderings
        for orderings in legal_orderings(state, dice).values()
        if any({a, b} <= set(o) for o in orderings)
    ]
    assert both, "expected a play moving both 13/7 and 8/7"
    orderings = both[0]
    assert (a, b) in {tuple(o) for o in orderings}
    assert (b, a) in {tuple(o) for o in orderings}
