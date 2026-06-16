"""Property-based move-gen invariants over reachable positions."""

from hypothesis import given, settings
from hypothesis import strategies as st

from bgrl.env import apply_submove, legal_moves
from bgrl.env.board import CHECKERS_PER_SIDE, total_checkers
from tests.env.helpers import reachable_states


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_movegen_properties(seed):
    for state, dice in reachable_states(seed=seed, n=25):
        moves = legal_moves(state, dice)
        if not moves:
            continue

        # All returned plays use the same (maximal) number of dice.
        assert len({len(m.submoves) for m, _a in moves}) == 1

        # Afterstates are pairwise distinct (de-duplicated).
        afters = [a for _m, a in moves]
        assert len({(a.board, a.bar, a.off) for a in afters}) == len(afters)

        for move, after in moves:
            # The canonical Move is executable and reproduces its afterstate.
            cur = state
            for sm in move.submoves:
                cur = apply_submove(cur, state.turn, sm)
            assert (cur.board, cur.bar, cur.off) == (after.board, after.bar, after.off)

            # Turn flips to the opponent; reserved cube fields untouched.
            assert after.turn == state.turn.opponent()
            assert after.cube_value == state.cube_value
            assert after.cube_owner == state.cube_owner

            # Checker conservation holds for both sides.
            assert total_checkers(after, state.turn) == CHECKERS_PER_SIDE
            assert total_checkers(after, state.turn.opponent()) == CHECKERS_PER_SIDE
