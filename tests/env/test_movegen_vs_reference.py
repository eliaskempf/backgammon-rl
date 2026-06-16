"""Differential test against the vendored gym-backgammon oracle.

We assert containment, not equality: the reference's bear-off-doubles enumerator
misses some legal max-dice plays our DFS finds (hand-verified — see the golden
test ``test_doubles_bear_off_superset_of_oracle_gap``), so the reference set is a
subset of ours. Over-generation is guarded separately by the replay / legality /
conservation properties in ``test_movegen_properties``.
"""

import pytest

from bgrl.env import legal_moves
from tests.env.helpers import OracleUnsupported, oracle_after_keys, reachable_states


@pytest.mark.slow
def test_oracle_plays_are_a_subset_of_ours():
    checked = 0
    unsupported = 0
    for state, dice in reachable_states(seed=99, n=2500):
        mine = {(a.board, a.bar, a.off) for _m, a in legal_moves(state, dice)}
        try:
            theirs = oracle_after_keys(state, dice)
        except OracleUnsupported:
            unsupported += 1
            continue
        checked += 1
        missing = theirs - mine
        assert not missing, f"oracle found plays we miss: state={state} dice={dice} missing={missing}"

    assert checked > 1000  # ensure we actually exercised the oracle
