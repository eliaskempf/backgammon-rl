"""The perspective invariant: a position and its colour-mirror encode identically."""

import numpy as np

from bgrl.env import encode
from tests.env.helpers import mirror, reachable_states


def test_mirror_invariance():
    for state, _dice in reachable_states(seed=7, n=300):
        f = encode(state, state.turn)
        g = encode(mirror(state), mirror(state).turn)
        assert np.array_equal(f, g)
