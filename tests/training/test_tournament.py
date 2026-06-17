"""Round-robin tournament: matrix shape, self-play diagonal, anti-symmetry, determinism."""

import numpy as np

from bgrl.agents import RandomAgent
from bgrl.training.tournament import round_robin


def _agents() -> dict[str, RandomAgent]:
    # Distinct streams so the agents actually differ; labels carry insertion order.
    return {
        "a": RandomAgent(np.random.default_rng(1)),
        "b": RandomAgent(np.random.default_rng(2)),
        "c": RandomAgent(np.random.default_rng(3)),
    }


def test_shape_labels_and_diagonal():
    res = round_robin(_agents(), pairs=8, rng=np.random.default_rng(0))
    assert res.labels == ("a", "b", "c")
    assert res.win_rate.shape == (3, 3)
    assert np.all(np.diag(res.win_rate) == 0.5)  # an agent vs. itself is exactly 50%


def test_matrix_is_anti_symmetric_about_half():
    res = round_robin(_agents(), pairs=8, rng=np.random.default_rng(0))
    # CRN excludes truncated games from both denominators identically, so the
    # off-diagonal complement is exact, not approximate.
    assert np.allclose(res.win_rate + res.win_rate.T, 1.0)
    assert np.all((res.win_rate >= 0.0) & (res.win_rate <= 1.0))


def test_reproducible_for_same_seed():
    a = round_robin(_agents(), pairs=8, rng=np.random.default_rng(0))
    b = round_robin(_agents(), pairs=8, rng=np.random.default_rng(0))
    assert np.array_equal(a.win_rate, b.win_rate)
    assert a.labels == b.labels
