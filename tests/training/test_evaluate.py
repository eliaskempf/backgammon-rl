"""CRN win-rate evaluation: balance, reproducibility, truncation handling (Phase A)."""

import numpy as np

from bgrl.agents import RandomAgent
from bgrl.training.evaluate import play_match


def test_random_vs_random_is_balanced():
    res = play_match(
        RandomAgent(np.random.default_rng(1)),
        RandomAgent(np.random.default_rng(2)),
        pairs=80,
        rng=np.random.default_rng(0),
    )
    assert res.games == 160
    assert res.wins_a + res.wins_b + res.truncated == res.games
    assert 0.35 < res.win_rate_a < 0.65  # ~0.5; generous band for 160 games


def test_crn_result_is_reproducible():
    def run():
        return play_match(
            RandomAgent(np.random.default_rng(1)),
            RandomAgent(np.random.default_rng(2)),
            pairs=10,
            rng=np.random.default_rng(7),
        )

    assert run() == run()  # frozen dataclass, field-wise equality


def test_truncated_games_excluded_from_win_rate():
    # max_plies=1 ends every game before a winner exists -> all truncated.
    res = play_match(
        RandomAgent(np.random.default_rng(1)),
        RandomAgent(np.random.default_rng(2)),
        pairs=3,
        rng=np.random.default_rng(0),
        max_plies=1,
    )
    assert res.truncated == res.games == 6
    assert res.wins_a == res.wins_b == 0
    assert res.win_rate_a == 0.0  # guarded division, not a ZeroDivisionError
