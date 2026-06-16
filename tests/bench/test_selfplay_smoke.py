"""Self-play determinism and the multiprocessing aggregation path."""

import numpy as np
from bgrl.bench.selfplay import play_random_game, run_selfplay


def test_play_random_game_terminates_and_is_deterministic():
    a = play_random_game(np.random.default_rng(0))
    b = play_random_game(np.random.default_rng(0))
    assert a == b  # same seed -> identical game
    calls, afterstates, plies = a
    assert calls > 0
    assert afterstates > 0
    assert 0 < plies < 20000  # a real game, not the safety cap


def test_run_selfplay_aggregates():
    out = run_selfplay(n_games=4, workers=2, seed=1)
    assert out["games"] == 4
    assert out["workers"] == 2
    assert out["calls"] > 0
    assert out["elapsed"] > 0
