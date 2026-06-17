"""Live round-trip + analysis tests that drive a real gnubg binary.

Skipped entirely when gnubg is not installed, so the suite stays green everywhere; on a
host with gnubg (``apt-get install gnubg``) these confirm our ``.mat`` imports cleanly and
that the analysis pipeline returns sane per-move equity losses. Marked ``slow`` (each
spawns gnubg and analyses a whole game).
"""

import numpy as np
import pytest

from bgrl.agents import RandomAgent
from bgrl.env import RandomDiceSource
from bgrl.game import play_game
from bgrl.serialization import analyse_mat, game_to_mat, gnubg_available, summarize

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not gnubg_available(), reason="gnubg not installed"),
]


def _game(seed):
    dice = RandomDiceSource(np.random.default_rng(seed))
    white = RandomAgent(np.random.default_rng(1000 + seed))
    black = RandomAgent(np.random.default_rng(2000 + seed))
    return play_game(white, black, dice, record=True)


def test_mat_round_trips_through_gnubg(tmp_path):
    res = _game(0)  # opens (6, 4): not doubles, so gnubg accepts the opening
    mat = game_to_mat(res.steps, res.outcome, white_name="white", black_name="black")
    path = tmp_path / "game.mat"
    path.write_text(mat)

    moves = analyse_mat(path, plies=0)  # 0-ply (raw net) keeps it fast
    assert moves, "gnubg parsed no chequer plays from our .mat"
    # A played move can never beat gnubg's best, so equity loss is >= 0 up to fp noise.
    assert all(m.equity_loss >= -1e-6 for m in moves)
    assert {m.player for m in moves} <= {"White", "Black"}


def test_summary_reports_per_side_stats(tmp_path):
    res = _game(0)
    path = tmp_path / "game.mat"
    path.write_text(game_to_mat(res.steps, res.outcome))

    summary = summarize(analyse_mat(path, plies=0))
    assert summary["overall"].moves > 0
    assert 0.0 <= summary["overall"].agreement <= 1.0
    assert summary["overall"].mean_equity_loss >= -1e-6
