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
from bgrl.money import play_money_game
from bgrl.nets.cube import CubeAction, CubeDecider
from bgrl.nets.equity import CENTERED_CUBE, CubeAccess, cubeful_equity
from bgrl.serialization import (
    analyse_cube,
    analyse_mat,
    game_to_mat,
    gnubg_available,
    money_game_to_mat,
    summarize,
)

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


class _CubeBot:
    """A CubeCapable bot (greedy first-legal play) for generating a cube money game."""

    def __init__(self, *, double_from_centered, take):
        self._double = double_from_centered
        self._take = take

    def act(self, state, dice, legal):
        return legal[0][0]

    def should_double(self, state, cube):
        return self._double and cube.value == 1 and cube.owner is None

    def should_take(self, state, cube):
        return self._take


def test_money_cube_mat_round_trips_through_gnubg(tmp_path):
    # BLACK doubles on its first turn (ply 1, after WHITE's opening move so it is not an
    # illegal pre-opening double), WHITE takes; the game then plays out at cube value 2.
    white = _CubeBot(double_from_centered=False, take=True)
    black = _CubeBot(double_from_centered=True, take=True)
    res = play_money_game(white, black, RandomDiceSource(np.random.default_rng(0)))
    assert res.cube_events and res.cube_events[0].taken  # the double was offered and taken

    mat = money_game_to_mat(res, white_name="white", black_name="black")
    path = tmp_path / "money.mat"
    path.write_text(mat)

    # gnubg must accept the Doubles/Takes tokens and still parse every chequer play.
    moves = analyse_mat(path, plies=0)
    assert moves, "gnubg parsed no chequer plays from the cube .mat"
    assert all(m.equity_loss >= -1e-6 for m in moves)


def _cube_money_game(seed=0):
    white = _CubeBot(double_from_centered=False, take=True)
    black = _CubeBot(double_from_centered=True, take=True)  # BLACK doubles from centered
    return play_money_game(white, black, RandomDiceSource(np.random.default_rng(seed)))


def test_cubeful_equity_matches_gnubg_on_gnubg_probs(tmp_path):
    # Feed gnubg's OWN cubeless distribution into our Janowski formula: this isolates the
    # cube math from net quality, so it must reproduce gnubg's no-double cubeful equity at
    # the centered cube. The default cube-life x=2/3 is what makes them agree.
    res = _cube_money_game()
    path = tmp_path / "money.mat"
    path.write_text(money_game_to_mat(res, white_name="white", black_name="black"))

    cubes = analyse_cube(path, plies=0)
    assert cubes, "gnubg extracted no cube decisions"
    doubles = [c for c in cubes if c.action_kind == "double"]
    assert doubles, "expected the centered double to be analysed"
    c = doubles[0]
    assert len(c.probs) == 5

    ours = float(cubeful_equity(np.array(c.probs), access=CubeAccess.CENTERED))
    assert abs(ours - c.nd_equity) < 0.05, (ours, c.nd_equity)

    # Our decider must return a valid action on gnubg's distribution.
    action = CubeDecider().decide_double(
        np.array(c.probs), CENTERED_CUBE, access=CubeAccess.CENTERED
    )
    assert action in set(CubeAction)
