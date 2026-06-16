"""Equity reduction: the cubeless gnubg-cumulative formula, anti-symmetry, shapes."""

import numpy as np

from bgrl.nets import CENTERED_CUBE, OUTCOME_DIM, equity


def _vec(p_win=0.0, p_wg=0.0, p_wbg=0.0, p_lg=0.0, p_lbg=0.0):
    return np.array([p_win, p_wg, p_wbg, p_lg, p_lbg], dtype=np.float64)


def test_certain_single_win_and_loss():
    assert float(equity(_vec(p_win=1.0))) == 1.0
    assert float(equity(_vec(p_win=0.0))) == -1.0


def test_even_position_is_zero():
    assert float(equity(_vec(p_win=0.5))) == 0.0


def test_v1_reduces_to_2p_minus_1():
    # With gammon/backgammon heads at zero, equity collapses to 2*p_win - 1.
    for p in (0.1, 0.37, 0.8):
        assert float(equity(_vec(p_win=p))) == 2 * p - 1


def test_gammon_backgammon_point_weighting():
    assert float(equity(_vec(p_win=1.0, p_wg=1.0))) == 2.0  # certain gammon win
    assert float(equity(_vec(p_win=1.0, p_wg=1.0, p_wbg=1.0))) == 3.0  # certain bg win
    assert float(equity(_vec(p_lg=1.0))) == -2.0  # certain gammon loss (p_win=0)
    assert float(equity(_vec(p_lg=1.0, p_lbg=1.0))) == -3.0  # certain bg loss


def test_anti_symmetry():
    # Swapping the win/loss heads (and p_win -> 1 - p_win) must negate the equity.
    rng = np.random.default_rng(0)
    o = rng.random((50, OUTCOME_DIM))
    flipped = o[:, [0, 3, 4, 1, 2]].copy()
    flipped[:, 0] = 1.0 - o[:, 0]
    assert np.allclose(equity(o) + equity(flipped), 0.0, atol=1e-12)


def test_vectorised_shapes():
    assert equity(_vec(p_win=0.5)).shape == ()  # single vector -> 0-d
    assert equity(np.zeros((3, 4, OUTCOME_DIM))).shape == (3, 4)


def test_cube_arg_accepted_and_ignored_in_v1():
    o = _vec(p_win=0.7)
    assert float(equity(o)) == float(equity(o, CENTERED_CUBE))
