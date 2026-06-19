"""Janowski cubeful money equity and the win/loss magnitude decomposition (WP6 B2)."""

from __future__ import annotations

import numpy as np
import pytest

from bgrl.env import Player
from bgrl.nets.cube import CubeAction, CubeDecider, TakeAction
from bgrl.nets.equity import (
    DEFAULT_CUBE_LIFE,
    CubeAccess,
    CubeContext,
    cube_access,
    cubeful_equity,
    equity,
    win_loss_magnitudes,
)


def _vec(p_win=0.0, p_wg=0.0, p_wbg=0.0, p_lg=0.0, p_lbg=0.0):
    return np.array([p_win, p_wg, p_wbg, p_lg, p_lbg], dtype=np.float64)


def _valid_vecs(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` *valid* cumulative outcome vectors (monotone, buckets sum to 1).

    Built from a Dirichlet over the six [win/lose]x[single/gammon/bg] buckets, so the
    cumulative heads are genuinely monotone and ``win_loss_magnitudes`` reconstructs them
    exactly (no clamping) — the regime in which the dead cube equals cubeless equity.
    """
    # lose-single mass is implied by p_lose = 1 - p_win, so it is not stored in the vector.
    w_s, w_g, w_bg, _l_s, l_g, l_bg = rng.dirichlet(np.ones(6), size=n).T
    out = np.empty((n, 5))
    out[:, 0] = w_s + w_g + w_bg  # p_win
    out[:, 1] = w_g + w_bg  # p_win_g
    out[:, 2] = w_bg  # p_win_bg
    out[:, 3] = l_g + l_bg  # p_lose_g
    out[:, 4] = l_bg  # p_lose_bg
    return out


# --- win_loss_magnitudes ---------------------------------------------------------------


def test_win_loss_magnitudes_basic() -> None:
    # win: single 0.4, gammon 0.2 -> 0.8 points over p=0.6 -> W=4/3; loss: single 0.3,
    # gammon 0.1 -> 0.5 points over p=0.4 -> L=1.25.
    p, w, ell = win_loss_magnitudes(_vec(p_win=0.6, p_wg=0.2, p_lg=0.1))
    assert float(p) == pytest.approx(0.6)
    assert float(w) == pytest.approx(0.8 / 0.6)
    assert float(ell) == pytest.approx(0.5 / 0.4)


def test_win_loss_magnitudes_clamps_nonmonotone_heads() -> None:
    # p_win_g > p_win (sigmoids are not monotone): buckets clamp to >= 0, W stays finite.
    _, w, ell = win_loss_magnitudes(_vec(p_win=0.3, p_wg=0.5, p_wbg=0.4))
    assert float(w) >= 1.0
    assert np.isfinite(float(w)) and np.isfinite(float(ell))


def test_win_loss_magnitudes_certainty_fallback() -> None:
    # p_win -> 1: the loss average is undefined; fall back to a pure single (L = 1).
    _, w, ell = win_loss_magnitudes(_vec(p_win=1.0, p_wg=1.0))
    assert float(w) == pytest.approx(2.0)  # certain gammon win
    assert float(ell) == pytest.approx(1.0)


# --- cubeful_equity --------------------------------------------------------------------


@pytest.mark.parametrize("access", list(CubeAccess))
def test_dead_cube_equals_cubeless(access: CubeAccess) -> None:
    # At x=0 every ownership branch collapses to the cubeless money equity (on valid,
    # monotone distributions, where the magnitude decomposition is exact).
    o = _valid_vecs(np.random.default_rng(0), 40)
    assert np.allclose(cubeful_equity(o, access=access, x=0.0), equity(o), atol=1e-12)


def test_cube_value_scales_equity() -> None:
    o = _vec(p_win=0.7, p_wg=0.2)
    single = cubeful_equity(o, CubeContext(value=1), access=CubeAccess.I_OWN)
    doubled = cubeful_equity(o, CubeContext(value=2), access=CubeAccess.I_OWN)
    assert float(doubled) == pytest.approx(2.0 * float(single))


@pytest.mark.parametrize("x", [0.0, 1.0 / 3.0, DEFAULT_CUBE_LIFE, 1.0])
def test_even_position_ownership_symmetry(x: float) -> None:
    # An even single position: owning the cube is worth +0.25x, the opponent owning it
    # is worth -0.25x, and a centered cube is exactly 0 — the cube-ownership value.
    o = _vec(p_win=0.5)
    assert float(cubeful_equity(o, access=CubeAccess.I_OWN, x=x)) == pytest.approx(0.25 * x)
    assert float(cubeful_equity(o, access=CubeAccess.OPP_OWNS, x=x)) == pytest.approx(-0.25 * x)
    assert float(cubeful_equity(o, access=CubeAccess.CENTERED, x=x)) == pytest.approx(0.0)


def test_owning_cube_beats_opponent_owning() -> None:
    # For any live cube, holding it is worth exactly 0.5x more than the opponent holding it.
    o = _vec(p_win=0.62, p_wg=0.15)
    mine = float(cubeful_equity(o, access=CubeAccess.I_OWN, x=DEFAULT_CUBE_LIFE))
    theirs = float(cubeful_equity(o, access=CubeAccess.OPP_OWNS, x=DEFAULT_CUBE_LIFE))
    assert mine - theirs == pytest.approx(0.5 * DEFAULT_CUBE_LIFE)


# --- cube_access -----------------------------------------------------------------------


def test_cube_access_from_absolute_owner() -> None:
    white_owned = CubeContext(value=2, owner=Player.WHITE)
    black_owned = CubeContext(value=2, owner=Player.BLACK)
    assert cube_access(Player.WHITE, CubeContext()) is CubeAccess.CENTERED
    assert cube_access(Player.WHITE, white_owned) is CubeAccess.I_OWN
    assert cube_access(Player.WHITE, black_owned) is CubeAccess.OPP_OWNS


# --- CubeDecider -----------------------------------------------------------------------

_DECIDER = CubeDecider()


def test_decide_take_take_point() -> None:
    # Even single (p=0.5) is a clear take; ~10% is a clear pass; ~25% is the dead-cube
    # boundary (and a take with the live owned cube).
    assert _DECIDER.decide_take(_vec(p_win=0.5), CubeContext()) is TakeAction.TAKE
    assert _DECIDER.decide_take(_vec(p_win=0.1), CubeContext()) is TakeAction.PASS
    assert _DECIDER.decide_take(_vec(p_win=0.25), CubeContext()) is TakeAction.TAKE


def test_decide_double_even_position_is_no_double() -> None:
    assert _DECIDER.decide_double(_vec(p_win=0.5), CubeContext()) is CubeAction.NO_DOUBLE


def test_decide_double_in_window_doubles() -> None:
    # A strong-but-takeable single (p=0.75): the opponent takes and doubling beats holding.
    assert _DECIDER.decide_double(_vec(p_win=0.75), CubeContext()) is CubeAction.DOUBLE


def test_decide_double_cash_without_gammons() -> None:
    # Very likely single win, no gammon threat: the opponent must pass, so it is a cash
    # (DOUBLE) rather than "too good".
    assert _DECIDER.decide_double(_vec(p_win=0.85), CubeContext()) is CubeAction.DOUBLE


def test_decide_double_too_good_with_heavy_gammon() -> None:
    # High win prob with a large gammon share: holding for the gammon beats cashing.
    heavy = _vec(p_win=0.85, p_wg=0.55)
    assert _DECIDER.decide_double(heavy, CubeContext()) is CubeAction.TOO_GOOD


def test_decide_double_rejects_opponent_owned_cube() -> None:
    with pytest.raises(ValueError, match="opponent owns"):
        _DECIDER.decide_double(_vec(p_win=0.7), CubeContext(value=2), access=CubeAccess.OPP_OWNS)
