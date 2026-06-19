"""Equity reduction: outcome vector -> scalar, the only thing move selection ranks.

Agents never compare raw net outputs; they compare **equity**. Keeping the
reduction in one place is what lets the doubling cube, gammon/backgammon scoring,
and match equity slot in later without touching any agent: only this module learns
about :class:`CubeContext`.

v1 is cubeless single games, so :func:`equity` implements the standard **cubeless
money equity** over the gnubg-cumulative 5-vector (see :mod:`bgrl.nets.base`):

    win pays +1 / +2 / +3 for single / gammon / backgammon, loss the negatives.

With cumulative heads this collapses to::

    equity = (p_win + p_win_g + p_win_bg) - (p_lose + p_lose_g + p_lose_bg)

where ``p_lose = 1 - p_win``. When the gammon/backgammon heads are zero (v1's
``p_win``-only net) this is exactly ``2 * p_win - 1``. The function is
**anti-symmetric**: swapping the win and loss heads negates the result, which is
what makes afterstate selection a single ``argmax`` (see
:class:`bgrl.agents.value_agent.ValueAgent`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from bgrl.env.types import Outcome, Player

_EPS = 1e-9  # guard for the p -> {0, 1} cube degeneracies in win_loss_magnitudes


@dataclass(frozen=True, slots=True)
class CubeContext:
    """The doubling-cube state equity reduction needs.

    v1 is always centered at value 1, but the argument exists from day one so the
    cube can be honoured later without changing the :func:`equity` signature.
    """

    value: int = 1
    owner: Player | None = None  # None = centered (nobody owns the cube)


CENTERED_CUBE = CubeContext()
"""The trivial v1 cube: value 1, centered."""


class CubeAccess(IntEnum):
    """Cube ownership **relative to the mover** (the player on roll).

    :class:`CubeContext.owner` is an *absolute* :class:`~bgrl.env.types.Player`; the
    cubeful formulas need only the mover-relative relationship, so the caller collapses
    ``(mover, owner)`` to this via :func:`cube_access` and the equity module never
    re-derives perspective.
    """

    CENTERED = 0  # nobody owns the cube (owner is None) — either side may double
    I_OWN = 1  # the mover owns or shares the cube — the mover may (re)double
    OPP_OWNS = 2  # the opponent owns the cube — the mover may not double


DEFAULT_CUBE_LIFE = 2.0 / 3.0
"""Janowski cube-life coefficient ``x`` (0 = dead cube, 1 = fully live). ~2/3 is the
usual money default; the gnubg cube cross-check (WP6 B4) is what tunes it."""


def cube_access(mover: Player, cube: CubeContext) -> CubeAccess:
    """Collapse absolute cube ownership to the mover-relative :class:`CubeAccess`."""
    if cube.owner is None:
        return CubeAccess.CENTERED
    return CubeAccess.I_OWN if cube.owner is mover else CubeAccess.OPP_OWNS


def equity(outcome: np.ndarray, cube: CubeContext = CENTERED_CUBE) -> np.ndarray:
    """Reduce outcome vector(s) ``(..., OUTCOME_DIM)`` to scalar equity ``(...)``.

    Vectorised over the leading axes; a single ``(5,)`` vector yields a 0-d array
    (call ``float(...)`` if a Python float is needed). ``cube`` is accepted for
    forward compatibility and ignored in cubeless v1 (it is always centered).
    """
    arr = np.asarray(outcome, dtype=np.float64)
    p_win = arr[..., 0]
    p_win_g = arr[..., 1]
    p_win_bg = arr[..., 2]
    p_lose_g = arr[..., 3]
    p_lose_bg = arr[..., 4]
    p_lose = 1.0 - p_win
    return (p_win + p_win_g + p_win_bg) - (p_lose + p_lose_g + p_lose_bg)


# The perspective flip on the cumulative 5-vector, expressed as a permutation with a
# sign/bias so one definition drives everything that needs the mover<->opponent map:
# the TD(λ) bootstrap target *and* its per-head eligibility-trace carry (WP6 Part A),
# and the cube evaluator's "opponent afterstate value -> mover POV" step (Part B).
# Read off the semantics of the heads: flipping POV turns *my* win into the
# opponent's loss, so head 0 (p_win) complements, the win/lose *gammon* heads {1,3}
# swap, and the win/lose *backgammon* heads {2,4} swap. Writing the result of head k
# as ``bias[k] + sign[k] * src[perm[k]]`` makes head 0 the lone complement
# (1 - src[0]) and the four swap heads pure relabels (+1 * src[paired]).
FLIP_PERM: tuple[int, ...] = (0, 3, 4, 1, 2)
"""``perm[k]``: head k of the flipped vector reads source head ``perm[k]``.

Also the eligibility-trace pairing in :class:`bgrl.training.td_lambda.TDLambda`:
head k's trace carries from its *paired* head ``FLIP_PERM[k]`` of the previous ply.
"""

FLIP_SIGN: np.ndarray = np.array([-1.0, 1.0, 1.0, 1.0, 1.0])
"""Linear coefficient per head of the flip's Jacobian: head 0 complements (-1), the
swapped pairs are pure permutations (+1). Signs the trace carry's per-head factor."""

_FLIP_BIAS: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
"""Constant term per head: the complement's ``1`` on head 0, zero elsewhere."""


def flip_outcome(outcome: np.ndarray) -> np.ndarray:
    """Map outcome vector(s) to the **opponent's** point of view (the perspective flip).

    For a cumulative mover-POV vector ``s``::

        flip_outcome(s) = [1 - s[0], s[3], s[4], s[1], s[2]]

    i.e. win ⟺ opponent loses (head 0 complements) and the win/lose gammon and
    backgammon heads swap. It is an **involution** (``flip_outcome(flip_outcome(s))
    == s``) and **anti-symmetric through equity** (``equity(flip_outcome(v)) ==
    -equity(v)``) — the same anti-symmetry afterstate selection already relies on.
    Vectorised over the leading axes via :data:`FLIP_PERM` / :data:`FLIP_SIGN`.
    """
    arr = np.asarray(outcome, dtype=np.float64)
    return _FLIP_BIAS + FLIP_SIGN * arr[..., FLIP_PERM]


def outcome_to_vector(result: Outcome, perspective: Player) -> np.ndarray:
    """The realised cumulative outcome 5-vector for a finished game, ``perspective`` POV.

    From the winner's POV: ``[1, kind>=GAMMON, kind>=BACKGAMMON, 0, 0]``. From the
    loser's POV the win heads are 0 and the loss-magnitude heads fire. A single *loss* is
    the all-zeros vector — its -1 equity comes from the implied ``p_lose = 1 - p_win``
    inside :func:`equity`, not from any explicit head. Shared by the exact terminal
    scoring in expectimax and by the cube evaluator's leaf vectors.
    """
    vec = np.zeros(5, dtype=np.float64)
    kind = int(result.kind)
    if result.winner is perspective:
        vec[0] = 1.0  # p_win
        if kind >= 2:
            vec[1] = 1.0  # p_win_gammon (cumulative: gammon or better)
        if kind >= 3:
            vec[2] = 1.0  # p_win_backgammon
    else:
        if kind >= 2:
            vec[3] = 1.0  # p_lose_gammon
        if kind >= 3:
            vec[4] = 1.0  # p_lose_backgammon
    return vec


def win_loss_magnitudes(
    outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose outcome vector(s) into ``(p_win, W, L)`` for the cubeful formulas.

    ``W`` and ``L`` are the **positive average magnitudes conditional on winning /
    losing** (1 = single, 2 = gammon, 3 = backgammon), recovered from the cumulative
    heads. The independent sigmoid heads are not guaranteed monotone
    (``p_win >= p_win_g >= p_win_bg``), so the per-bucket masses are clamped to be
    non-negative; and where ``p_win`` is ~0 or ~1 the conditional average is undefined
    (and the cube is irrelevant at certainty), so ``W`` / ``L`` fall back to 1 (a pure
    single). Vectorised over the leading axes like :func:`equity`.
    """
    arr = np.asarray(outcome, dtype=np.float64)
    p_win = np.clip(arr[..., 0], 0.0, 1.0)
    p_lose = 1.0 - p_win
    # Cumulative heads -> non-negative single/gammon/backgammon bucket masses.
    w_g = np.clip(arr[..., 1] - arr[..., 2], 0.0, None)
    w_bg = np.clip(arr[..., 2], 0.0, None)
    w_s = np.clip(p_win - arr[..., 1], 0.0, None)
    l_g = np.clip(arr[..., 3] - arr[..., 4], 0.0, None)
    l_bg = np.clip(arr[..., 4], 0.0, None)
    l_s = np.clip(p_lose - arr[..., 3], 0.0, None)
    win_points = w_s + 2.0 * w_g + 3.0 * w_bg
    lose_points = l_s + 2.0 * l_g + 3.0 * l_bg
    # Conditional averages, guarding the p -> {0, 1} degeneracies (cube irrelevant there).
    big_win = p_win > _EPS
    big_lose = p_lose > _EPS
    win_mag = np.where(big_win, win_points / np.where(big_win, p_win, 1.0), 1.0)
    lose_mag = np.where(big_lose, lose_points / np.where(big_lose, p_lose, 1.0), 1.0)
    return p_win, win_mag, lose_mag


def cubeful_equity(
    outcome: np.ndarray,
    cube: CubeContext = CENTERED_CUBE,
    *,
    access: CubeAccess = CubeAccess.CENTERED,
    x: float = DEFAULT_CUBE_LIFE,
) -> np.ndarray:
    """Janowski cubeful money equity, mover POV, in absolute points (scaled by cube value).

    Given the cubeless outcome distribution, the cube-life coefficient ``x``, and the
    mover-relative ownership ``access``, returns the standard Janowski dead/live-cube
    interpolation (Janowski, *Take-Points in Money Games*; gnubg's "basic formula for
    cubeful equities")::

        base = p·(W + L + 0.5x)
        I_OWN     :  base - L
        OPP_OWNS  :  base - L - 0.5x
        CENTERED  :  (4/(4-x))·(base - L - 0.25x)

    where ``p, W, L`` come from :func:`win_loss_magnitudes`. At ``x = 0`` every branch
    collapses to the dead cube ``p(W+L) - L`` -- exactly ``cube.value`` times the
    cubeless :func:`equity` (for a valid, monotone outcome vector) -- so the cubeless
    reduction remains the take/drop reference. Vectorised over the leading axes.
    """
    p, win_mag, lose_mag = win_loss_magnitudes(outcome)
    base = p * (win_mag + lose_mag + 0.5 * x)
    if access is CubeAccess.I_OWN:
        e = base - lose_mag
    elif access is CubeAccess.OPP_OWNS:
        e = base - lose_mag - 0.5 * x
    else:  # CENTERED
        e = (4.0 / (4.0 - x)) * (base - lose_mag - 0.25 * x)
    return float(cube.value) * e
