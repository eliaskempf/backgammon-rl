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

import numpy as np

from bgrl.env.types import Player


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
