"""Doubling-cube decision policy: a pure, agent-independent module (WP6 B3).

The double / take / pass decision is a pure function of a position's cubeless outcome
distribution and the cube context — it belongs to no checker-play agent. The money-game
loop and every agent share one :class:`CubeDecider`, which reduces the decision to
comparisons of Janowski cubeful equities (:func:`~bgrl.nets.equity.cubeful_equity`):
double when doubling-and-being-taken beats holding, cash (the opponent must drop) when
that is worth more, and play on ("too good") when holding for the gammon beats cashing.

The decider is the piece validated against gnubg (WP6 B4): the cube-life coefficient
``x`` is the only knob, tuned so the actions match gnubg on a reference position set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from bgrl.nets.equity import DEFAULT_CUBE_LIFE, CubeAccess, CubeContext, cubeful_equity


class CubeAction(Enum):
    """The on-roll player's cube action."""

    NO_DOUBLE = "no_double"
    DOUBLE = "double"
    TOO_GOOD = "too_good"  # do not double: holding for the gammon beats cashing


class TakeAction(Enum):
    """The opponent's response to an offered double."""

    TAKE = "take"
    PASS = "pass"


@dataclass(frozen=True, slots=True)
class CubeDecider:
    """Money-play cube policy parameterised by the Janowski cube-life coefficient ``x``."""

    x: float = DEFAULT_CUBE_LIFE

    def decide_double(
        self,
        outcome: np.ndarray,
        cube: CubeContext,
        access: CubeAccess = CubeAccess.CENTERED,
    ) -> CubeAction:
        """The on-roll player's action given its cubeless 5-vector ``outcome``.

        ``access`` is the mover's current cube access — ``CENTERED`` (initial double) or
        ``I_OWN`` (a redouble). ``OPP_OWNS`` is illegal (the mover may not double) and
        raises; the caller checks ownership first. Compares three cubeful equities, all
        mover-POV absolute points: hold (no double), double-and-taken (opponent then owns
        the doubled cube), and cash (opponent drops, mover collects the current stake).
        """
        if access is CubeAccess.OPP_OWNS:
            raise ValueError("cannot double when the opponent owns the cube")
        hold = float(cubeful_equity(outcome, cube, access=access, x=self.x))
        doubled = replace(cube, value=cube.value * 2)
        taken = float(cubeful_equity(outcome, doubled, access=CubeAccess.OPP_OWNS, x=self.x))
        cash = float(cube.value)  # opponent passes -> mover collects the current stake
        # The opponent responds with whichever is worse for the mover.
        if taken < cash:  # opponent would take
            return CubeAction.DOUBLE if taken >= hold else CubeAction.NO_DOUBLE
        # opponent would pass: cashing is on the table, but holding may be worth more.
        return CubeAction.TOO_GOOD if hold > cash else CubeAction.DOUBLE

    def decide_take(self, responder_outcome: np.ndarray, cube: CubeContext) -> TakeAction:
        """The opponent's take / pass when offered a double from the pre-double ``cube``.

        ``responder_outcome`` is the **responder's** cubeless 5-vector. Taking doubles the
        stake to ``2*cube.value`` with the responder owning the cube; passing forfeits the
        current ``cube.value``. Take iff the cubeful take equity is no worse than dropping.
        """
        doubled = replace(cube, value=cube.value * 2)
        take = float(cubeful_equity(responder_outcome, doubled, access=CubeAccess.I_OWN, x=self.x))
        drop = -float(cube.value)
        return TakeAction.TAKE if take >= drop else TakeAction.PASS
