"""The value-network contract.

A :class:`ValueNet` maps a batch of encoded afterstates (see
:func:`bgrl.env.encoding.encode`) to a fixed-length outcome vector per position.
Every value-based agent (TD, expectimax, MCTS-with-net) consumes this same
interface, so nets stay swappable behind the agent layer.

The output layout is **fixed at length** :data:`OUTCOME_DIM` and is the cube-ready
5-vector from the *mover's* point of view::

    [p_win, p_win_gammon, p_win_bg, p_lose_gammon, p_lose_bg]

interpreted **cumulatively**, gnubg-style (``p_win`` = P(win at all);
``p_win_gammon`` = P(win a gammon *or better*); ``p_win_bg`` = P(win a backgammon);
likewise for the loss heads). ``p_lose = 1 - p_win`` is implied. v1 trains only
``p_win`` and leaves the other heads near zero, but the shape never changes — see
:func:`bgrl.nets.equity.equity` for the reduction to a scalar.

A policy head (WP5) would be a *second* output and must not break this contract;
it is intentionally absent here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

OUTCOME_DIM = 5
"""Length of the outcome vector. Fixed by the contract, never v1-specific."""


@runtime_checkable
class ValueNet(Protocol):
    """Anything that scores encoded afterstates into outcome vectors."""

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """Map float32 features ``(..., N_FEATURES)`` to ``(..., OUTCOME_DIM)``.

        Batched over the leading axes. The input is ``encode(afterstate, pov)``
        in the mover's point of view; the output is that mover's outcome
        distribution. Implementations must be side-effect free (no training).
        """
        ...
