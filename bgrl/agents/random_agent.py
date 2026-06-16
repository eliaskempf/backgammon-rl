"""Uniform-random reference agent.

Doubles as a test/dev opponent and as the dummy agent WP3 builds against before a
trained checkpoint exists. Dice and choices come from an injected
``numpy.random.Generator`` so self-play is reproducible (no global RNG).
"""

from __future__ import annotations

import numpy as np

from bgrl.env import Dice, EnvState, Move


class RandomAgent:
    """Picks uniformly at random among the legal moves."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move:
        return legal[int(self._rng.integers(len(legal)))][0]
