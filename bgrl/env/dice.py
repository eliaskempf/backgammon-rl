"""Reproducible dice and common-random-numbers (CRN) replay.

All randomness in a game flows through a :class:`DiceSource`, never the global
RNG, so a game is fully determined by ``(agents, dice source)``. **CRN** = record
the dice stream of one run and replay the identical stream to a different
agent/config, which removes dice variance from agent-vs-agent comparisons (WP1
eval) and LLM prompt A/B tests (WP4): the only difference between the two runs is
the thing under test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .types import Dice


def roll_dice(rng: np.random.Generator) -> Dice:
    """Roll two independent dice in ``1..6``.

    Doubles (equal dice) are meaningful to move generation, which expands them to
    four moves; this function does not special-case them.
    """
    return (int(rng.integers(1, 7)), int(rng.integers(1, 7)))


@runtime_checkable
class DiceSource(Protocol):
    """A stream of dice rolls a game pulls from, one per ply."""

    def roll(self) -> Dice: ...


class RandomDiceSource:
    """Rolls from an injected ``Generator`` and records every roll.

    The recorded :attr:`history` can be replayed via :class:`ReplayDiceSource` to
    reproduce the exact same dice in another run (CRN).
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.history: list[Dice] = []

    def roll(self) -> Dice:
        d = roll_dice(self._rng)
        self.history.append(d)
        return d


class ReplayDiceSource:
    """Replays a previously recorded roll sequence, in order.

    Raises ``RuntimeError`` when the recorded sequence is exhausted, so a replay
    that is too short for the new run fails loudly instead of silently diverging.
    """

    def __init__(self, history: Sequence[Dice]) -> None:
        self._history: list[Dice] = list(history)
        self._index = 0

    def roll(self) -> Dice:
        if self._index >= len(self._history):
            raise RuntimeError("ReplayDiceSource exhausted: recorded dice sequence is too short")
        d = self._history[self._index]
        self._index += 1
        return d
