"""Reproducible dice and common-random-numbers (CRN) replay.

All randomness in a game flows through a :class:`DiceSource`, never the global
RNG, so a game is fully determined by ``(agents, dice source)``. **CRN** = record
the dice stream of one run and replay the identical stream to a different
agent/config, which removes dice variance from agent-vs-agent comparisons (WP1
eval) and LLM prompt A/B tests (WP4): the only difference between the two runs is
the thing under test.
"""

from __future__ import annotations

from collections import deque
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


def weighted_rolls() -> tuple[tuple[Dice, float], ...]:
    """The 21 distinct dice rolls with probabilities: doubles 1/36, non-doubles 2/36.

    Each non-double is listed once as ``(a, b)`` with ``a <= b``; move generation
    explores both die orderings internally, so listing ``(b, a)`` too would double-count.
    The weights sum to exactly 1. This is the chance-node distribution shared by
    expectimax search and the pre-roll cube evaluator.
    """
    rolls: list[tuple[Dice, float]] = []
    for a in range(1, 7):
        for b in range(a, 7):
            rolls.append(((a, b), (1.0 if a == b else 2.0) / 36.0))
    return tuple(rolls)


WEIGHTED_ROLLS = weighted_rolls()
"""The 21 distinct rolls with their probabilities (see :func:`weighted_rolls`)."""


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


class ManualDiceSource:
    """A queue of human-supplied rolls, consumed one per ply.

    Used by the play server's manual-dice mode: the human enters the dice for *both*
    seats so the agent never draws from an RNG, which rebuts any "the bot rigs its
    rolls" objection. :meth:`push` enqueues a validated roll; :meth:`roll` pops the
    oldest pending roll and raises ``RuntimeError`` if none is queued, so a missing
    roll fails loudly rather than silently fabricating one.
    """

    def __init__(self) -> None:
        self._pending: deque[Dice] = deque()

    def push(self, dice: Dice) -> None:
        """Enqueue a roll. Each die must be in ``1..6`` (doubles are allowed)."""
        d0, d1 = int(dice[0]), int(dice[1])
        if not (1 <= d0 <= 6 and 1 <= d1 <= 6):
            raise ValueError(f"dice must each be in 1..6, got {dice!r}")
        self._pending.append((d0, d1))

    def roll(self) -> Dice:
        if not self._pending:
            raise RuntimeError("ManualDiceSource is empty: supply the dice for this roll first")
        return self._pending.popleft()


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
