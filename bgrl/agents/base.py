"""The minimal agent interface.

WP0 freezes only ``act`` (enough to drive self-play and the benchmark). The
optional learning hooks (``observe_step`` / ``observe_game_end``) are added in the
contract-freeze follow-up so the value-based trainers can plug in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bgrl.env import Dice, EnvState, Move


@runtime_checkable
class Agent(Protocol):
    """Chooses one move from the precomputed legal ``(Move, afterstate)`` list.

    Move generation lives in the env, not the agent, so every algorithm consumes
    the same afterstate enumeration. ``legal`` is guaranteed non-empty (the
    caller handles a pass when there are no legal moves).
    """

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move: ...
