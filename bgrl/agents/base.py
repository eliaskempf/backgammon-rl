"""The agent interface.

:class:`Agent` is the minimal contract — choose a move from the precomputed legal
``(Move, afterstate)`` list — and is all a non-learning agent (random, LLM,
expectimax) needs. :class:`LearningAgent` adds the self-play lifecycle hooks the
online trainers (WP1's TD(λ)) use to learn from the games they generate.

Move generation lives in the env, not the agent, so every algorithm consumes the
same afterstate enumeration; only selection (and, for learners, the update rule)
differ.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bgrl.env import Dice, EnvState, Move, Outcome


@runtime_checkable
class Agent(Protocol):
    """Chooses one move from the precomputed legal ``(Move, afterstate)`` list.

    ``legal`` is guaranteed non-empty (the caller handles a pass when there are
    no legal moves).
    """

    def act(self, state: EnvState, dice: Dice, legal: list[tuple[Move, EnvState]]) -> Move: ...


@runtime_checkable
class LearningAgent(Agent, Protocol):
    """An :class:`Agent` that also learns from the self-play games it plays.

    The game driver (:func:`bgrl.game.play_game`) drives the lifecycle:

    * ``observe_step`` fires once for **every ply the agent makes**, in order,
      right after its chosen move is applied. In TD-Gammon-style self-play one
      learning agent plays *both* seats, so it observes the whole afterstate
      trajectory. Consecutive afterstates belong to opposite movers, so their
      encodings alternate point of view — the perspective flip lives in the
      update rule (WP1), not in this contract.
    * ``observe_game_end`` fires once per distinct learning agent when the game
      terminates, with the **absolute** :class:`~bgrl.env.Outcome` (winner +
      magnitude); the agent maps it to its own point of view.

    Both hooks return nothing and must tolerate games the agent does not learn
    from (e.g. evaluation). The argument set carries the full transition so the
    update rule never needs the driver to hand it more.
    """

    def observe_step(self, state: EnvState, dice: Dice, move: Move, afterstate: EnvState) -> None:
        """Record one applied ply ``state --(dice, move)--> afterstate``."""
        ...

    def observe_game_end(self, outcome: Outcome) -> None:
        """Record the terminal result (absolute winner + magnitude)."""
        ...
