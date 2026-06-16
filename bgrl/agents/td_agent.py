"""TD(λ) self-play learning agent.

Wires the WP0 :class:`~bgrl.agents.value_agent.ValueAgent` (greedy 0-ply selection
by equity) to the online :class:`~bgrl.training.td_lambda.TDLambda` update.
Selection is **greedy** — no ε-exploration — because the dice supply all the
exploration a self-play TD learner needs (the TD-Gammon result). This class is
pure wiring: construction plus forwarding the two lifecycle hooks to the trainer,
so the TD math lives entirely in :mod:`bgrl.training.td_lambda` and none of it
leaks here (note: no ``torch`` import in this module).
"""

from __future__ import annotations

import numpy as np

from bgrl.agents.value_agent import ValueAgent
from bgrl.env import Dice, EnvState, Move, Outcome
from bgrl.nets.equity import CENTERED_CUBE, CubeContext
from bgrl.nets.value_net import MLPValueNet
from bgrl.training.td_lambda import TDLambda


class TDAgent(ValueAgent):
    """A :class:`ValueAgent` that learns its net online via TD(λ) during self-play.

    Plays greedily through the inherited :meth:`ValueAgent.act` (with ``rng=None``
    the tie-break is deterministic, so seeded self-play is reproducible) and learns
    from the games it generates through the
    :class:`~bgrl.agents.base.LearningAgent` hooks, which forward to a
    :class:`~bgrl.training.td_lambda.TDLambda` trainer over the **same** net object.
    """

    def __init__(
        self,
        net: MLPValueNet,
        *,
        lam: float,
        lr: float,
        gamma: float = 1.0,
        cube: CubeContext = CENTERED_CUBE,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(net, cube=cube, rng=rng)
        self._trainer = TDLambda(net, lam=lam, gamma=gamma, lr=lr)

    def observe_step(self, state: EnvState, dice: Dice, move: Move, afterstate: EnvState) -> None:
        self._trainer.step(state, dice, move, afterstate)

    def observe_game_end(self, outcome: Outcome) -> None:
        self._trainer.episode_end(outcome)
