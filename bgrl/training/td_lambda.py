"""TD(λ) update for the afterstate value net — the learning core (WP1).

**This module is the self-implemented core (CLAUDE.md §11 / `plans/wp1-td-lambda.md`).**
The surrounding structure is provided — the net handle, the hyperparameters, the
eligibility-trace storage, and the differentiable value helper :meth:`TDLambda._value`
— but the actual TD(λ) update is left for the human to implement at the
``# TODO(human)`` markers in :meth:`TDLambda.step` and :meth:`TDLambda.episode_end`.
The comments there state the *invariant each block must satisfy*, never the line
that satisfies it.

Learning setup (the problem definition, not the solution):

* **Online TD(λ)** with eligibility traces — the net is updated once per ply as the
  game is played, not in batches.
* **Afterstate value, ``p_win`` head only (v1).** The value of a position is the
  network's ``p_win`` output (index 0 of the outcome vector), read from the
  *mover's* point of view. The other four heads stay wired but untrained; the
  output shape is unchanged (cube-ready, CLAUDE.md §5).
* **Reward** is 0 on every non-terminal ply and the realised game result at
  termination — the only learning signal beyond bootstrapping.
* **Undiscounted** episodic returns by default (``gamma = 1.0``); ``gamma`` stays a
  parameter so the trace/return arithmetic is written in general form.
* **Perspective.** Consecutive afterstates belong to opposite movers, so their
  ``p_win`` values are stated from opposite points of view; relating two successive
  values requires reconciling that sign flip. This is the classic backgammon-TD bug
  locus and is part of the work below.

The correction is applied to the weights **manually** (no :mod:`torch.optim`):
eligibility traces are not gradients, so an optimiser would be the wrong abstraction.
"""

from __future__ import annotations

import numpy as np
import torch

from bgrl.env import Dice, EnvState, Move, Outcome, Player, encode
from bgrl.nets.value_net import MLPValueNet

_PWIN = 0  # index of the p_win head in the outcome vector (v1 trains only this head)


class TDLambda:
    """Online TD(λ) trainer for one self-play learner across many episodes.

    :meth:`step` fires once per ply (via the agent's ``observe_step``) and
    :meth:`episode_end` once per game (via ``observe_game_end``). The instance owns
    the net's eligibility traces and the per-episode carry-over between plies.
    """

    def __init__(self, net: MLPValueNet, *, lam: float, gamma: float, lr: float) -> None:
        self.net = net
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        # One eligibility trace per trainable parameter, same shape, starting at
        # zero. They persist across the plies of an episode and are cleared at each
        # episode boundary (see episode_end). Storage is provided; maintaining them
        # is the update rule's job.
        self._traces: list[torch.Tensor] = [torch.zeros_like(p) for p in net.parameters()]
        # Per-episode carry-over: whatever the update needs from the previous ply to
        # relate it to the current one (e.g. the previous afterstate's value). None
        # at an episode boundary; what to store here is part of the update rule.
        self._prev: torch.Tensor | None = None

    def _value(self, afterstate: EnvState, perspective: Player) -> torch.Tensor:
        """Differentiable ``p_win`` of ``afterstate`` from ``perspective``'s POV.

        Encodes the position from ``perspective`` and runs the **differentiable**
        forward pass (:meth:`MLPValueNet.forward`), returning the ``p_win`` head as a
        0-d tensor that carries gradients back to the net's parameters — the value
        the TD update is built on.

        Use this, **not** :meth:`MLPValueNet.evaluate`: ``evaluate`` runs under
        :func:`torch.inference_mode` and returns NumPy, so it produces no autograd
        graph and cannot update the net. The caller picks which ``perspective`` to
        pass and is responsible for detaching where a value must act as a fixed
        bootstrap target rather than carry gradients.
        """
        features = encode(afterstate, perspective)
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        return self.net(x)[..., _PWIN]

    def step(self, state: EnvState, dice: Dice, move: Move, afterstate: EnvState) -> None:
        """Online TD(λ) update for one (non-terminal) ply.

        Fires once per ply, in afterstate order, with the transition
        ``state --(dice, move)--> afterstate`` (``afterstate.turn`` is the opponent,
        since they move next). Across the episode the update must satisfy:

        * Eligibility of previously visited afterstates carries forward and fades
          over subsequent plies, while the just-visited afterstate becomes freshly
          eligible.
        * The correction is driven by the **TD error between successive afterstate
          values, expressed from a single consistent point of view** — mind the sign
          flip between plies (the classic bug locus).
        * Apply the correction to the net's parameters in place, using the
          eligibility traces, without retaining the autograd graph from one ply into
          the next.

        ``state`` / ``dice`` / ``move`` are part of the contract; a pure afterstate
        value method may not need all of them.
        """
        # TODO(human): implement the online TD(λ) per-ply update described above,
        # using self._value(...), self._traces, self._prev, and the hyperparameters.
        raise NotImplementedError("TODO(human): TD(λ) per-ply update — WP1 Phase B")

    def episode_end(self, outcome: Outcome) -> None:
        """Terminal update at game end, then reset for the next episode.

        Fires once when the game terminates, with the **absolute** ``outcome``
        (winner + magnitude). The update must satisfy:

        * The final afterstate's target is the **realised game result mapped to the
          relevant mover's point of view**, never a value-net estimate — the only
          non-bootstrapped signal in the episode.
        * After applying the terminal correction, clear the eligibility traces and
          the per-episode carry-over so the next episode starts clean.

        v1 reduces the outcome to win/lose for the ``p_win`` head; the
        gammon/backgammon magnitude carried by :class:`~bgrl.env.Outcome` is
        available for richer targets later.
        """
        # TODO(human): implement the terminal update + reset described above.
        raise NotImplementedError("TODO(human): TD(λ) terminal update + reset — WP1 Phase B")
