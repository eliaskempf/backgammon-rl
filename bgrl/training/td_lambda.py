"""TD(λ) update for the afterstate value net — the learning core (WP1).

Online TD(λ) with eligibility traces over a **mover-relative** afterstate value:
the quantity learned is ``f(a) = p_win(a)`` read from the point of view of the
player to move at afterstate ``a`` (index 0 of the outcome vector), the same query
the agent makes when ranking moves. The other four heads stay wired but untrained;
the output shape is unchanged (cube-ready, CLAUDE.md §5). The correction is applied
to the weights **manually** (no :mod:`torch.optim`): eligibility traces are not
gradients, so an optimiser would be the wrong abstraction.

This is exactly standard episodic TD(λ) in the centered value ``u(a) = 2·f(a) - 1``,
which obeys a **negamax** recursion ``u(a_t) = -u(a_{t+1})`` (no reward) on
non-terminal transitions and ``u(a_{T-1}) = +1`` (reward at the winning move). That
is plain TD(λ) with discount ``gamma = -1`` and reward 0, so the textbook backward-view
equivalence holds for *every* λ. Written back in ``f``-space it needs two coupled
perspective adjustments (CLAUDE.md §6 — sign-flip bugs live here):

* **Bootstrap complement.** The successor ``a_{t+1}`` is the opponent's afterstate,
  so ``f(a_t)`` bootstraps toward ``1 - f(a_{t+1})``, giving a TD error
  ``δ_t = (1 - f(a_{t+1})) - f(a_t)``.
* **Trace sign-flip.** The value's POV flips every ply, so the eligibility trace
  negates its carry-over: ``e_t = -λγ·e_{t-1} + ∇f(a_t)``. The complement handles
  the one-step target; the sign-flip handles multi-step credit when ``λ > 0``.
* **Terminal handling — the last *valued* afterstate is the winner-to-move, target
  1.** A game ends only by bearing off one's own last checker, so the mover that
  produced the terminal afterstate ``a_T`` is the *winner*; the to-move player *at*
  ``a_T`` is the *loser*. The terminal afterstate is **not a valued state** (its true
  value is the fixed terminal 0): we do **not** fold ``∇f(a_T)`` into the trace and
  do **not** bootstrap off the estimate ``f(a_T)``. Instead the previous afterstate
  ``a_{T-1}`` (whose mover is the winner) is regressed toward its **realised** target
  ``1`` — the only non-bootstrapped signal in the episode. Bootstrapping ``a_{T-1}``
  off the terminal *estimate* ``f(a_T)`` instead (and folding ``a_T``) leaves a
  residual ``λ(1-λ)·f(a_T)`` term in the multi-step credit that vanishes only at
  ``λ ∈ {0, 1}`` — it silently breaks training for intermediate λ (the real
  ``λ=0.7`` setting), which is exactly the bug this design avoids.

(v1's ``p_win`` target is 0/1 only and the winner is fixed by the terminal structure,
so ``Outcome``'s gammon/backgammon magnitude — for the other heads later — is unused
here.)

**Correctness guarantee (forward-view / Monte-Carlo equivalence).** With weights
frozen across an episode, the total backward-view update equals the forward-view
λ-return update ``lr·Σ_t (G_t^λ - f(a_t))·∇f(a_t)`` over the **non-terminal**
afterstates, where ``G_t^λ`` is the λ-return ``G_t^λ = 1 - ((1-λ)·f(a_{t+1}) + λ·G_{t+1}^λ)``
seeded by ``G_{T-1}^λ = 1``. At ``λ=1`` this collapses to Monte-Carlo regression
``Σ_t (y_t - f(a_t))·∇f(a_t)`` with ``y_t = 1 if a_t.turn == outcome.winner else 0``
(using ``y_t + y_{t+1} = 1``). Both are asserted in ``tests/training/test_td_lambda.py``;
the general-λ check is what pins the terminal handling.

The implementation is **deferred online**: each ply folds the current afterstate's
gradient into the trace and applies the (now-complete) correction for the *previous*
afterstate; the winner-to-move correction lands in :meth:`TDLambda.episode_end`.
Undiscounted by default (``gamma = 1.0``); ``gamma`` stays a parameter so the
arithmetic is general.
"""

from __future__ import annotations

import numpy as np
import torch

from bgrl.env import Dice, EnvState, Move, Outcome, Player, encode, is_terminal
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
        """Online TD(λ) update for one ply (terminal afterstates are skipped).

        Fires once per ply, in afterstate order, with the transition
        ``state --(dice, move)--> afterstate`` (``afterstate.turn`` is the opponent,
        since they move next). Only ``afterstate`` is used — ``f`` is a pure
        afterstate value — but ``state`` / ``dice`` / ``move`` are part of the
        contract.

        Deferred online structure: compute ``f(a_t) = _value(afterstate,
        afterstate.turn)`` with a fresh graph, fold its gradient into the
        sign-flipped eligibility trace, and apply the complement-bootstrap correction
        for the *previous* afterstate ``a_{t-1}`` (whose target ``1 - f(a_t)`` only
        becomes known now). The winner-to-move correction for the last *valued*
        afterstate is left to :meth:`episode_end`.

        A **terminal** afterstate is not a valued state (its true value is the fixed
        terminal 0): we skip it entirely — no forward, no gradient folded, no
        bootstrap off its estimate — so the previous afterstate is regressed toward
        its realised target in :meth:`episode_end` rather than toward ``f(a_T)``. See
        the module docs for why folding it breaks training at intermediate λ.

        Only the **detached scalar** ``f(a_t)`` is cached in ``self._prev``; a fresh
        forward is recomputed each ply. Caching a graph-bearing tensor and calling
        ``.backward()`` on it a ply later would raise (the in-place weight update
        mutates a tensor the saved graph needs) — see CLAUDE.md §6 / the module docs.
        """
        if is_terminal(afterstate):
            return

        self.net.zero_grad()
        v_cur = self._value(afterstate, afterstate.turn)  # f(a_t), fresh graph
        v_cur.backward()  # param.grad = ∇f(a_t)

        with torch.no_grad():
            if self._prev is not None:
                # Complete a_{t-1}'s deferred update: target 1 - f(a_t) (opponent's
                # afterstate), applied over the trace e_{t-1} as it stands now.
                delta = (1.0 - v_cur.detach()) - self._prev
                for param, trace in zip(self.net.parameters(), self._traces, strict=True):
                    param.add_(self.lr * delta * trace)
            # Fold ∇f(a_t) into the sign-flipped trace: e_t = -λγ·e_{t-1} + ∇f(a_t).
            for i, param in enumerate(self.net.parameters()):
                self._traces[i] = -self.lam * self.gamma * self._traces[i] + param.grad
            self._prev = v_cur.detach()

    def episode_end(self, outcome: Outcome) -> None:
        """Winner-to-move terminal update at game end, then reset for next episode.

        Fires once when the game terminates. ``self._prev`` holds ``f(a_{T-1})``, the
        last *valued* afterstate (the terminal afterstate ``a_T`` was skipped by
        :meth:`step`). Its mover made the move that bore off the final checker, so it
        is the **winner**; its realised target is therefore ``1``. Complete that
        deferred update over the current trace, then clear the eligibility traces and
        the per-episode carry-over so the next episode starts clean.

        This realised ``1`` is the only non-bootstrapped signal in the episode.
        ``outcome`` is unused in v1: the ``p_win`` target is 0/1 only and the winner
        is fixed by the terminal structure, not the magnitude. The gammon/backgammon
        magnitude carried by :class:`~bgrl.env.Outcome` is reserved for the other
        heads later.
        """
        with torch.no_grad():
            if self._prev is not None:  # guard the degenerate 0-ply episode
                delta = 1.0 - self._prev  # winner-to-move target 1 for f(a_{T-1})
                for param, trace in zip(self.net.parameters(), self._traces, strict=True):
                    param.add_(self.lr * delta * trace)

        for trace in self._traces:
            trace.zero_()
        self._prev = None
