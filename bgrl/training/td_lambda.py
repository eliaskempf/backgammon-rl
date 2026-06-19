"""Multi-head TD(λ) update for the afterstate value net — the learning core.

Online TD(λ) with eligibility traces over a **mover-relative** afterstate value.
WP1 learned a single head; WP6 generalises the *same* update to the full cube-ready
outcome vector ``V(a) = [p_win, p_win_g, p_win_bg, p_lose_g, p_lose_bg]`` (CLAUDE.md
§5), each entry read from the point of view of the player to move at afterstate ``a``
— the same query the agent makes when ranking moves. The correction is applied to the
weights **manually** (no :mod:`torch.optim`): eligibility traces are not gradients, so
an optimiser would be the wrong abstraction.

For head 0 (``p_win``) this is exactly standard episodic TD(λ) in the centered value
``u(a) = 2·f(a) - 1``, which obeys a **negamax** recursion ``u(a_t) = -u(a_{t+1})``
(no reward) on non-terminal transitions and ``u(a_{T-1}) = +1`` at the winning move —
plain TD(λ) with discount ``gamma = -1``, so the textbook backward-view equivalence
holds for *every* λ. The four magnitude heads ride the same machinery; the only thing
that generalises is *how the perspective flips between plies* (CLAUDE.md §6 — sign-flip
bugs live here):

* **Bootstrap target = the perspective flip of the successor.** ``a_{t+1}`` is the
  opponent's afterstate, so ``V(a_t)`` bootstraps toward :func:`flip_outcome`
  ``(V(a_{t+1}))``: head 0 complements (``1 - p_win``; win ⟺ opponent loses) while the
  win/lose **gammon** heads {1,3} and **backgammon** heads {2,4} *swap* (my win-gammon
  chance ⟺ the opponent's lose-gammon chance). This is the single linear involution
  :data:`~bgrl.nets.equity.FLIP_PERM` / :data:`~bgrl.nets.equity.FLIP_SIGN`, shared
  with the cube evaluator so the two can never drift.

* **Per-head eligibility traces.** A vector value has a Jacobian, not a scalar
  gradient: each head owns its own trace set (~5x storage). The TD error ``δ`` is a
  5-vector; component ``δ_k`` scales head ``k``'s trace. The cross-ply carry uses the
  **same** ``(perm, sign)`` as the bootstrap target — head ``k``'s trace carries from
  its *paired* head ``FLIP_PERM[k]`` of the previous ply, scaled by
  ``λ·gamma·FLIP_SIGN[k]``::

      e_t[k] = (λ·gamma·FLIP_SIGN[k]) · e_{t-1}[FLIP_PERM[k]] + ∇f_k(a_t)

  Head 0 reproduces WP1's scalar ``e_t = -λγ·e_{t-1} + ∇f`` exactly (``perm[0]=0``,
  ``sign[0]=-1``) — the strongest internal check. The swapped pairs carry ``+λγ`` from
  their paired head: the POV flip shows up there as the index swap, **not** a sign
  flip, because those heads relabel rather than complement. This pairing is the primary
  correctness locus; the general-λ test pins it.

* **Terminal target — the last *valued* afterstate is the winner-to-move.** A game ends
  only by bearing off one's own last checker, so the mover that produced the terminal
  afterstate ``a_T`` is the *winner*; the to-move player *at* ``a_T`` is the *loser*.
  The terminal afterstate is **not a valued state** (its true value is the fixed
  terminal 0): we do **not** fold ``∇V(a_T)`` into the trace and do **not** bootstrap
  off ``V(a_T)``. Instead ``a_{T-1}`` (whose mover is the winner) is regressed toward
  its **realised** cumulative outcome, now read from ``outcome.kind`` (WP1 discarded
  it)::

      target = [1, kind ≥ GAMMON, kind ≥ BACKGAMMON, 0, 0]

  (Single win → ``[1,0,0,0,0]``; the loser's heads are 0 because the winner did not
  lose.) Bootstrapping ``a_{T-1}`` off the terminal *estimate* ``V(a_T)`` instead
  leaves a residual ``λ(1-λ)·V(a_T)`` in the multi-step credit that vanishes only at
  ``λ ∈ {0, 1}`` — it silently breaks training at the real ``λ=0.7`` setting.

**Correctness guarantee (forward-view / Monte-Carlo equivalence), per head.** With
weights frozen across an episode, the total backward-view update equals the
forward-view λ-return update ``lr·Σ_t Σ_k (G_t^λ[k] - f_k(a_t))·∇f_k(a_t)`` over the
**non-terminal** afterstates, where the λ-return is the coupled recurrence
``G_t^λ[k] = bias[k] + sign[k]·((1-λ)·f_{perm[k]}(a_{t+1}) + λ·G_{t+1}^λ[perm[k]])``
(``bias = [1,0,0,0,0]``) seeded by the terminal target above. At ``λ=1`` each head
collapses to Monte-Carlo regression toward its realised cumulative indicator, the
magnitude landing on the win heads on the winner's plies and the lose heads on the
loser's. Both are asserted in ``tests/training/test_td_lambda.py``; the general-λ check
with a forced gammon/backgammon outcome is what pins the head-pairing carry.

The implementation is **deferred online**: each ply folds the current afterstate's
per-head gradients into the traces and applies the (now-complete) correction for the
*previous* afterstate; the winner-to-move correction lands in
:meth:`TDLambda.episode_end`. Undiscounted by default (``gamma = 1.0``); ``gamma``
stays a parameter so the arithmetic is general.
"""

from __future__ import annotations

import numpy as np
import torch

from bgrl.env import Dice, EnvState, Move, Outcome, Player, WinKind, encode, is_terminal
from bgrl.nets.base import OUTCOME_DIM
from bgrl.nets.equity import FLIP_PERM, FLIP_SIGN, flip_outcome
from bgrl.nets.value_net import MLPValueNet


def _terminal_target(kind: WinKind) -> np.ndarray:
    """Winner-to-move realised cumulative outcome for the magnitude ``kind``.

    The last valued afterstate's mover bore off the final checker and is the winner,
    so ``p_win = 1`` and the win-magnitude heads fire cumulatively while the loss heads
    stay 0 (the winner did not lose). Returns a float64 ``(OUTCOME_DIM,)`` vector.
    """
    return np.array(
        [
            1.0,
            1.0 if kind >= WinKind.GAMMON else 0.0,
            1.0 if kind >= WinKind.BACKGAMMON else 0.0,
            0.0,
            0.0,
        ]
    )


class TDLambda:
    """Online multi-head TD(λ) trainer for one self-play learner across many episodes.

    :meth:`step` fires once per ply (via the agent's ``observe_step``) and
    :meth:`episode_end` once per game (via ``observe_game_end``). The instance owns
    the net's per-head eligibility traces and the per-episode carry-over between plies.
    """

    def __init__(self, net: MLPValueNet, *, lam: float, gamma: float, lr: float) -> None:
        self.net = net
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        # One eligibility-trace set **per outcome head**, each a list of per-parameter
        # tensors (so storage is ~OUTCOME_DIM x the scalar case). They persist across the
        # plies of an episode and are cleared at each episode boundary (see
        # episode_end). Storage is provided; maintaining them is the update rule's job.
        self._traces: list[list[torch.Tensor]] = [
            [torch.zeros_like(p) for p in net.parameters()] for _ in range(OUTCOME_DIM)
        ]
        # Per-episode carry-over: the previous ply's detached value vector V(a_{t-1})
        # (numpy, OUTCOME_DIM), needed to complete its deferred update once the target
        # becomes known. None at an episode boundary.
        self._prev: np.ndarray | None = None

    def _value_vec(self, afterstate: EnvState, perspective: Player) -> torch.Tensor:
        """Differentiable outcome **vector** of ``afterstate`` from ``perspective``'s POV.

        Encodes the position from ``perspective`` and runs the **differentiable**
        forward pass (:meth:`MLPValueNet.forward`), returning the full
        ``(OUTCOME_DIM,)`` tensor that carries gradients back to the net's parameters —
        one Jacobian row per head is taken from it in :meth:`step`.

        Use this, **not** :meth:`MLPValueNet.evaluate`: ``evaluate`` runs under
        :func:`torch.inference_mode` and returns NumPy, so it produces no autograd
        graph and cannot update the net. The caller picks which ``perspective`` to
        pass and detaches where a value must act as a fixed bootstrap target.
        """
        features = encode(afterstate, perspective)
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        return self.net(x)

    def step(self, state: EnvState, dice: Dice, move: Move, afterstate: EnvState) -> None:
        """Online multi-head TD(λ) update for one ply (terminal afterstates skipped).

        Fires once per ply, in afterstate order, with the transition
        ``state --(dice, move)--> afterstate`` (``afterstate.turn`` is the opponent,
        since they move next). Only ``afterstate`` is used — ``V`` is a pure afterstate
        value — but ``state`` / ``dice`` / ``move`` are part of the contract.

        Deferred online structure: compute ``V(a_t) = _value_vec(afterstate,
        afterstate.turn)`` with a fresh graph, take one gradient per head, fold them
        into the paired eligibility traces, and apply the now-known correction for the
        *previous* afterstate ``a_{t-1}`` (target :func:`flip_outcome` ``(V(a_t))``).
        The winner-to-move correction for the last *valued* afterstate is left to
        :meth:`episode_end`.

        A **terminal** afterstate is not a valued state (its true value is the fixed
        terminal 0): we skip it entirely — no forward, no gradient folded, no bootstrap
        off its estimate — so the previous afterstate is regressed toward its realised
        target in :meth:`episode_end` rather than toward ``V(a_T)``. See the module
        docs for why folding it breaks training at intermediate λ.

        Only the **detached** ``V(a_t)`` (numpy) is cached in ``self._prev``; a fresh
        forward is recomputed each ply. Caching a graph-bearing tensor and calling
        ``backward`` on it a ply later would raise — the in-place weight update mutates
        a tensor the saved graph needs (CLAUDE.md §6 / the module docs).
        """
        if is_terminal(afterstate):
            return

        params = list(self.net.parameters())
        v_cur = self._value_vec(afterstate, afterstate.turn)  # (OUTCOME_DIM,), fresh graph
        # One Jacobian row per head: ∇_θ f_k(a_t). retain_graph keeps the shared-trunk
        # graph alive across the heads; the last call frees it. Every param feeds every
        # head through the trunk, so no grad is None.
        head_grads: list[tuple[torch.Tensor, ...]] = [
            torch.autograd.grad(v_cur[k], params, retain_graph=(k < OUTCOME_DIM - 1))
            for k in range(OUTCOME_DIM)
        ]

        with torch.no_grad():
            succ = v_cur.detach().numpy()  # V(a_t), opponent of a_{t-1}'s mover
            if self._prev is not None:
                # Complete a_{t-1}'s deferred update: target flip_outcome(V(a_t)),
                # per-head error δ_k scaling head k's trace as it stands now.
                delta = flip_outcome(succ) - self._prev
                for k in range(OUTCOME_DIM):
                    dk = self.lr * float(delta[k])
                    for param, trace in zip(params, self._traces[k], strict=True):
                        param.add_(dk * trace)

            # Carry each head's trace from its PAIRED previous-ply head (the same
            # (perm, sign) as the bootstrap target), then fold in ∇f_k(a_t). Built into
            # a fresh list so the permuted reads see the *previous* traces, not a
            # half-updated state.
            new_traces: list[list[torch.Tensor]] = []
            for k in range(OUTCOME_DIM):
                carry = self.lam * self.gamma * float(FLIP_SIGN[k])
                paired = self._traces[FLIP_PERM[k]]
                new_traces.append(
                    [carry * e_prev + g for e_prev, g in zip(paired, head_grads[k], strict=True)]
                )
            self._traces = new_traces
            self._prev = succ

    def episode_end(self, outcome: Outcome) -> None:
        """Winner-to-move terminal update at game end, then reset for next episode.

        Fires once when the game terminates. ``self._prev`` holds ``V(a_{T-1})``, the
        last *valued* afterstate (the terminal afterstate ``a_T`` was skipped by
        :meth:`step`). Its mover bore off the final checker, so it is the **winner**;
        its realised target is the cumulative one-hot for ``outcome.kind`` (see
        :func:`_terminal_target`). Complete that deferred per-head update over the
        current traces, then clear the traces and the carry-over so the next episode
        starts clean.

        This realised target is the only non-bootstrapped signal in the episode.
        ``outcome.winner`` is not needed — the winner is fixed by the terminal
        structure; only the **magnitude** is consumed (the multi-head generalisation of
        WP1, which discarded ``kind`` entirely).
        """
        with torch.no_grad():
            if self._prev is not None:  # guard the degenerate 0-ply episode
                delta = _terminal_target(outcome.kind) - self._prev
                for k in range(OUTCOME_DIM):
                    dk = self.lr * float(delta[k])
                    for param, trace in zip(self.net.parameters(), self._traces[k], strict=True):
                        param.add_(dk * trace)

        for head_traces in self._traces:
            for trace in head_traces:
                trace.zero_()
        self._prev = None
