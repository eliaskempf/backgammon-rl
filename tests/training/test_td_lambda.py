"""Correctness tests for the TD(λ) core (:mod:`bgrl.training.td_lambda`).

The primary test is the **forward-view equivalence**: with weights frozen across an
episode, the production backward-view update must equal the forward-view λ-return
update over the non-terminal afterstates,

    lr * Σ_t (G_t^λ - f(a_t)) ∇f(a_t),   G_t^λ = 1 - ((1-λ) f(a_{t+1}) + λ G_{t+1}^λ),

seeded by ``G_{T-1}^λ = 1`` (the last valued afterstate's mover is the winner). At
λ=1 this collapses to Monte-Carlo regression toward the realised win indicator. We
check λ ∈ {0, 0.5, 1}: the λ=0.5 case is the one that pins the **terminal handling**
(skipping the terminal afterstate / bootstrapping the winner-to-move toward 1) — the
alternative of folding the terminal estimate leaves a residual that vanishes only at
λ ∈ {0, 1} and silently breaks intermediate λ. It exercises the production
``TDLambda`` directly (no re-implementation of the rule).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bgrl.agents import ValueAgent
from bgrl.env import Env, Outcome, Player, RandomDiceSource, WinKind, is_terminal, legal_moves
from bgrl.game import play_game
from bgrl.nets.value_net import MLPValueNet
from bgrl.training.td_lambda import TDLambda


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_forward_view_equivalence(lam: float) -> None:
    """Offline (frozen-weight) backward update equals the forward-view λ-return update.

    Plays one real self-play game, then replays its recorded plies through the
    production ``TDLambda``: snapshot ``θ0``, run each ``step`` (and ``episode_end``),
    accumulate the weight change, and restore ``θ0`` after every call — so every
    forward/gradient is evaluated at ``θ0`` (the frozen-weight condition) while the
    real trace recurrence still runs. The summed change must match the independently
    computed forward-view update at ``θ0``.
    """
    torch.manual_seed(0)
    net = MLPValueNet(hidden=16)
    agent = ValueAgent(net)  # rng=None -> deterministic greedy selection
    result = play_game(agent, agent, RandomDiceSource(np.random.default_rng(0)), record=True)

    assert result.outcome is not None, "game should reach a terminal outcome"
    steps = result.steps
    nonterminal = [s.afterstate for s in steps if not is_terminal(s.afterstate)]
    assert len(nonterminal) >= 2, "need a multi-ply game to exercise the trace recurrence"

    lr = 0.1
    trainer = TDLambda(net, lam=lam, gamma=1.0, lr=lr)
    params = list(net.parameters())
    theta0 = [p.detach().clone() for p in params]
    deltas = [torch.zeros_like(p) for p in params]

    def capture_and_restore() -> None:
        """Fold the change since θ0 into ``deltas``, then reset weights to θ0.

        Traces and ``_prev`` are left untouched, so the recurrence continues across
        plies even though the weights never actually move.
        """
        with torch.no_grad():
            for d, p, p0 in zip(deltas, params, theta0):
                d.add_(p - p0)
                p.copy_(p0)

    for s in steps:  # terminal afterstates are skipped inside step(); harmless here
        trainer.step(s.state, s.dice, s.move, s.afterstate)
        capture_and_restore()
    trainer.episode_end(result.outcome)
    capture_and_restore()

    # Independent forward-view λ-return update at θ0 (net is restored to θ0 above).
    fvals: list[float] = []
    grads: list[tuple[torch.Tensor | None, ...]] = []
    for a in nonterminal:
        fa = trainer._value(a, a.turn)
        grads.append(torch.autograd.grad(fa, params, allow_unused=True))
        fvals.append(float(fa.detach()))

    n = len(nonterminal)
    g_lambda = [0.0] * n
    g_lambda[n - 1] = 1.0  # last valued afterstate's mover is the winner
    for t in range(n - 2, -1, -1):
        g_lambda[t] = 1.0 - ((1.0 - lam) * fvals[t + 1] + lam * g_lambda[t + 1])

    forward = [torch.zeros_like(p) for p in params]
    with torch.no_grad():
        for t in range(n):
            coeff = lr * (g_lambda[t] - fvals[t])
            for fwd, g in zip(forward, grads[t]):
                if g is not None:
                    fwd.add_(coeff * g)

    for d, fwd in zip(deltas, forward):
        assert torch.allclose(d, fwd, atol=1e-4, rtol=1e-4)


def test_traces_and_prev_reset_after_episode_end() -> None:
    """A ``step`` arms the carry-over/trace; ``episode_end`` clears both."""
    torch.manual_seed(0)
    net = MLPValueNet(hidden=8)
    trainer = TDLambda(net, lam=0.7, gamma=1.0, lr=0.1)

    state = Env.initial_state()
    dice = (3, 1)
    move, afterstate = legal_moves(state, dice)[0]
    trainer.step(state, dice, move, afterstate)

    assert trainer._prev is not None
    assert any(torch.count_nonzero(t) > 0 for t in trainer._traces)

    trainer.episode_end(Outcome(winner=Player.WHITE, kind=WinKind.SINGLE))

    assert trainer._prev is None
    assert all(torch.count_nonzero(t) == 0 for t in trainer._traces)
