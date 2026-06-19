"""Correctness tests for the multi-head TD(λ) core (:mod:`bgrl.training.td_lambda`).

The primary test is the **per-head forward-view equivalence**: with weights frozen
across an episode, the production backward-view update must equal the forward-view
λ-return update over the non-terminal afterstates, summed over heads,

    lr * Σ_t Σ_k (G_t^λ[k] - f_k(a_t)) ∇f_k(a_t),

with the coupled per-head λ-return

    G_t^λ[k] = bias[k] + sign[k]·((1-λ) f_{perm[k]}(a_{t+1}) + λ G_{t+1}^λ[perm[k]]),

``bias = [1,0,0,0,0]``, ``perm = FLIP_PERM``, ``sign = FLIP_SIGN``, seeded by the
winner-to-move terminal target ``_terminal_target(kind)``. At λ=1 each head collapses
to Monte-Carlo regression toward its realised cumulative indicator. We check
λ ∈ {0, 0.5, 1} crossed with ``WinKind`` ∈ {single, gammon, backgammon}: the λ=0.5
case with a gammon/backgammon outcome is the one that pins the **head-pairing** trace
carry (head k carrying from its paired head FLIP_PERM[k] with sign FLIP_SIGN[k]) — a
wrong sign or pairing leaves an O(λ(1-λ)) residual on the magnitude heads. It exercises
the production ``TDLambda`` directly (no re-implementation of the rule). The win
magnitude is *forced* onto the recorded plies: the backward/forward equivalence is an
algebraic identity in the recorded trajectory and the terminal target, independent of
whether the board truly is a gammon (a random-net self-play game is almost always a
single), so heads 1-4 still get non-trivially exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bgrl.agents import ValueAgent
from bgrl.env import Env, Outcome, Player, RandomDiceSource, WinKind, is_terminal, legal_moves
from bgrl.game import play_game
from bgrl.nets.base import OUTCOME_DIM
from bgrl.nets.equity import FLIP_PERM, FLIP_SIGN
from bgrl.nets.value_net import MLPValueNet
from bgrl.training.td_lambda import TDLambda, _terminal_target


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", [WinKind.SINGLE, WinKind.GAMMON, WinKind.BACKGAMMON])
def test_forward_view_equivalence(lam: float, kind: WinKind) -> None:
    """Offline (frozen-weight) backward update equals the forward-view λ-return update.

    Plays one real self-play game, then replays its recorded plies through the
    production ``TDLambda``: snapshot ``θ0``, run each ``step`` (and ``episode_end``
    with the *forced* ``kind``), accumulate the weight change, and restore ``θ0`` after
    every call — so every forward/gradient is evaluated at ``θ0`` (the frozen-weight
    condition) while the real trace recurrence still runs. The summed change must match
    the independently computed per-head forward-view update at ``θ0``.
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

    # Force the win magnitude onto the recorded plies (the winner identity is fixed by
    # the terminal structure; only kind drives the magnitude heads' target).
    forced = Outcome(winner=result.outcome.winner, kind=kind)
    for s in steps:  # terminal afterstates are skipped inside step(); harmless here
        trainer.step(s.state, s.dice, s.move, s.afterstate)
        capture_and_restore()
    trainer.episode_end(forced)
    capture_and_restore()

    # Independent per-head forward-view λ-return update at θ0 (net restored above).
    fvecs: list[np.ndarray] = []
    grads: list[list[tuple[torch.Tensor, ...]]] = []  # grads[t][k] = per-param grads of head k
    for a in nonterminal:
        fa = trainer._value_vec(a, a.turn)
        grads.append(
            [
                torch.autograd.grad(fa[k], params, retain_graph=(k < OUTCOME_DIM - 1))
                for k in range(OUTCOME_DIM)
            ]
        )
        fvecs.append(fa.detach().numpy().astype(np.float64))

    n = len(nonterminal)
    bias = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    g_lambda = [np.zeros(OUTCOME_DIM) for _ in range(n)]
    g_lambda[n - 1] = _terminal_target(kind)  # last valued afterstate's mover is the winner
    for t in range(n - 2, -1, -1):
        for k in range(OUTCOME_DIM):
            pk = FLIP_PERM[k]
            cont = (1.0 - lam) * fvecs[t + 1][pk] + lam * g_lambda[t + 1][pk]
            g_lambda[t][k] = bias[k] + FLIP_SIGN[k] * cont

    forward = [torch.zeros_like(p) for p in params]
    with torch.no_grad():
        for t in range(n):
            for k in range(OUTCOME_DIM):
                coeff = lr * (g_lambda[t][k] - fvecs[t][k])
                for fwd, g in zip(forward, grads[t][k]):
                    fwd.add_(coeff * g)

    for d, fwd in zip(deltas, forward):
        assert torch.allclose(d, fwd, atol=1e-4, rtol=1e-4)


def test_traces_and_prev_reset_after_episode_end() -> None:
    """A ``step`` arms the carry-over/per-head traces; ``episode_end`` clears both."""
    torch.manual_seed(0)
    net = MLPValueNet(hidden=8)
    trainer = TDLambda(net, lam=0.7, gamma=1.0, lr=0.1)

    state = Env.initial_state()
    dice = (3, 1)
    move, afterstate = legal_moves(state, dice)[0]
    trainer.step(state, dice, move, afterstate)

    assert trainer._prev is not None
    assert any(torch.count_nonzero(t) > 0 for head in trainer._traces for t in head)

    # A gammon outcome flows through episode_end without error and consumes kind.
    trainer.episode_end(Outcome(winner=Player.WHITE, kind=WinKind.GAMMON))

    assert trainer._prev is None
    assert all(torch.count_nonzero(t) == 0 for head in trainer._traces for t in head)


def test_terminal_target_is_cumulative_by_magnitude() -> None:
    """The winner-to-move target fires win-magnitude heads cumulatively, loss heads 0."""
    assert np.array_equal(_terminal_target(WinKind.SINGLE), [1, 0, 0, 0, 0])
    assert np.array_equal(_terminal_target(WinKind.GAMMON), [1, 1, 0, 0, 0])
    assert np.array_equal(_terminal_target(WinKind.BACKGAMMON), [1, 1, 1, 0, 0])
