"""Self-study scaffold for the multi-head TD(λ) update (CLAUDE.md §11).

This module is **not imported anywhere** and **does not gate training** — the real,
tested implementation lives in :mod:`bgrl.training.td_lambda` and is what runs. This
file exists only so the multi-head generalisation of WP1's scalar TD(λ) can be worked
through by hand at leisure: the orchestration (the deferred-online flow, the terminal
skip, taking one gradient per head, the trace reset) is written out, and the three
learning-critical pieces are left as ``TODO(human)`` stubs whose docstrings state the
**invariant** each must satisfy, never the line that satisfies it.

The three things to derive:

1. :meth:`TDLambdaExercise._bootstrap_target` — how a successor value, which is in the
   *opponent's* point of view, maps back to the mover's POV target.
2. :meth:`TDLambdaExercise._carry_trace` — how each head's eligibility trace carries
   across a ply, given that the POV flips every ply.
3. :meth:`TDLambdaExercise._terminal_target` — the realised target at the winning move.

When you have filled them in, compare against :mod:`bgrl.training.td_lambda` and run the
forward-view / Monte-Carlo equivalence tests in ``tests/training/test_td_lambda.py``
against your version (point them at this class) to check the head-pairing carry. No
peeking at :data:`bgrl.nets.equity.FLIP_PERM` / ``FLIP_SIGN`` until you have a candidate.
"""

from __future__ import annotations

import numpy as np
import torch

from bgrl.env import Dice, EnvState, Move, Outcome, Player, WinKind, encode, is_terminal
from bgrl.nets.base import OUTCOME_DIM
from bgrl.nets.value_net import MLPValueNet


class TDLambdaExercise:
    """A scaffolded twin of :class:`bgrl.training.td_lambda.TDLambda` to fill in.

    The structure, shapes, and lifecycle match the production trainer exactly; only the
    three ``TODO(human)`` methods are missing. Everything else — the deferred-online
    bookkeeping, the terminal-afterstate skip, the per-head gradient extraction, the
    weight update over traces, and the episode reset — is provided so the focus stays on
    the perspective algebra.
    """

    def __init__(self, net: MLPValueNet, *, lam: float, gamma: float, lr: float) -> None:
        self.net = net
        self.lam = lam
        self.gamma = gamma
        self.lr = lr
        # One eligibility-trace set per outcome head, each a list of per-parameter
        # tensors. Persist across an episode's plies; cleared at each episode boundary.
        self._traces: list[list[torch.Tensor]] = [
            [torch.zeros_like(p) for p in net.parameters()] for _ in range(OUTCOME_DIM)
        ]
        # Detached previous-ply value vector V(a_{t-1}) (numpy, OUTCOME_DIM), or None at
        # an episode boundary.
        self._prev: np.ndarray | None = None

    def _value_vec(self, afterstate: EnvState, perspective: Player) -> torch.Tensor:
        """Differentiable outcome vector of ``afterstate`` from ``perspective``'s POV.

        Provided plumbing: encodes the position and runs the differentiable forward pass
        (not :meth:`MLPValueNet.evaluate`, which has no autograd graph), returning the
        ``(OUTCOME_DIM,)`` tensor the per-head gradients are taken from.
        """
        features = encode(afterstate, perspective)
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        return self.net(x)

    # ------------------------------------------------------------------ TODO(human) ---

    def _bootstrap_target(self, succ: np.ndarray) -> np.ndarray:
        """TODO(human): the target the *previous* afterstate ``a_{t-1}`` regresses toward.

        ``succ = V(a_t)`` is the value of ``a_{t-1}``'s successor. But ``a_t`` is the
        *opponent's* afterstate, so ``succ`` is expressed in the opponent's point of
        view. The target for ``a_{t-1}`` must be in ``a_{t-1}``'s mover's point of view.

        Invariant to satisfy: switching sides relabels every outcome. Whatever "I win"
        means to one player is exactly "the other player loses" to the other; likewise a
        gammon/backgammon win for one side is a gammon/backgammon *loss* for the other.
        Translate each of the OUTCOME_DIM heads of ``succ`` accordingly and return a
        length-``OUTCOME_DIM`` vector. (Reaching for
        :func:`bgrl.nets.equity.flip_outcome` is the answer — derive it first.)
        """
        raise NotImplementedError

    def _carry_trace(
        self,
        head: int,
        prev_traces: list[list[torch.Tensor]],
        head_grads: list[tuple[torch.Tensor, ...]],
    ) -> list[torch.Tensor]:
        """TODO(human): the new eligibility trace for ``head`` (one tensor per parameter).

        Invariant to satisfy: an eligibility trace is the current gradient plus a decayed
        carry of the previous ply's trace. In the *scalar* WP1 case that carry was
        ``-λ·gamma`` times this head's own previous trace — the minus encoding the POV
        flip. Generalise it: because the value flips POV every ply, the carry for ``head``
        does not in general come from ``head``'s own previous trace, but from whichever
        head ``head`` *maps onto* under that same perspective flip (the one
        :meth:`_bootstrap_target` encodes), and the sign of the carry is whatever that
        mapping implies for this head. ``head_grads[k]`` is ``∇f_k(a_t)`` as a tuple
        aligned to ``net.parameters()``; ``prev_traces[k]`` is head ``k``'s
        previous-ply trace list. Return the per-parameter trace list for ``head``.

        (Hint on a self-check: for whichever head plays the role WP1's single head did,
        your rule must reduce exactly to ``-λ·gamma·e_prev + grad``.)
        """
        raise NotImplementedError

    def _terminal_target(self, kind: WinKind) -> np.ndarray:
        """TODO(human): the realised target for the last *valued* afterstate.

        Invariant to satisfy: the last valued afterstate's mover made the move that bore
        off the final checker, so that mover is the **winner** — its realised outcome is
        a certainty, not an estimate. Encode the win of magnitude ``kind`` (single /
        gammon / backgammon) as a length-``OUTCOME_DIM`` vector in the winner's POV,
        remembering the heads are cumulative and that the winner did not lose. Return it.
        """
        raise NotImplementedError

    # ------------------------------------------------------------ provided orchestration

    def step(self, state: EnvState, dice: Dice, move: Move, afterstate: EnvState) -> None:
        """Deferred-online per-ply update (provided, given the three methods above).

        Skips terminal afterstates (not valued states), takes one gradient per head,
        completes the previous afterstate's deferred correction toward
        :meth:`_bootstrap_target`, carries the traces via :meth:`_carry_trace`, and
        caches the detached value for next ply.
        """
        if is_terminal(afterstate):
            return

        params = list(self.net.parameters())
        v_cur = self._value_vec(afterstate, afterstate.turn)  # (OUTCOME_DIM,), fresh graph
        head_grads: list[tuple[torch.Tensor, ...]] = [
            torch.autograd.grad(v_cur[k], params, retain_graph=(k < OUTCOME_DIM - 1))
            for k in range(OUTCOME_DIM)
        ]

        with torch.no_grad():
            succ = v_cur.detach().numpy()
            if self._prev is not None:
                delta = self._bootstrap_target(succ) - self._prev
                for k in range(OUTCOME_DIM):
                    dk = self.lr * float(delta[k])
                    for param, trace in zip(params, self._traces[k], strict=True):
                        param.add_(dk * trace)

            self._traces = [
                self._carry_trace(k, self._traces, head_grads) for k in range(OUTCOME_DIM)
            ]
            self._prev = succ

    def episode_end(self, outcome: Outcome) -> None:
        """Winner-to-move terminal update, then reset (provided).

        Completes the deferred correction for the last valued afterstate toward
        :meth:`_terminal_target`, then clears the traces and carry-over.
        """
        with torch.no_grad():
            if self._prev is not None:
                delta = self._terminal_target(outcome.kind) - self._prev
                for k in range(OUTCOME_DIM):
                    dk = self.lr * float(delta[k])
                    for param, trace in zip(self.net.parameters(), self._traces[k], strict=True):
                        param.add_(dk * trace)

        for head_traces in self._traces:
            for trace in head_traces:
                trace.zero_()
        self._prev = None
