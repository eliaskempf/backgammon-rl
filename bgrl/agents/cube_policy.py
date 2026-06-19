"""Pre-roll position evaluation and the cube-decision dispatch glue (WP6 B3).

The cube decision needs an *outcome distribution*, not a move. A :class:`PositionEvaluator`
produces one for a pre-roll position; because the cube is decided before the dice are
thrown, the evaluator must average over the on-roll player's rolls (>=1-ply is the natural
floor). :class:`NetEvaluator` is the 1-ply adapter over a raw :class:`~bgrl.nets.base.ValueNet`
(the checker selector and the cube decider read the *same* net, neither wrapping the other).
A deeper, vector-valued n-ply evaluator (an ``ExpectimaxAgent`` that also implements
``evaluate_outcome``) is the natural extension; it is deferred behind the gnubg cross-check
(WP6 B4), and :func:`evaluator_for` already routes to it via the :class:`PositionEvaluator`
protocol when present.

The money-game loop never calls an agent's cube methods directly; it calls
:func:`wants_to_double` / :func:`wants_to_take`, which delegate to the agent's own policy
when it is :class:`~bgrl.agents.base.CubeCapable` and otherwise to a shared
:class:`~bgrl.nets.cube.CubeDecider` over the agent's evaluator. So existing agents
(``RandomAgent``, ``ValueAgent``, ``ExpectimaxAgent``, ``LLMAgent``) need no changes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np

from bgrl.agents.base import Agent, CubeCapable
from bgrl.env import WEIGHTED_ROLLS, EnvState, encode, is_terminal, legal_moves, outcome
from bgrl.nets.base import OUTCOME_DIM, ValueNet
from bgrl.nets.cube import CubeAction, CubeDecider, TakeAction
from bgrl.nets.equity import (
    CENTERED_CUBE,
    CubeAccess,
    CubeContext,
    cube_access,
    equity,
    flip_outcome,
    outcome_to_vector,
)


@runtime_checkable
class PositionEvaluator(Protocol):
    """Produces a pre-roll cubeless outcome distribution for cube decisions."""

    def evaluate_outcome(self, state: EnvState) -> np.ndarray:
        """Cubeless 5-vector from the on-roll player's (``state.turn``) POV.

        Averaged over that player's 21 dice (pre-roll), so it reflects the on-roll
        distribution a cube decision is made against. Cumulative gnubg-style heads.
        """
        ...


def _mover_outcomes(net: ValueNet, afterstates: list[EnvState]) -> np.ndarray:
    """``(N, OUTCOME_DIM)`` cubeless vectors from the just-moved mover's POV.

    Each afterstate's ``turn`` is the opponent, so the net gives the opponent's
    distribution; :func:`~bgrl.nets.equity.flip_outcome` returns it to the mover. Terminal
    afterstates are scored exactly from the outcome (the mover just bore off, so it is the
    winner) rather than the net.
    """
    out = np.zeros((len(afterstates), OUTCOME_DIM))
    rows: list[int] = []
    feats: list[np.ndarray] = []
    for i, after in enumerate(afterstates):
        if is_terminal(after):
            result = outcome(after)
            assert result is not None, "is_terminal guarantees an outcome"
            out[i] = outcome_to_vector(result, after.turn.opponent())
        else:
            rows.append(i)
            feats.append(encode(after, after.turn))
    if feats:
        out[rows] = flip_outcome(net.evaluate(np.stack(feats)))
    return out


def onroll_outcome(net: ValueNet, state: EnvState, cube: CubeContext = CENTERED_CUBE) -> np.ndarray:
    """1-ply pre-roll cubeless 5-vector for ``state.turn`` (averaged over the 21 rolls).

    For each roll the on-roll player picks the reply that maximises its own equity (the
    same greedy selection :class:`~bgrl.agents.value_agent.ValueAgent` makes), and that
    reply's mover-POV outcome vector is averaged in. A roll with no legal move is the
    forced pass — a single afterstate with the turn flipped, exactly as the game driver
    records it — so the recursion bottoms out at the net (no unbounded pass chains).
    """
    mover = state.turn
    total = np.zeros(OUTCOME_DIM)
    for roll, weight in WEIGHTED_ROLLS:
        replies = legal_moves(state, roll)
        if replies:
            afterstates = [after for _, after in replies]
        else:
            afterstates = [replace(state, turn=mover.opponent())]  # forced pass
        vecs = _mover_outcomes(net, afterstates)
        total += weight * vecs[int(np.argmax(equity(vecs, cube)))]
    return total


class NetEvaluator:
    """The 1-ply pre-roll :class:`PositionEvaluator` over a raw :class:`ValueNet` (Adapter A)."""

    def __init__(self, net: ValueNet, cube: CubeContext = CENTERED_CUBE) -> None:
        self._net = net
        self._cube = cube

    def evaluate_outcome(self, state: EnvState) -> np.ndarray:
        return onroll_outcome(self._net, state, self._cube)


def evaluator_for(
    agent: Agent,
    *,
    fallback_net: ValueNet | None = None,
    cube: CubeContext = CENTERED_CUBE,
) -> PositionEvaluator | None:
    """The evaluator a default cube policy should use for ``agent``.

    An agent that is itself a :class:`PositionEvaluator` (a future n-ply evaluator) is used
    directly. Otherwise, an agent carrying a :class:`ValueNet` (``ValueAgent``, ``TDAgent``,
    ``ExpectimaxAgent``) gets a 1-ply :class:`NetEvaluator` over that net. A net-less agent
    (``RandomAgent``, ``LLMAgent``) gets the injected ``fallback_net`` if any, else ``None``
    — the dispatchers read ``None`` as "never double / always take".
    """
    if isinstance(agent, PositionEvaluator):
        return agent
    net = getattr(agent, "_net", None)
    if isinstance(net, ValueNet):
        return NetEvaluator(net, cube)
    if fallback_net is not None:
        return NetEvaluator(fallback_net, cube)
    return None


def wants_to_double(
    agent: Agent,
    state: EnvState,
    cube: CubeContext,
    *,
    evaluator: PositionEvaluator | None,
    decider: CubeDecider,
) -> bool:
    """Whether ``agent`` (on roll in ``state``) offers a double.

    Delegates to the agent's own policy if it is :class:`~bgrl.agents.base.CubeCapable`,
    else to ``decider`` over ``evaluator``. A net-less agent (``evaluator is None``) or one
    that cannot legally double (opponent owns the cube) never doubles.
    """
    if isinstance(agent, CubeCapable):
        return agent.should_double(state, cube)
    access = cube_access(state.turn, cube)
    if access is CubeAccess.OPP_OWNS or evaluator is None:
        return False
    on_roll = evaluator.evaluate_outcome(state)
    return decider.decide_double(on_roll, cube, access=access) is CubeAction.DOUBLE


def wants_to_take(
    agent: Agent,
    state: EnvState,
    cube: CubeContext,
    *,
    evaluator: PositionEvaluator | None,
    decider: CubeDecider,
) -> bool:
    """Whether ``agent`` (the responder) takes a double offered in ``state``.

    Delegates to the agent's own policy if it is :class:`~bgrl.agents.base.CubeCapable`. The
    cube is offered pre-roll, so the position is still ``state.turn`` (the doubler) on roll;
    the responder's view is the perspective flip of its own evaluation. A net-less agent
    (``evaluator is None``) always takes rather than auto-lose on a drop.
    """
    if isinstance(agent, CubeCapable):
        return agent.should_take(state, cube)
    if evaluator is None:
        return True
    responder_view = flip_outcome(evaluator.evaluate_outcome(state))
    return decider.decide_take(responder_view, cube) is TakeAction.TAKE
