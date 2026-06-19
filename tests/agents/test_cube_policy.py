"""Pre-roll evaluator (Adapter A) and cube-decision dispatch glue (WP6 B3)."""

from __future__ import annotations

import numpy as np

from bgrl.agents import (
    CubeCapable,
    NetEvaluator,
    RandomAgent,
    ValueAgent,
    evaluator_for,
    wants_to_double,
    wants_to_take,
)
from bgrl.env import Env, Move, Player
from bgrl.nets.base import OUTCOME_DIM
from bgrl.nets.cube import CubeDecider
from bgrl.nets.equity import CubeContext

_DECIDER = CubeDecider()


class _ConstNet:
    """A ValueNet stub returning a fixed mover-POV vector for every afterstate."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = np.asarray(vec, dtype=np.float64)

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        batch = np.asarray(features).shape[:-1]
        return np.broadcast_to(self._vec, (*batch, OUTCOME_DIM)).copy()


def test_netevaluator_flips_to_on_roll_pov() -> None:
    # The net reports the just-moved opponent's p_win=0.6 everywhere; the on-roll mover's
    # pre-roll outcome is the flip (p_win=0.4), constant across all 21 rolls (no terminals).
    evaluator = NetEvaluator(_ConstNet([0.6, 0.0, 0.0, 0.0, 0.0]))
    out = evaluator.evaluate_outcome(Env.initial_state())
    assert out.shape == (OUTCOME_DIM,)
    assert np.allclose(out, [0.4, 0.0, 0.0, 0.0, 0.0])


def test_evaluator_for_net_agent_and_netless_agent() -> None:
    net = _ConstNet([0.5, 0.0, 0.0, 0.0, 0.0])
    assert isinstance(evaluator_for(ValueAgent(net)), NetEvaluator)
    assert evaluator_for(RandomAgent(np.random.default_rng(0))) is None  # net-less -> None


def test_existing_agents_are_not_cube_capable_but_still_agents() -> None:
    net = _ConstNet([0.5, 0.0, 0.0, 0.0, 0.0])
    assert not isinstance(ValueAgent(net), CubeCapable)
    assert not isinstance(RandomAgent(np.random.default_rng(0)), CubeCapable)


def test_netless_agent_never_doubles_always_takes() -> None:
    agent = RandomAgent(np.random.default_rng(0))
    state = Env.initial_state()
    cube = CubeContext()
    assert wants_to_double(agent, state, cube, evaluator=None, decider=_DECIDER) is False
    assert wants_to_take(agent, state, cube, evaluator=None, decider=_DECIDER) is True


def test_cannot_double_when_opponent_owns_cube() -> None:
    # Even with a strong evaluation, an agent may not double a cube the opponent owns.
    net = _ConstNet([0.1, 0.0, 0.0, 0.0, 0.0])  # opp p_win 0.1 -> mover crushing
    state = Env.initial_state()  # WHITE on roll
    opp_owned = CubeContext(value=2, owner=Player.BLACK)
    evaluator = NetEvaluator(net)
    doubles = wants_to_double(
        ValueAgent(net), state, opp_owned, evaluator=evaluator, decider=_DECIDER
    )
    assert doubles is False


class _CubeBot:
    """A CubeCapable stub with a fixed, deliberately contrarian cube policy."""

    def act(self, state, dice, legal: list[tuple[Move, Env]]) -> Move:
        return legal[0][0]

    def should_double(self, state, cube: CubeContext) -> bool:
        return True

    def should_take(self, state, cube: CubeContext) -> bool:
        return False


def test_cube_capable_agent_overrides_default_policy() -> None:
    bot = _CubeBot()
    assert isinstance(bot, CubeCapable)
    state = Env.initial_state()
    cube = CubeContext()
    # The dispatchers must defer to the agent's own policy, ignoring evaluator/decider.
    assert wants_to_double(bot, state, cube, evaluator=None, decider=_DECIDER) is True
    assert wants_to_take(bot, state, cube, evaluator=None, decider=_DECIDER) is False
