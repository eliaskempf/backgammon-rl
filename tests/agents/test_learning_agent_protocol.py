"""Agent vs LearningAgent protocol membership (the parallel-fan-out seam)."""

import numpy as np

from bgrl.agents import Agent, LearningAgent, RandomAgent, ValueAgent


class _Learner:
    def act(self, state, dice, legal):
        return legal[0][0]

    def observe_step(self, state, dice, move, afterstate):
        pass

    def observe_game_end(self, outcome):
        pass


def test_random_agent_is_agent_but_not_learning():
    a = RandomAgent(np.random.default_rng(0))
    assert isinstance(a, Agent)
    assert not isinstance(a, LearningAgent)


def test_value_agent_is_agent_but_not_learning():
    a = ValueAgent(object())  # net unused for a membership check
    assert isinstance(a, Agent)
    assert not isinstance(a, LearningAgent)


def test_learner_satisfies_both_protocols():
    a = _Learner()
    assert isinstance(a, Agent)
    assert isinstance(a, LearningAgent)
