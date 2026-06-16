"""TDAgent: protocol membership + inherited greedy selection (no TD core needed)."""

from bgrl.agents import Agent, LearningAgent
from bgrl.agents.td_agent import TDAgent
from bgrl.env import Env, legal_moves
from bgrl.nets.value_net import MLPValueNet


def test_td_agent_is_a_learning_agent():
    agent = TDAgent(MLPValueNet(hidden=8), lam=0.7, lr=0.1)
    assert isinstance(agent, Agent)
    assert isinstance(agent, LearningAgent)


def test_td_agent_selects_a_legal_move_greedily():
    # act() is inherited from ValueAgent and goes through net.evaluate (no
    # gradients, no TD core), so this exercises selection without touching the
    # hollow trainer — the only place a real TDAgent meets play_game is the
    # (skip-gated) smoke test.
    agent = TDAgent(MLPValueNet(hidden=8), lam=0.7, lr=0.1)
    state = Env.initial_state()
    dice = (3, 1)
    legal = legal_moves(state, dice)
    move = agent.act(state, dice, legal)
    assert move in [m for m, _ in legal]
