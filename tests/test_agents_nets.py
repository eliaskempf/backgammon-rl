"""Smoke tests for the reference agent and the value net."""

import numpy as np
import torch
from bgrl.agents import RandomAgent
from bgrl.env import Env, encode
from bgrl.nets import OUTCOME_DIM, MLPValueNet


def test_random_agent_is_deterministic_per_seed():
    s = Env.initial_state()
    legal = Env.legal_moves(s, (3, 1))
    a1 = RandomAgent(np.random.default_rng(42))
    a2 = RandomAgent(np.random.default_rng(42))
    picks1 = [a1.act(s, (3, 1), legal) for _ in range(20)]
    picks2 = [a2.act(s, (3, 1), legal) for _ in range(20)]
    assert picks1 == picks2
    assert all(p in {m for m, _a in legal} for p in picks1)


def test_value_net_batched_eval():
    torch.manual_seed(0)
    net = MLPValueNet(hidden=32)
    s = Env.initial_state()
    feats = np.stack([encode(s, s.turn) for _ in range(8)])
    out = net.evaluate(feats)
    assert out.shape == (8, OUTCOME_DIM)
    assert out.dtype == np.float32
    assert np.all(out > 0.0) and np.all(out < 1.0)  # sigmoid outputs
