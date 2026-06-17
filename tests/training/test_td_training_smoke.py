"""End-to-end TD(λ) training smoke test — self-activates once the core is filled.

With the TD core implemented, self-play training must beat ``RandomAgent``
decisively and reproducibly. The ``_td_core_ready`` gate (kept as a guard) skips the
test only while ``TDLambda.step`` is a hollow ``NotImplementedError`` stub. Marked
``slow`` (it trains ~3000 games); run with ``-m slow`` to include it.
"""

import numpy as np
import pytest

from bgrl.agents import RandomAgent, ValueAgent
from bgrl.agents.td_agent import TDAgent
from bgrl.env import Env, legal_moves
from bgrl.nets.value_net import MLPValueNet
from bgrl.training.evaluate import play_match
from bgrl.training.loop import train
from bgrl.training.td_lambda import TDLambda


def _td_core_ready() -> bool:
    """False while ``TDLambda.step`` is hollow (raises ``NotImplementedError``)."""
    trainer = TDLambda(MLPValueNet(hidden=8), lam=0.7, gamma=1.0, lr=0.1)
    state = Env.initial_state()
    dice = (3, 1)
    move, afterstate = legal_moves(state, dice)[0]
    try:
        trainer.step(state, dice, move, afterstate)
    except NotImplementedError:
        return False
    except Exception:
        return True  # implemented but raised for another reason — let the test surface it
    return True


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _td_core_ready(), reason="TD(λ) core not implemented yet (WP1 Phase B)"),
]


def _final_win_rate_vs_random(seed: int) -> float:
    import torch

    torch.manual_seed(seed)  # reproducible net initialisation
    train_rng, eval_rng = np.random.default_rng(seed).spawn(2)
    net = MLPValueNet(hidden=40)
    agent = TDAgent(net, lam=0.7, lr=0.1)
    train(agent, games=3000, rng=train_rng)
    # Evaluate the trained weights with a fresh non-learning ValueAgent.
    return play_match(ValueAgent(net), RandomAgent(eval_rng), pairs=100, rng=eval_rng).win_rate_a


def test_training_beats_random_and_is_reproducible():
    wr1 = _final_win_rate_vs_random(seed=0)
    assert wr1 > 0.9, f"win-rate vs random {wr1:.3f} should exceed 0.9 after training"
    assert _final_win_rate_vs_random(seed=0) == wr1  # same seed -> identical result
