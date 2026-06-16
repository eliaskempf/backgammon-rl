"""MLPValueNet conforms to the ValueNet protocol + the fixed output shape/arch."""

import numpy as np
import torch

from bgrl.env import N_FEATURES, Env, encode
from bgrl.nets import OUTCOME_DIM, MLPValueNet, ValueNet


def test_mlp_is_valuenet():
    assert isinstance(MLPValueNet(), ValueNet)


def test_evaluate_shape_and_dtype():
    torch.manual_seed(0)
    net = MLPValueNet(hidden=16)
    s = Env.initial_state()
    feats = np.stack([encode(s, s.turn) for _ in range(5)])
    out = net.evaluate(feats)
    assert out.shape == (5, OUTCOME_DIM)
    assert out.dtype == np.float32


def test_arch_config_roundtrip():
    net = MLPValueNet(hidden=48)
    cfg = net.arch_config()
    assert cfg == {
        "class": "MLPValueNet",
        "hidden": 48,
        "n_features": N_FEATURES,
        "outcome_dim": OUTCOME_DIM,
    }
    assert MLPValueNet.from_config(cfg).arch_config() == cfg
