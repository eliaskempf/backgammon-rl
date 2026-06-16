"""Checkpoint round-trip, version guards, and the generic load_agent factory."""

import numpy as np
import pytest
import torch

from bgrl.agents import Agent, RandomAgent
from bgrl.env import ENCODING_VERSION, Outcome, RandomDiceSource
from bgrl.game import play_game
from bgrl.nets import MLPValueNet
from bgrl.serialization import (
    CHECKPOINT_FORMAT_VERSION,
    load_agent,
    load_checkpoint,
    load_net,
    save_checkpoint,
)


def test_roundtrip_preserves_weights_arch_and_metadata(tmp_path):
    torch.manual_seed(0)
    net = MLPValueNet(hidden=32)
    path = tmp_path / "nested" / "ckpt.pt"  # nested => exercises parent mkdir
    save_checkpoint(net, path, trained_with="random", metadata={"games_trained": 7})

    ck = load_checkpoint(path)
    assert ck["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert ck["encoding_version"] == ENCODING_VERSION
    assert ck["trained_with"] == "random"
    assert ck["metadata"]["games_trained"] == 7
    assert "created_at" in ck["metadata"] and "git_sha" in ck["metadata"]

    net2 = load_net(ck)
    assert net2.arch_config() == net.arch_config()
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), net2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_load_agent_plays_full_game(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(MLPValueNet(hidden=16), path, trained_with="random")
    agent = load_agent(load_checkpoint(path))
    assert isinstance(agent, Agent)
    res = play_game(
        agent,
        RandomAgent(np.random.default_rng(1)),
        RandomDiceSource(np.random.default_rng(2)),
    )
    assert isinstance(res.outcome, Outcome)


def test_format_version_mismatch_raises(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(MLPValueNet(hidden=8), path, trained_with="random")
    ck = torch.load(path, weights_only=False)
    ck["format_version"] = 999
    torch.save(ck, path)
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


def test_encoding_version_mismatch_raises(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(MLPValueNet(hidden=8), path, trained_with="random")
    ck = torch.load(path, weights_only=False)
    ck["encoding_version"] = ENCODING_VERSION + 1
    torch.save(ck, path)
    with pytest.raises(ValueError, match="encoding_version"):
        load_checkpoint(path)


def test_unknown_net_class_raises_clearly(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(MLPValueNet(hidden=8), path, trained_with="random")
    ck = load_checkpoint(path)
    ck["net_arch"] = {**ck["net_arch"], "class": "NopeNet"}
    with pytest.raises(ValueError, match="unknown net class"):
        load_net(ck)
