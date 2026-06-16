"""Dice rolls and common-random-numbers (CRN) replay."""

import numpy as np
import pytest

from bgrl.env import RandomDiceSource, ReplayDiceSource, roll_dice


def test_roll_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = roll_dice(rng)
        assert len(d) == 2
        assert all(1 <= x <= 6 for x in d)


def test_random_source_records_history():
    src = RandomDiceSource(np.random.default_rng(1))
    rolls = [src.roll() for _ in range(10)]
    assert src.history == rolls


def test_replay_reproduces_recorded_stream():
    src = RandomDiceSource(np.random.default_rng(2))
    original = [src.roll() for _ in range(15)]
    replay = ReplayDiceSource(src.history)
    assert [replay.roll() for _ in range(15)] == original


def test_replay_exhaustion_raises_loudly():
    replay = ReplayDiceSource([(1, 2), (3, 4)])
    replay.roll()
    replay.roll()
    with pytest.raises(RuntimeError, match="exhausted"):
        replay.roll()


def test_two_sources_same_seed_agree():
    a = RandomDiceSource(np.random.default_rng(5))
    b = RandomDiceSource(np.random.default_rng(5))
    assert [a.roll() for _ in range(20)] == [b.roll() for _ in range(20)]
