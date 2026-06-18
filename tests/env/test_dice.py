"""Dice rolls and common-random-numbers (CRN) replay."""

import numpy as np
import pytest

from bgrl.env import ManualDiceSource, RandomDiceSource, ReplayDiceSource, roll_dice


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


def test_manual_source_pops_in_fifo_order():
    src = ManualDiceSource()
    src.push((1, 2))
    src.push((6, 6))
    assert src.roll() == (1, 2)
    assert src.roll() == (6, 6)


def test_manual_source_raises_when_empty():
    src = ManualDiceSource()
    with pytest.raises(RuntimeError, match="empty"):
        src.roll()


def test_manual_source_rejects_out_of_range_push():
    src = ManualDiceSource()
    for bad in [(0, 3), (3, 7), (-1, 1)]:
        with pytest.raises(ValueError, match="must each be"):
            src.push(bad)


def test_manual_source_coerces_to_int_tuple():
    src = ManualDiceSource()
    src.push((np.int64(2), np.int64(5)))
    d = src.roll()
    assert d == (2, 5)
    assert all(type(x) is int for x in d)
