"""The play_game driver: termination, CRN determinism, lifecycle-hook firing."""

import numpy as np

from bgrl.agents import RandomAgent
from bgrl.env import Outcome, RandomDiceSource, ReplayDiceSource
from bgrl.game import play_game


def test_random_game_terminates_with_outcome():
    res = play_game(
        RandomAgent(np.random.default_rng(1)),
        RandomAgent(np.random.default_rng(2)),
        RandomDiceSource(np.random.default_rng(0)),
        record=True,
    )
    assert isinstance(res.outcome, Outcome)
    assert res.plies > 0
    assert res.plies == len(res.steps)


def test_crn_replay_is_bit_identical():
    ds = RandomDiceSource(np.random.default_rng(0))
    r1 = play_game(
        RandomAgent(np.random.default_rng(1)),
        RandomAgent(np.random.default_rng(2)),
        ds,
        record=True,
    )
    r2 = play_game(
        RandomAgent(np.random.default_rng(1)),
        RandomAgent(np.random.default_rng(2)),
        ReplayDiceSource(ds.history),
        record=True,
    )
    assert r1.outcome == r2.outcome
    assert r1.plies == r2.plies
    assert r1.steps == r2.steps


class _Spy:
    """A LearningAgent that just counts hook calls and plays randomly."""

    def __init__(self):
        self.steps = 0
        self.ends = 0
        self._rng = np.random.default_rng(9)

    def act(self, state, dice, legal):
        return legal[int(self._rng.integers(len(legal)))][0]

    def observe_step(self, state, dice, move, afterstate):
        self.steps += 1

    def observe_game_end(self, outcome):
        self.ends += 1


def test_self_play_fires_step_per_ply_and_one_game_end():
    spy = _Spy()  # one object in both seats == TD-Gammon self-play
    res = play_game(spy, spy, RandomDiceSource(np.random.default_rng(3)))
    assert spy.steps == res.plies
    assert spy.ends == 1


def test_two_distinct_learners_split_plies_and_each_get_game_end():
    a, b = _Spy(), _Spy()
    res = play_game(a, b, RandomDiceSource(np.random.default_rng(4)))
    assert a.ends == 1 and b.ends == 1
    assert a.steps + b.steps == res.plies  # each observes only its own plies
