"""The algorithm-agnostic training loop: reusability + reproducibility (Phase A)."""

import numpy as np

from bgrl.agents import RandomAgent
from bgrl.training.loop import train


class _NoOpLearner:
    """A trivial second 'trainer': a LearningAgent whose hooks do nothing.

    Demonstrates the reuse contract — a different learner drops into ``train()``
    without touching loop.py. Plays randomly; learns nothing.
    """

    def __init__(self, rng):
        self._rng = rng
        self.steps = 0
        self.ends = 0

    def act(self, state, dice, legal):
        return legal[int(self._rng.integers(len(legal)))][0]

    def observe_step(self, state, dice, move, afterstate):
        self.steps += 1

    def observe_game_end(self, outcome):
        self.ends += 1


class _DiceSpy:
    """Records the dice of every ply (via observe_step, so forced passes count)."""

    def __init__(self, rng):
        self._rng = rng
        self.dice = []

    def act(self, state, dice, legal):
        return legal[int(self._rng.integers(len(legal)))][0]

    def observe_step(self, state, dice, move, afterstate):
        self.dice.append(dice)

    def observe_game_end(self, outcome):
        pass


def test_loop_drives_learning_hooks_once_per_game_and_ply():
    learner = _NoOpLearner(np.random.default_rng(0))
    seen = []
    train(
        learner, games=3, rng=np.random.default_rng(1), on_game_end=lambda n, r: seen.append((n, r))
    )
    assert [n for n, _ in seen] == [1, 2, 3]  # 1-based game index
    assert learner.ends == 3
    assert learner.steps == sum(r.plies for _, r in seen)


def test_loop_runs_non_learning_agent_untouched():
    # A plain Agent (not a LearningAgent) passes straight through, no hook errors.
    completed = []
    train(
        RandomAgent(np.random.default_rng(1)),
        games=2,
        rng=np.random.default_rng(2),
        on_game_end=lambda n, r: completed.append(n),
    )
    assert completed == [1, 2]


def test_same_seed_reproduces_the_run():
    def run():
        learner = _NoOpLearner(np.random.default_rng(0))
        trace = []
        train(
            learner,
            games=4,
            rng=np.random.default_rng(123),
            on_game_end=lambda n, r: trace.append((r.outcome, r.plies)),
        )
        return trace

    assert run() == run()


def test_eval_stream_does_not_perturb_training_dice():
    # Splitting one seed into independent (train, eval) streams means eval draws
    # never shift the training dice — the property scripts/train.py relies on.
    def training_dice(consume_eval):
        train_rng, eval_rng = np.random.default_rng(0).spawn(2)
        spy = _DiceSpy(np.random.default_rng(5))

        def cb(n, r):
            if consume_eval:
                eval_rng.integers(1, 7, size=50)  # stand-in for an eval pass

        train(spy, games=3, rng=train_rng, on_game_end=cb)
        return spy.dice

    quiet = training_dice(False)
    assert len(quiet) > 0
    assert quiet == training_dice(True)
