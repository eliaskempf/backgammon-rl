"""The algorithm-agnostic self-play training loop.

WP1's training driver. It generates self-play games and lets a learning agent
learn from them **only** through the agent lifecycle hooks (``observe_step`` /
``observe_game_end`` fired by :func:`bgrl.game.play_game`), so the loop itself
knows nothing about the update rule. TD(λ) (WP1) plugs in by being a
:class:`~bgrl.agents.base.LearningAgent`; a future DQN-afterstate or any other
online learner plugs in the same way, and a non-learning agent (random, a frozen
checkpoint) runs through untouched. That is the reuse contract: a different
"trainer" swaps in without editing this module.

The unit of iteration is an *episode* — one cubeless single game in v1 — but the
loop never assumes that, so match-to-N-points can slot in later (CLAUDE.md §5).
Self-play means the same agent object occupies both seats (TD-Gammon style). Each
game draws its dice from a fresh :class:`~bgrl.env.RandomDiceSource` over a single
injected ``Generator``, so one seed reproduces the whole training curve.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bgrl.agents.base import Agent
from bgrl.env import RandomDiceSource
from bgrl.game import GameResult, play_game

GameEndCallback = Callable[[int, GameResult], None]
"""Called after each game with ``(n_completed, result)``; ``n_completed`` is 1-based."""


def train(
    agent: Agent,
    *,
    games: int,
    rng: np.random.Generator,
    on_game_end: GameEndCallback | None = None,
    max_plies: int = 10_000,
) -> None:
    """Run ``games`` self-play episodes, driving the agent's learning hooks.

    ``agent`` plays both seats. If it is a
    :class:`~bgrl.agents.base.LearningAgent`, it learns online via the
    ``observe_*`` hooks :func:`~bgrl.game.play_game` fires; otherwise the games are
    just generated. Nothing here references TD or any update rule.

    Every game pulls its dice from a fresh :class:`~bgrl.env.RandomDiceSource` over
    the single ``rng``, so a fixed seed reproduces the entire run (note: a torch
    seed is also needed for reproducible *net* initialisation — the script handles
    that). After each game ``on_game_end(n_completed, result)`` is called with
    ``n_completed`` the 1-based number of games finished (so
    ``metadata={"games_trained": n_completed}`` is correct); ``result.outcome is
    None`` when the game hit ``max_plies`` without terminating — surfaced here so a
    caller can react rather than silently miscount.
    """
    for n in range(1, games + 1):
        dice = RandomDiceSource(rng)
        result = play_game(agent, agent, dice, max_plies=max_plies)
        if on_game_end is not None:
            on_game_end(n, result)
