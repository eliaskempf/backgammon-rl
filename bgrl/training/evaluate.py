"""Common-random-numbers (CRN) win-rate evaluation.

Comparing two agents by plain self-play is noisy: dice swings and WHITE's
first-move edge dwarf small skill gaps. :func:`play_match` removes both with
**common random numbers** — each matchup is a *pair* of games sharing one dice
stream with the seats swapped between them, so the dice and the first-move
advantage cancel and (for deterministic agents) only the skill difference
remains. WP1 tracks a checkpoint's win-rate vs. :class:`~bgrl.agents.RandomAgent`
and vs. earlier checkpoints with this; WP3/WP4 reuse it for A/B comparisons.

Pass **non-learning** agents (e.g. :class:`~bgrl.agents.ValueAgent`): a
:class:`~bgrl.agents.base.LearningAgent` would train on these games via its hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgrl.agents.base import Agent
from bgrl.env import Player, RandomDiceSource
from bgrl.game import play_game


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A CRN match summarised from agent A's point of view.

    ``games`` counts every game actually played (``2 * pairs``). ``truncated`` is
    the number that hit ``max_plies`` with no winner; they are excluded from the
    win-rate denominator. ``win_rate_a`` is ``wins_a / (games - truncated)`` (0.0
    if every game truncated). ``avg_plies`` averages over all games played.
    """

    games: int
    wins_a: int
    wins_b: int
    truncated: int
    win_rate_a: float
    avg_plies: float


def play_match(
    a: Agent,
    b: Agent,
    *,
    pairs: int,
    rng: np.random.Generator,
    max_plies: int = 10_000,
) -> MatchResult:
    """Play ``pairs`` CRN game-pairs between ``a`` and ``b``; report A's win-rate.

    Each pair plays one game with ``a`` as WHITE and ``b`` as BLACK, then a second
    game with the seats swapped. Both games draw their dice from generators seeded
    **identically** (a per-pair seed taken from ``rng``), so they share the same
    dice stream regardless of how their lengths diverge — common random numbers
    that cancel dice variance and the first-move advantage. (Seed-pairing rather
    than record-then-replay precisely because swapping seats changes the
    trajectory, so the second game may need more rolls than the first produced.)
    ``rng`` must be independent of any training dice stream so evaluation never
    shifts a training run's reproducibility.
    """
    wins_a = 0
    wins_b = 0
    truncated = 0
    total_plies = 0

    for _ in range(pairs):
        pair_seed = int(rng.integers(1 << 31))
        game_a_white = play_game(
            a, b, RandomDiceSource(np.random.default_rng(pair_seed)), max_plies=max_plies
        )
        game_b_white = play_game(
            b, a, RandomDiceSource(np.random.default_rng(pair_seed)), max_plies=max_plies
        )

        for result, a_is_white in ((game_a_white, True), (game_b_white, False)):
            total_plies += result.plies
            if result.outcome is None:
                truncated += 1
                continue
            if (result.outcome.winner is Player.WHITE) == a_is_white:
                wins_a += 1
            else:
                wins_b += 1

    games = 2 * pairs
    decided = games - truncated
    return MatchResult(
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        truncated=truncated,
        win_rate_a=wins_a / decided if decided else 0.0,
        avg_plies=total_plies / games if games else 0.0,
    )
