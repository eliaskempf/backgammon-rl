"""Round-robin win-rate tournament between a set of agents.

Builds a pairwise win-rate matrix by running :func:`~bgrl.training.evaluate.play_match`
(common random numbers) on every unordered pair of agents. Used to compare a sweep's
trained checkpoints against each other (and against an absolute anchor such as
``pubeval``); the matrix feeds the pairwise win-rate heatmap.

Pass **non-learning** agents only (e.g. :class:`~bgrl.agents.ValueAgent`,
:class:`~bgrl.agents.PubevalAgent`) — exactly as :func:`play_match` requires, since a
:class:`~bgrl.agents.base.LearningAgent` would train on the evaluation games.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bgrl.agents.base import Agent

from .evaluate import play_match


@dataclass(frozen=True, slots=True)
class RoundRobinResult:
    """Pairwise win-rate matrix for a labelled set of agents.

    ``labels[i]`` names row/column ``i`` (insertion order of the input mapping).
    ``win_rate[i, j]`` is row ``i``'s win-rate *against* column ``j``: it is in
    ``[0, 1]``, the diagonal is exactly ``0.5`` (an agent vs. itself), and the matrix
    is exactly anti-symmetric about ``0.5`` (``win_rate[i, j] + win_rate[j, i] == 1``),
    because CRN excludes truncated games from both agents' denominators identically.
    """

    labels: tuple[str, ...]
    win_rate: np.ndarray


def round_robin(
    agents: dict[str, Agent],
    *,
    pairs: int,
    rng: np.random.Generator,
    max_plies: int = 10_000,
) -> RoundRobinResult:
    """Play every unordered pair of ``agents`` and return the win-rate matrix.

    Each matchup is ``pairs`` CRN game-pairs via :func:`play_match`; only the upper
    triangle is played (``2 * pairs`` games per pair) and the lower triangle is its
    exact complement. ``rng`` is consumed sequentially across matchups in label order,
    so the result is reproducible for a given ``agents`` ordering and seed.
    """
    labels = tuple(agents)
    members = [agents[label] for label in labels]
    n = len(members)
    win_rate = np.full((n, n), 0.5, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            wr = play_match(members[i], members[j], pairs=pairs, rng=rng, max_plies=max_plies)
            win_rate[i, j] = wr.win_rate_a
            win_rate[j, i] = 1.0 - wr.win_rate_a

    return RoundRobinResult(labels=labels, win_rate=win_rate)
