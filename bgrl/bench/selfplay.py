"""Random self-play and multiprocessing throughput measurement.

``play_random_game`` is pure-Python move-gen only (no net), so workers are true
parallel processes with no GIL contention and nothing to oversubscribe. Workers
synchronise on a barrier before timing so process-spawn cost is excluded, and the
aggregate rate divides summed work by the *max* worker wall-time (the real
overlap), not the sum.
"""

from __future__ import annotations

import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from bgrl.env import Env, EnvState

# A backgammon game is a few hundred plies; this only guards against a pathological
# non-terminating loop, never a real game.
_MAX_PLIES = 20000


def play_random_game(rng: np.random.Generator, max_plies: int = _MAX_PLIES) -> tuple[int, int, int]:
    """Play one uniform-random game. Returns ``(legal_moves_calls, afterstates, plies)``."""
    s = Env.initial_state()
    if int(rng.integers(2)) == 1:  # randomise the starting player
        s = EnvState(board=s.board, bar=s.bar, off=s.off, turn=s.turn.opponent())

    calls = 0
    afterstates = 0
    plies = 0
    while plies < max_plies and not Env.is_terminal(s):
        dice = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
        legal = Env.legal_moves(s, dice)
        calls += 1
        afterstates += len(legal)
        if legal:
            s = legal[int(rng.integers(len(legal)))][1]
        else:  # no legal move: the turn passes
            s = EnvState(board=s.board, bar=s.bar, off=s.off, turn=s.turn.opponent())
        plies += 1
    return calls, afterstates, plies


def _worker(n_games: int, seed: int, barrier: Any, q: Any) -> None:
    rng = np.random.default_rng(seed)
    play_random_game(rng)  # one warmup game, not timed
    barrier.wait()
    t0 = time.perf_counter()
    games = calls = afterstates = plies = 0
    for _ in range(n_games):
        c, a, p = play_random_game(rng)
        games += 1
        calls += c
        afterstates += a
        plies += p
    q.put((games, calls, afterstates, plies, time.perf_counter() - t0))


def run_selfplay(n_games: int, workers: int, seed: int) -> dict:
    """Run ``n_games`` random games across ``workers`` spawn processes.

    Returns summed counters plus the max worker wall-time (the overlap window).
    """
    workers = max(1, workers)
    per = [n_games // workers] * workers
    for i in range(n_games % workers):
        per[i] += 1
    per = [x for x in per if x > 0]
    workers = len(per)

    ctx = get_context("spawn")
    barrier = ctx.Barrier(workers)
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(per[i], seed + i, barrier, q)) for i in range(workers)
    ]
    for p in procs:
        p.start()
    results = [q.get() for _ in range(workers)]
    for p in procs:
        p.join()

    return {
        "games": sum(r[0] for r in results),
        "calls": sum(r[1] for r in results),
        "afterstates": sum(r[2] for r in results),
        "plies": sum(r[3] for r in results),
        "elapsed": max(r[4] for r in results),
        "workers": workers,
    }
