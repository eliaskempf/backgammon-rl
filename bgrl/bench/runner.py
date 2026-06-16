"""Benchmark orchestration: env throughput, encode rate, net sweep, eval share.

All rates are wall-clock. The net sweep runs single-threaded (``torch`` threads =
1) because that mirrors the per-worker cost in a CPU multiprocessing self-play
pipeline — the realistic comparison for the WP0 decision.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from datetime import UTC, datetime

import numpy as np

from bgrl.env import Env, EnvState, encode
from bgrl.env.encoding import N_FEATURES
from bgrl.nets import MLPValueNet

from .fingerprint import machine_fingerprint
from .schema import build_result
from .selfplay import play_random_game, run_selfplay

NET_BATCH_SIZES = (1, 8, 16, 32, 64)


def run_env_bench(games: int, workers: int, seed: int) -> dict:
    """Aggregate self-play throughput plus a single-worker baseline for scaling."""
    multi = run_selfplay(games, workers, seed)
    single_games = min(games, max(50, games // max(workers, 1)))
    single = run_selfplay(single_games, 1, seed + 10_000)

    gps = multi["games"] / multi["elapsed"]
    gps1 = single["games"] / single["elapsed"]
    return {
        "games_per_sec": round(gps, 2),
        "legal_moves_calls_per_sec": round(multi["calls"] / multi["elapsed"], 1),
        "afterstates_per_sec": round(multi["afterstates"] / multi["elapsed"], 1),
        "mean_afterstates_per_position": round(multi["afterstates"] / multi["calls"], 3),
        "mean_positions_per_game": round(multi["calls"] / multi["games"], 2),
        "workers": multi["workers"],
        "games_per_sec_single_worker": round(gps1, 2),
        "positions_per_sec_single_worker": round(single["calls"] / single["elapsed"], 1),
        "scaling_efficiency": round(gps / (gps1 * multi["workers"]), 3) if gps1 > 0 else None,
    }


def _collect_states(rng: np.random.Generator, n: int) -> list[EnvState]:
    states: list[EnvState] = []
    s = Env.initial_state()
    plies = 0
    while len(states) < n:
        if Env.is_terminal(s) or plies > 400:
            s = Env.initial_state()
            plies = 0
        states.append(s)
        dice = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
        legal = Env.legal_moves(s, dice)
        if legal:
            s = legal[int(rng.integers(len(legal)))][1]
        else:
            s = EnvState(board=s.board, bar=s.bar, off=s.off, turn=s.turn.opponent())
        plies += 1
    return states


def run_encode_bench(n_positions: int, seed: int) -> float:
    """Throughput of ``encode`` over sampled positions (positions/sec)."""
    rng = np.random.default_rng(seed)
    states = _collect_states(rng, n_positions)
    for s in states[:50]:  # warmup
        encode(s, s.turn)
    t0 = time.perf_counter()
    for s in states:
        encode(s, s.turn)
    elapsed = time.perf_counter() - t0
    return round(len(states) / elapsed, 1)


def run_net_bench(hidden: int, batch_sizes: tuple[int, ...], iters: int, warmup: int) -> dict:
    """Single-threaded CPU forward-pass throughput at realistic batch sizes."""
    import torch

    torch.set_num_threads(1)
    net = MLPValueNet(hidden=hidden)
    net.eval()
    by_batch = {}
    with torch.inference_mode():
        for b in batch_sizes:
            x = torch.rand(b, N_FEATURES, dtype=torch.float32)
            for _ in range(warmup):
                net(x)
            t0 = time.perf_counter()
            for _ in range(iters):
                net(x)
            elapsed = time.perf_counter() - t0
            by_batch[str(b)] = {
                "pos_per_sec": round(iters * b / elapsed, 1),
                "ms_per_batch": round(1000 * elapsed / iters, 4),
            }
    return {"threads": 1, "by_batch": by_batch}


def run_profile(n_games: int, seed: int, top: int = 20) -> list[dict]:
    """cProfile a single-worker self-play run; return the top cumulative-time funcs."""
    rng = np.random.default_rng(seed)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_games):
        play_random_game(rng)
    pr.disable()
    stats = pstats.Stats(pr)
    rows = []
    for (filename, lineno, func), (_cc, nc, _tt, ct, _callers) in sorted(
        stats.stats.items(), key=lambda kv: kv[1][3], reverse=True
    )[:top]:
        short = filename.rsplit("/", 1)[-1]
        rows.append({"func": f"{short}:{lineno}:{func}", "cumtime_s": round(ct, 4), "ncalls": nc})
    return rows


def _net_eval_share(env: dict, net: dict) -> dict:
    """Estimate the per-position split between move-gen, encode, and net eval.

    Models a 0-ply value-agent step: enumerate B afterstates, encode them, run one
    batched forward pass of size ~B. Uses single-worker per-position move-gen cost
    so it reflects one self-play worker.
    """
    b = env["mean_afterstates_per_position"]
    by_batch = net["by_batch"]
    nearest = min(by_batch, key=lambda k: abs(int(k) - b))
    net_pps = by_batch[nearest]["pos_per_sec"]

    eval_t = b / net_pps
    movegen_t = 1.0 / env["positions_per_sec_single_worker"]
    encode_t = b / env["encode_per_sec"]
    total = eval_t + movegen_t + encode_t
    return {
        "branching_factor_B": round(b, 2),
        "nearest_batch": int(nearest),
        "movegen_t_us": round(movegen_t * 1e6, 2),
        "encode_t_us": round(encode_t * 1e6, 2),
        "net_eval_t_us": round(eval_t * 1e6, 2),
        "movegen_fraction": round(movegen_t / total, 3),
        "encode_fraction": round(encode_t / total, 3),
        "net_eval_fraction": round(eval_t / total, 3),
    }


def run_all(
    *,
    games: int,
    seed: int,
    workers: int,
    bench_net: bool,
    net_hidden: int,
    net_iters: int,
    net_warmup: int,
    profile: bool,
    tag: str,
) -> dict:
    """Run the full benchmark and return a schema result dict."""
    config = {
        "games": games,
        "seed": seed,
        "workers": workers,
        "net_hidden": net_hidden,
        "net_iters": net_iters,
    }
    fingerprint = machine_fingerprint()
    env = run_env_bench(games, workers, seed)
    env["encode_per_sec"] = run_encode_bench(2000, seed + 7)
    env["profile_top"] = run_profile(min(games, 200), seed + 3) if profile else None

    net = None
    if bench_net:
        net = run_net_bench(net_hidden, NET_BATCH_SIZES, net_iters, net_warmup)
        env["net_eval_share"] = _net_eval_share(env, net)
    else:
        env["net_eval_share"] = None

    return build_result(
        tag=tag,
        timestamp=datetime.now(UTC).isoformat(),
        fingerprint=fingerprint,
        config=config,
        env=env,
        net=net,
    )
