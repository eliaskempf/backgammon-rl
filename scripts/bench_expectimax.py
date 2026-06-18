#!/usr/bin/env python
"""Strength vs. latency sweep for the WP2 expectimax agent (thin CLI).

Wraps a checkpoint's net in n-ply expectimax search at several depths and plays each
depth against ``PubevalAgent`` under common random numbers, reporting win-rate alongside
the mean wall-clock time per move. The latency column is the point of the exercise: it
shows how strength trades against move cost as depth grows, and whether a given depth is
feasible at all on this machine (gnubg ply convention: raw net = 0-ply).

Example
-------
    uv run python scripts/bench_expectimax.py --checkpoint runs/.../final.pt \
        --plies-list 0,1,2 --pairs 50
    # 3-ply is expensive; run it with pruning and a few pairs to gauge latency:
    uv run python scripts/bench_expectimax.py --checkpoint runs/.../final.pt \
        --plies-list 3 --top-k 8 --pairs 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


class TimingAgent:
    """Delegates ``act`` to an inner agent while accumulating its wall-clock cost."""

    def __init__(self, inner):
        self._inner = inner
        self.elapsed = 0.0
        self.calls = 0

    def act(self, state, dice, legal):
        start = time.perf_counter()
        move = self._inner.act(state, dice, legal)
        self.elapsed += time.perf_counter() - start
        self.calls += 1
        return move


def main() -> None:
    parser = argparse.ArgumentParser(description="WP2 expectimax win-rate vs. latency sweep.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="WP1 checkpoint (.pt)")
    parser.add_argument(
        "--plies-list", default="0,1,2", help="comma-separated search depths to sweep"
    )
    parser.add_argument("--pairs", type=int, default=50, help="CRN game-pairs per depth")
    parser.add_argument(
        "--top-k", type=int, default=None, help="candidate pruning per node (default: off)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (shared dice across depths)")
    args = parser.parse_args()

    import numpy as np

    from bgrl.agents import ExpectimaxAgent, PubevalAgent
    from bgrl.serialization import load_checkpoint, load_net
    from bgrl.training.evaluate import play_match

    plies_list = [int(p) for p in args.plies_list.split(",")]
    net = load_net(load_checkpoint(args.checkpoint))

    prune = "off" if args.top_k is None else f"top-{args.top_k}"
    print(f"checkpoint: {args.checkpoint}")
    print(f"opponent: pubeval | pairs/depth: {args.pairs} | pruning: {prune} | seed: {args.seed}")
    print(f"{'ply':>3}  {'win-rate':>9}  {'ms/move':>9}  {'moves':>7}  {'avg-plies':>9}")
    for plies in plies_list:
        agent = TimingAgent(ExpectimaxAgent(net, plies=plies, top_k=args.top_k))
        # Fresh, identically-seeded RNG per depth -> every depth faces the same dice (CRN),
        # so win-rate differences are attributable to search depth, not luck.
        res = play_match(
            agent, PubevalAgent(), pairs=args.pairs, rng=np.random.default_rng(args.seed)
        )
        ms_per_move = 1000.0 * agent.elapsed / agent.calls if agent.calls else float("nan")
        print(
            f"{plies:>3}  {res.win_rate_a:>9.3f}  {ms_per_move:>9.2f}  "
            f"{agent.calls:>7}  {res.avg_plies:>9.1f}"
        )


if __name__ == "__main__":
    main()
