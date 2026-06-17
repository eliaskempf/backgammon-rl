#!/usr/bin/env python
"""Rank sweep runs by win-rate vs the benchmark and name the best checkpoint.

Reads each run directory's ``metrics.csv`` (written by ``scripts/train.py``) and,
optionally, re-evaluates each ``best.pt`` against pubeval with more CRN pairs for a
lower-variance final ranking.

    uv run python scripts/aggregate_runs.py runs/sweep/*
    uv run python scripts/aggregate_runs.py runs/sweep/* --reeval-pairs 500
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def _best_from_metrics(run: Path) -> tuple[float, float, int] | None:
    """Return ``(best_win_rate, final_win_rate, final_games)`` from ``metrics.csv``."""
    path = run / "metrics.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    win_rates = [float(r["win_rate"]) for r in rows]
    return max(win_rates), float(rows[-1]["win_rate"]), int(rows[-1]["games"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank TD(λ) sweep runs by vs-pubeval win-rate.")
    parser.add_argument("paths", nargs="*", help="run dirs (default: runs/sweep/*)")
    parser.add_argument(
        "--reeval-pairs",
        type=int,
        default=0,
        help="re-eval each best.pt vs pubeval with this many CRN pairs (0 = use metrics.csv)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for re-eval dice")
    args = parser.parse_args()

    runs = [Path(p) for p in (args.paths or sorted(glob.glob("runs/sweep/*"))) if Path(p).is_dir()]
    if not runs:
        print("no run directories found (pass paths or populate runs/sweep/)")
        return

    reeval = None
    if args.reeval_pairs > 0:
        import numpy as np

        from bgrl.agents import PubevalAgent
        from bgrl.serialization import load_agent, load_checkpoint
        from bgrl.training.evaluate import play_match

        def reeval(best_pt: Path) -> float:
            rng = np.random.default_rng(args.seed)
            agent = load_agent(load_checkpoint(best_pt))
            return play_match(agent, PubevalAgent(), pairs=args.reeval_pairs, rng=rng).win_rate_a

    rows = []
    for run in runs:
        m = _best_from_metrics(run)
        if m is None:
            print(f"skip {run} (no metrics.csv)")
            continue
        best_wr, final_wr, games = m
        best_pt = run / "best.pt"
        rank_wr = reeval(best_pt) if (reeval and best_pt.exists()) else best_wr
        rows.append((rank_wr, best_wr, final_wr, games, run))

    if not rows:
        print("no usable runs")
        return

    rows.sort(key=lambda r: r[0], reverse=True)  # by rank_wr (Path isn't orderable on ties)
    label = f"reeval({args.reeval_pairs})" if reeval else "best(csv)"
    print(f"{'rank_wr':>10} {'best_csv':>9} {'final':>7} {'games':>9}  run ({label})")
    for rank_wr, best_wr, final_wr, games, run in rows:
        print(f"{rank_wr:>10.3f} {best_wr:>9.3f} {final_wr:>7.3f} {games:>9}  {run}")
    winner = rows[0][4]
    print(f"\nbest run: {winner}  ->  {winner / 'best.pt'}")


if __name__ == "__main__":
    main()
