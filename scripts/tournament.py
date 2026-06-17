#!/usr/bin/env python
"""Round-robin CRN tournament over sweep checkpoints -> win-rate matrix CSV (thin CLI).

Loads one checkpoint per run directory, plays every pair against each other with common
random numbers (:func:`bgrl.training.round_robin`), and writes a square win-rate matrix
to CSV. ``scripts/plot_heatmap.py`` renders that CSV as the pairwise heatmap.

Each run is labelled from its checkpoint metadata (``lr``/``lam``/``hidden``); rows are
ordered by ``(hidden, lr, lam)`` so capacity/step-size blocks read cleanly in the heatmap.
An absolute anchor (``pubeval`` by default) is appended as the final row/column.

    uv run python scripts/tournament.py runs/sweep/* --pairs 200 --anchor pubeval \
        --out runs/sweep/winrate_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin CRN win-rate tournament.")
    parser.add_argument("paths", nargs="*", help="run dirs (default: runs/sweep/*)")
    parser.add_argument(
        "--checkpoint", default="best.pt", help="checkpoint file within each run dir"
    )
    parser.add_argument("--pairs", type=int, default=200, help="CRN game-pairs per matchup")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the tournament dice")
    parser.add_argument(
        "--anchor",
        choices=("pubeval", "random", "none"),
        default="pubeval",
        help="absolute-reference opponent appended as the final row/col",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/sweep/winrate_matrix.csv"), help="output CSV"
    )
    args = parser.parse_args()

    import numpy as np

    from bgrl.agents import PubevalAgent, RandomAgent
    from bgrl.agents.base import Agent
    from bgrl.serialization import load_agent, load_checkpoint
    from bgrl.training import round_robin

    run_dirs = [
        Path(p)
        for p in (args.paths or sorted(glob.glob("runs/sweep/*")))
        if (Path(p) / args.checkpoint).exists()
    ]
    if not run_dirs:
        print(f"no run dirs with a {args.checkpoint} found (pass paths or populate runs/sweep/)")
        return

    # Load each checkpoint and derive a (hidden, lr, lam)-ordered label.
    loaded: list[tuple[int, float, float, str, Path, Agent]] = []
    for run in run_dirs:
        ckpt = load_checkpoint(run / args.checkpoint)
        meta = ckpt["metadata"]
        lr, lam, hidden = float(meta["lr"]), float(meta["lam"]), int(meta["hidden"])
        label = f"lr{lr:g}_l{lam:g}_h{hidden}"
        loaded.append((hidden, lr, lam, label, run, load_agent(ckpt)))
    loaded.sort(key=lambda r: (r[0], r[1], r[2]))

    # Disambiguate any colliding labels by suffixing the run-dir name.
    label_counts: dict[str, int] = {}
    for _, _, _, label, _, _ in loaded:
        label_counts[label] = label_counts.get(label, 0) + 1
    agents: dict[str, Agent] = {}
    for _, _, _, label, run, agent in loaded:
        key = label if label_counts[label] == 1 else f"{label}@{run.name}"
        agents[key] = agent

    if args.anchor == "pubeval":
        agents["pubeval"] = PubevalAgent()
    elif args.anchor == "random":
        agents["random"] = RandomAgent(np.random.default_rng(args.seed))

    print(f"tournament: {len(agents)} agents, {args.pairs} CRN pairs/matchup")
    res = round_robin(agents, pairs=args.pairs, rng=np.random.default_rng(args.seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *res.labels])  # corner cell blank; column headers
        for label, row in zip(res.labels, res.win_rate, strict=True):
            writer.writerow([label, *(f"{v:.4f}" for v in row)])
    print(f"wrote {res.win_rate.shape[0]}x{res.win_rate.shape[1]} win-rate matrix -> {args.out}")


if __name__ == "__main__":
    main()
