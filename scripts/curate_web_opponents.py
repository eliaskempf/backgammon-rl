#!/usr/bin/env python
"""Curate a small difficulty ladder of play-server opponents from a sweep run (thin CLI).

The browser play server (``scripts/play_web.py``) offers any ``*.pt`` in its
``--checkpoints-dir`` as an opponent, listing each by ``path.stem`` plus metadata. The
sweep, however, nests dozens of snapshots under ``runs/sweep/<run>/`` — too many, and not
flat. This script copies a few chosen snapshots of *one* run into a flat, committed
directory under friendly names, re-stamping each with the eval ``win_rate`` read from the
run's ``metrics.csv`` (snapshots don't record it) so the picker shows why a rung is harder.

Rungs are ordered by training volume (more games -> at least as skilled); the per-snapshot
win-rate vs ``pubeval`` is noisy and saturates, so it's shown for information, not ordering.

    uv run python scripts/curate_web_opponents.py \
        --run runs/sweep/lr0.1_lam0.7_h64_s0 --out bgrl/web/checkpoints
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bgrl.serialization import load_checkpoint, load_net, save_checkpoint

# (output stem, snapshot game-count, human-facing difficulty label). The default run is the
# one whose summed pubeval win-rate across these three points is highest (see plans/).
DEFAULT_RUNGS: list[tuple[str, int, str]] = [
    ("td-easy", 50_000, "Easy"),
    ("td-medium", 300_000, "Medium"),
    ("td-hard", 1_000_000, "Hard"),
]


def _win_rate_at(metrics_csv: Path, games: int) -> float | None:
    """The recorded eval win rate at exactly ``games`` games, or ``None`` if not evaluated."""
    if not metrics_csv.is_file():
        return None
    with metrics_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["games"]) == games:
                return float(row["win_rate"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate web-player opponents from a sweep run.")
    parser.add_argument("--run", type=Path, default=Path("runs/sweep/lr0.1_lam0.7_h64_s0"))
    parser.add_argument("--out", type=Path, default=Path("bgrl/web/checkpoints"))
    args = parser.parse_args()

    metrics_csv = args.run / "metrics.csv"
    args.out.mkdir(parents=True, exist_ok=True)

    for stem, games, label in DEFAULT_RUNGS:
        src = args.run / f"td_{games:07d}.pt"
        ckpt = load_checkpoint(src)
        net = load_net(ckpt)
        win_rate = _win_rate_at(metrics_csv, games)

        # Carry the training hyperparams forward; add the picker fields (win_rate, label) and
        # a provenance note. save_checkpoint re-stamps created_at/git_sha.
        metadata = dict(ckpt.get("metadata") or {})
        metadata["games_trained"] = games
        metadata["win_rate"] = win_rate
        metadata["notes"] = f"{label} rung — {args.run.name} @ {games} games"
        metadata["source"] = str(src)

        dst = args.out / f"{stem}.pt"
        save_checkpoint(
            net, dst, trained_with=ckpt.get("trained_with", "td_lambda"), metadata=metadata
        )
        wr = f"{win_rate:.3f}" if win_rate is not None else "n/a"
        opp = metadata.get("eval_opponent")
        print(f"{label:7s} {dst}  (games={games}, win_rate={wr} vs {opp})")


if __name__ == "__main__":
    main()
