#!/usr/bin/env python
"""Deep per-head calibration report for a multi-head value-net checkpoint (WP6 A4).

The in-training ``--calib-games`` diagnostic (a few hundred games per eval) is enough to
*watch* the heads converge, but its realised rates for the rare gammon/backgammon heads
(backgammon is ~1-2% of games) are too noisy to *trust* as a final read. This runs one
large greedy-self-play pass over a saved checkpoint and prints, per outcome head, the mean
predicted vs realised rate, the expected calibration error, and the full reliability curve
(predicted vs realised within each probability bin) — the authoritative check on whether
the magnitude heads calibrated or collapsed to ~0. A collapsed *gammon* head is what gates
the optional A5 rollout fine-tuning (see ``plans/wp6-multihead-cube.md`` §A5).

20k games is ~10 min single-core, over the login-node budget, so run it on a compute node:

    srun -p lmbdlc2_cpu-epyc9655 -t 0:20:00 \
        uv run python scripts/calibrate.py --checkpoint runs/wp6_<run>/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# The rare magnitude heads — the reason this deep pass exists — get a full reliability dump.
MAGNITUDE_HEADS = ("p_win_g", "p_win_bg", "p_lose_g", "p_lose_bg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep per-head calibration of a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="value-net checkpoint (.pt)")
    parser.add_argument(
        "--games", type=int, default=20_000, help="greedy self-play games to sample (default 20000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for self-play dice")
    parser.add_argument("--bins", type=int, default=10, help="reliability-curve probability bins")
    parser.add_argument(
        "--out", type=Path, default=None, help="optional CSV path for the reliability table"
    )
    args = parser.parse_args()

    import numpy as np

    from bgrl.serialization import load_checkpoint, load_net
    from bgrl.training.calibration import HEAD_NAMES, collect_predictions, reliability_table

    net = load_net(load_checkpoint(args.checkpoint))
    rng = np.random.default_rng(args.seed)
    predicted, realized = collect_predictions(net, games=args.games, rng=rng)
    samples = len(predicted)
    print(f"{args.checkpoint}: {samples} non-terminal afterstates from {args.games} games\n")
    if samples == 0:
        print("no non-terminal afterstates sampled (all games truncated?)")
        return

    tables: dict[str, list] = {}
    print(f"{'head':>10} {'pred':>7} {'real':>7} {'ece':>7}  flag")
    for k, name in enumerate(HEAD_NAMES):
        p, r = predicted[:, k], realized[:, k]
        table = reliability_table(p, r, bins=args.bins)
        tables[name] = table
        pred_mean, real_mean = float(p.mean()), float(r.mean())
        ece = sum((b.count / samples) * abs(b.predicted_mean - b.realized_mean) for b in table)
        # The collapse signal: a head whose outcome clearly happens (real > 1%) but which the
        # net predicts at near-zero (< a quarter of the realised rate) never learned it.
        collapsed = real_mean >= 0.01 and pred_mean <= 0.25 * real_mean
        flag = "COLLAPSE?" if collapsed else ""
        print(f"{name:>10} {pred_mean:>7.3f} {real_mean:>7.3f} {ece:>7.3f}  {flag}")

    for name in MAGNITUDE_HEADS:
        print(f"\n{name} reliability:")
        print(f"  {'bin':>13} {'count':>8} {'pred':>7} {'real':>7}")
        for b in tables[name]:
            label = f"[{b.lo:.2f},{b.hi:.2f})"
            print(f"  {label:>13} {b.count:>8} {b.predicted_mean:>7.3f} {b.realized_mean:>7.3f}")

    if args.out is not None:
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["head", "bin_lo", "bin_hi", "count", "pred_mean", "real_mean"])
            for name in HEAD_NAMES:
                for b in tables[name]:
                    w.writerow([name, b.lo, b.hi, b.count, b.predicted_mean, b.realized_mean])
        print(f"\nwrote reliability table -> {args.out}")


if __name__ == "__main__":
    main()
