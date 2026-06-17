#!/usr/bin/env python
"""Render a pairwise win-rate matrix CSV as a divergent red/blue heatmap (thin CLI).

Reads the square CSV written by ``scripts/tournament.py`` (row/col labels + win-rates)
and draws it with a diverging ``RdBu`` scale centred at 0.5: **blue = the row agent beats
the column agent (win-rate > 50%)**, red = it loses, white ≈ even. Saved as a PNG.

Needs the ``viz`` dependency group:

    uv run --group viz python scripts/plot_heatmap.py runs/sweep/winrate_matrix.csv \
        --out runs/sweep/winrate_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise win-rate heatmap (RdBu, blue>50%).")
    parser.add_argument("csv", type=Path, help="win-rate matrix CSV from scripts/tournament.py")
    parser.add_argument("--out", type=Path, default=None, help="output PNG (default: CSV + .png)")
    parser.add_argument("--title", default="Pairwise win-rate (row vs column)", help="plot title")
    parser.add_argument(
        "--annot",
        choices=("auto", "on", "off"),
        default="auto",
        help="annotate cells with win-rates ('auto' = on for <=24 agents)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # headless (login/compute node has no display)
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(csv.reader(args.csv.open()))
    labels = rows[0][1:]
    matrix = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=np.float64)
    n = len(labels)
    if matrix.shape != (n, n):
        raise ValueError(f"expected a square {n}x{n} matrix, got {matrix.shape}")
    annotate = args.annot == "on" or (args.annot == "auto" and n <= 24)

    side = max(6.0, 0.45 * n + 2.5)
    fig, ax = plt.subplots(figsize=(side, side))
    # RdBu maps low->red, high->blue; vmin/vmax=0/1 puts 0.5 at the white centre.
    im = ax.imshow(matrix, cmap="RdBu", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n), labels=labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels=labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    if annotate:
        for i in range(n):
            for j in range(n):
                v = matrix[i, j]
                # White text on the saturated ends, dark near the white centre.
                color = "white" if abs(v - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row win-rate vs column (blue > 50%)")
    ax.set_title(args.title)
    fig.tight_layout()

    out = args.out or args.csv.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {n}x{n} heatmap -> {out}")


if __name__ == "__main__":
    main()
