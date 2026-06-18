#!/usr/bin/env python
"""Marginal / sliced win-rate plots from sweep tournament matrices (thin CLI).

Reads one or more win-rate matrix CSVs written by ``scripts/tournament.py``, extracts
each run's win-rate **vs the absolute anchor** (``pubeval`` by default), parses
``(lr, lambda, hidden)`` out of the run labels (``lr<lr>_l<lam>_h<H>``), and draws:

1. **Marginal main-effect plots** — win-rate vs each hyperparameter (lambda, lr, hidden),
   marginalising (averaging) over the other two. Individual runs are faint points; the
   mean over the marginal is the bold line. Each input CSV (e.g. best.pt / final.pt) is a
   separate series.
2. **An ``lr x lambda`` heatmap grid faceted by hidden** (one figure per input CSV), so a
   fixed slice such as ``lambda=0.7`` reads off as a single column.

    uv run --group viz python scripts/plot_marginals.py \
        runs/sweep/winrate_matrix_best.csv runs/sweep/winrate_matrix_final.csv \
        --out-prefix runs/sweep/winrate
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Run labels emitted by tournament.py: f"lr{lr:g}_l{lam:g}_h{hidden}" (e.g. lr0.05_l0.7_h64).
_LABEL_RE = re.compile(r"^lr([\d.]+)_l([\d.]+)_h(\d+)")


def _load_vs_anchor(path: Path, anchor: str) -> dict[tuple[float, float, int], float]:
    """Map ``(lr, lam, hidden) -> win-rate vs anchor`` from a tournament matrix CSV."""
    rows = list(csv.reader(path.open()))
    labels = rows[0][1:]
    matrix = [[float(v) for v in r[1:]] for r in rows[1:]]
    if anchor not in labels:
        raise ValueError(f"anchor {anchor!r} not a column in {path}")
    ai = labels.index(anchor)
    out: dict[tuple[float, float, int], float] = {}
    for i, lbl in enumerate(labels):
        m = _LABEL_RE.match(lbl)
        if not m:
            continue  # skips the anchor row itself
        key = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
        out[key] = matrix[i][ai]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Marginal / sliced win-rate plots.")
    parser.add_argument("csvs", nargs="+", type=Path, help="tournament win-rate matrix CSV(s)")
    parser.add_argument("--anchor", default="pubeval", help="reference column to score against")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("runs/sweep/winrate"),
        help="output path prefix; writes <prefix>_marginals.png and <prefix>_grid_<series>.png",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # series name <- CSV filename: winrate_matrix_best.csv -> "best".
    series = {
        p.stem.replace("winrate_matrix_", "") or p.stem: _load_vs_anchor(p, args.anchor)
        for p in args.csvs
    }

    lrs = sorted({k[0] for d in series.values() for k in d})
    lams = sorted({k[1] for d in series.values() for k in d})
    hiddens = sorted({k[2] for d in series.values() for k in d})
    dims = [("lambda", 1, lams), ("lr", 0, lrs), ("hidden", 2, hiddens)]
    # more colours than series is intentional, so stop at the shorter (strict=False).
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    colors = {name: c for name, c in zip(series, palette, strict=False)}

    # ---- Figure 1: marginal main effects (one panel per hyperparameter) -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, idx, values) in zip(axes, dims, strict=True):
        for sname, data in series.items():
            means = []
            for v in values:
                wrs = [wr for key, wr in data.items() if key[idx] == v]
                xj = values.index(v) + (list(series).index(sname) - 0.5) * 0.08
                ax.scatter([xj] * len(wrs), wrs, s=18, alpha=0.35, color=colors[sname])
                means.append(float(np.mean(wrs)))
            ax.plot(range(len(values)), means, "-o", color=colors[sname], label=sname, linewidth=2)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="pubeval parity")
        ax.set_xticks(range(len(values)), [f"{v:g}" for v in values])
        ax.set_xlabel(name)
        ax.set_ylabel(f"win-rate vs {args.anchor}")
        ax.set_title(f"Marginal effect of {name}")
        ax.grid(True, alpha=0.3)
    # de-duplicate legend (parity line repeats across panels)
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels, strict=True):
        seen.setdefault(lbl, h)
    fig.legend(
        seen.values(), seen.keys(), loc="upper center", ncol=len(seen), bbox_to_anchor=(0.5, 1.02)
    )
    fig.suptitle(
        f"Marginal win-rate vs {args.anchor} (averaged over the other two hyperparameters)", y=1.06
    )
    fig.tight_layout()
    out_marg = Path(f"{args.out_prefix}_marginals.png")
    out_marg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_marg, dpi=150, bbox_inches="tight")
    print(f"wrote {out_marg}")

    # ---- Figure 1b: the classic lambda curve (single panel, marginalised over lr & hidden) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = list(range(len(lams)))
    for sname, data in series.items():
        means = [float(np.mean([wr for k, wr in data.items() if k[1] == v])) for v in lams]
        stds = [float(np.std([wr for k, wr in data.items() if k[1] == v])) for v in lams]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=f"{sname}.pt")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label=f"{args.anchor} parity")
    ax.set_xticks(xs, [f"{v:g}" for v in lams])
    ax.set_xlabel("lambda (eligibility-trace decay)")
    ax.set_ylabel(f"win-rate vs {args.anchor}")
    ax.set_title("Effect of lambda on TD(lambda) strength (marginalised over lr & hidden)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_lam = Path(f"{args.out_prefix}_lambda.png")
    fig.savefig(out_lam, dpi=150, bbox_inches="tight")
    print(f"wrote {out_lam}")

    # ---- Figure 2 (per series): lr x lambda heatmap grid, faceted by hidden -------------
    for sname, data in series.items():
        fig, axes = plt.subplots(1, len(hiddens), figsize=(5 * len(hiddens), 5), squeeze=False)
        for ax, h in zip(axes[0], hiddens, strict=True):
            grid = np.full((len(lrs), len(lams)), np.nan)
            for (lr, lam, hh), wr in data.items():
                if hh == h:
                    grid[lrs.index(lr), lams.index(lam)] = wr
            im = ax.imshow(grid, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
            ax.set_yticks(range(len(lrs)), [f"{v:g}" for v in lrs])
            ax.set_xlabel("lambda")
            ax.set_ylabel("lr")
            ax.set_title(f"hidden={h}")
            for i in range(len(lrs)):
                for j in range(len(lams)):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(
                            j,
                            i,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if abs(v - 0.5) > 0.18 else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"win-rate vs {args.anchor}")
        fig.suptitle(
            f"{sname}.pt: win-rate vs {args.anchor} by (lr, lambda); fix a column to slice a lambda"
        )
        fig.tight_layout()
        out_grid = Path(f"{args.out_prefix}_grid_{sname}.png")
        fig.savefig(out_grid, dpi=150, bbox_inches="tight")
        print(f"wrote {out_grid}")


if __name__ == "__main__":
    main()
