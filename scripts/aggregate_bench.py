#!/usr/bin/env python
"""Aggregate benchmark JSON files into a comparison table + decision block.

uv run python scripts/aggregate_bench.py bench_results/*.json
"""

from __future__ import annotations

import argparse
import glob


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark result JSON files.")
    parser.add_argument("paths", nargs="*", help="result JSON (default: bench_results/*.json)")
    args = parser.parse_args()

    from bgrl.bench.aggregate import load_results, render

    paths = args.paths or sorted(glob.glob("bench_results/*.json"))
    if not paths:
        print("no result files found (pass paths or populate bench_results/)")
        return
    print(render(load_results(paths)))


if __name__ == "__main__":
    main()
