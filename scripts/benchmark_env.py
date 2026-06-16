#!/usr/bin/env python
"""Env + net throughput benchmark (thin CLI; logic lives in ``bgrl.bench``).

Examples
--------
    uv run python scripts/benchmark_env.py --games 2000 --bench-net --tag laptop
    uv run python scripts/benchmark_env.py --games 5000 --workers 32 --bench-net \
        --tag cluster-cpu --out bench_results/cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _default_workers() -> int:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return physical
    except Exception:
        pass
    return os.cpu_count() or 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Backgammon env/net throughput benchmark.")
    parser.add_argument("--games", type=int, default=2000, help="self-play games for env bench")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=0, help="0 = physical cores (fallback: logical)"
    )
    parser.add_argument("--profile", action="store_true", help="cProfile the single-worker loop")
    parser.add_argument("--bench-net", action="store_true", help="run the CPU net-eval sweep")
    parser.add_argument("--net-hidden", type=int, default=64)
    parser.add_argument("--net-iters", type=int, default=200)
    parser.add_argument("--net-warmup", type=int, default=20)
    parser.add_argument("--tag", default="local", help="free-form label stored in the result")
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    workers = args.workers or _default_workers()

    # Imported here (not at module top) so spawn workers re-importing this script
    # as __mp_main__ never pull torch into every worker.
    from bgrl.bench.runner import run_all
    from bgrl.bench.schema import write_json

    result = run_all(
        games=args.games,
        seed=args.seed,
        workers=workers,
        bench_net=args.bench_net,
        net_hidden=args.net_hidden,
        net_iters=args.net_iters,
        net_warmup=args.net_warmup,
        profile=args.profile,
        tag=args.tag,
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
    else:
        host = result["fingerprint"]["hostname"]
        out = Path("bench_results") / f"{host}-{args.tag}.json"
    write_json(result, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
