#!/usr/bin/env python
"""Train a TD(λ) value net by self-play (thin CLI; logic lives in ``bgrl``).

Periodically evaluates the live net against a benchmark opponent (``pubeval`` by
default — Tesauro's public-domain linear evaluator, the standard absolute yardstick),
logs the curve to ``{out-dir}/metrics.csv`` (and optionally Weights & Biases), and
checkpoints periodically plus a ``best.pt`` (best eval win-rate) and ``final.pt``.

Online TD(λ) is single-process; pin BLAS to one thread (the net is tiny, so intra-op
threading is pure overhead and would oversubscribe a shared node).

Example
-------
    uv run python scripts/train.py --games 1000000 --eval-opponent pubeval \
        --eval-every 25000 --save-every 50000 --seed 0 --out-dir runs/wp1 --wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play TD(λ) trainer.")
    parser.add_argument("--games", type=int, default=1_000_000, help="self-play games to run")
    parser.add_argument("--hidden", type=int, default=64, help="hidden units in the value net")
    parser.add_argument("--lam", type=float, default=0.7, help="TD(λ) trace-decay λ")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount (1.0 = undiscounted)")
    parser.add_argument("--save-every", type=int, default=50_000, help="checkpoint cadence (games)")
    parser.add_argument(
        "--eval-every", type=int, default=25_000, help="eval cadence (games; 0=off)"
    )
    parser.add_argument("--eval-pairs", type=int, default=100, help="CRN game-pairs per eval")
    parser.add_argument(
        "--eval-opponent",
        default="pubeval",
        help="eval opponent: 'pubeval', 'random', or a checkpoint path",
    )
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed (numpy + torch)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/wp1"), help="output directory")
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="backgammon-rl", help="W&B project name")
    args = parser.parse_args()

    # Heavy imports inside main so `--help` stays light and mp workers (if added
    # later) don't pull torch into every process. Mirrors scripts/benchmark_env.py.
    import csv
    import time

    import numpy as np
    import torch

    from bgrl.agents import Agent, PubevalAgent, RandomAgent, ValueAgent
    from bgrl.agents.td_agent import TDAgent
    from bgrl.game import GameResult
    from bgrl.nets.value_net import MLPValueNet
    from bgrl.serialization import load_agent, load_checkpoint, save_checkpoint
    from bgrl.training.evaluate import play_match
    from bgrl.training.loop import train

    torch.set_num_threads(1)  # tiny net: intra-op threads only add overhead / oversubscribe
    # Seed torch so net initialisation is reproducible, and split the numpy seed into two
    # INDEPENDENT streams: training dice must never be perturbed by eval, or "same seed ->
    # same curve" breaks.
    torch.manual_seed(args.seed)
    train_rng, eval_rng = np.random.default_rng(args.seed).spawn(2)

    net = MLPValueNet(hidden=args.hidden)
    agent = TDAgent(net, lam=args.lam, lr=args.lr, gamma=args.gamma)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "games": args.games,
        "hidden": args.hidden,
        "lam": args.lam,
        "lr": args.lr,
        "gamma": args.gamma,
        "seed": args.seed,
        "eval_opponent": args.eval_opponent,
    }

    def make_opponent() -> Agent:
        if args.eval_opponent == "pubeval":
            return PubevalAgent()
        if args.eval_opponent == "random":
            return RandomAgent(eval_rng)
        return load_agent(load_checkpoint(args.eval_opponent))  # a checkpoint path

    opponent = make_opponent()

    def ckpt_metadata(n: int, **extra: object) -> dict:
        return {"games_trained": n, **config, **extra}

    # Stream metrics to disk so a long unattended run is monitorable / survives a crash.
    metrics_path = args.out_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(["games", "win_rate", "avg_plies", "truncated", "wall_seconds"])
    metrics_file.flush()

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, config=config, dir=str(args.out_dir))
        except Exception as exc:  # never let optional logging kill a long run
            # CSV remains the reliable record; warn loudly but keep training.
            print(f"WARNING: --wandb requested but unavailable ({exc!r}); continuing with CSV only")

    start = time.perf_counter()
    best_win_rate = -1.0
    last_win_rate: float | None = None

    def run_eval(n: int) -> float:
        """Evaluate the live net vs the benchmark, log/record, and update ``best.pt``.

        Uses a fresh, non-learning ``ValueAgent`` so eval fires no learning hooks and
        never perturbs the trainer's traces.
        """
        nonlocal best_win_rate
        res = play_match(ValueAgent(net), opponent, pairs=args.eval_pairs, rng=eval_rng)
        elapsed = time.perf_counter() - start
        print(
            f"[game {n}] win-rate vs {args.eval_opponent}: {res.win_rate_a:.3f} "
            f"(avg plies {res.avg_plies:.1f}, {res.truncated} truncated)"
        )
        metrics.writerow([n, res.win_rate_a, res.avg_plies, res.truncated, round(elapsed, 1)])
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "win_rate": res.win_rate_a,
                    "avg_plies": res.avg_plies,
                    "truncated": res.truncated,
                    "wall_seconds": elapsed,
                },
                step=n,
            )
        if res.win_rate_a > best_win_rate:
            best_win_rate = res.win_rate_a
            save_checkpoint(
                net,
                args.out_dir / "best.pt",
                trained_with="td_lambda",
                metadata=ckpt_metadata(n, win_rate=res.win_rate_a),
            )
        return res.win_rate_a

    def on_game_end(n: int, result: GameResult) -> None:
        nonlocal last_win_rate
        if args.eval_every and n % args.eval_every == 0:
            last_win_rate = run_eval(n)
        if args.save_every and n % args.save_every == 0:
            path = args.out_dir / f"td_{n:07d}.pt"
            save_checkpoint(net, path, trained_with="td_lambda", metadata=ckpt_metadata(n))
            print(f"[game {n}] saved {path}")

    print(f"training {args.games} games ({config}) -> {args.out_dir}")
    try:
        train(agent, games=args.games, rng=train_rng, on_game_end=on_game_end)
        # Final eval unless the last game already landed on the eval cadence (avoids a
        # duplicate metrics row / wandb step), then always write final.pt.
        if args.eval_every and args.games % args.eval_every != 0:
            last_win_rate = run_eval(args.games)
        save_checkpoint(
            net,
            args.out_dir / "final.pt",
            trained_with="td_lambda",
            metadata=ckpt_metadata(args.games, win_rate=last_win_rate),
        )
        print(f"done (best win-rate vs {args.eval_opponent}: {best_win_rate:.3f})")
    finally:
        metrics_file.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
