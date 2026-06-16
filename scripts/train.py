#!/usr/bin/env python
"""Train a TD(λ) value net by self-play (thin CLI; logic lives in ``bgrl``).

Example
-------
    uv run python scripts/train.py --games 20000 --save-every 2000 \
        --eval-every 1000 --seed 0 --out-dir runs/wp1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play TD(λ) trainer.")
    parser.add_argument("--games", type=int, default=20_000, help="self-play games to run")
    parser.add_argument("--hidden", type=int, default=64, help="hidden units in the value net")
    parser.add_argument("--lam", type=float, default=0.7, help="TD(λ) trace-decay λ")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount (1.0 = undiscounted)")
    parser.add_argument("--save-every", type=int, default=2_000, help="checkpoint cadence (games)")
    parser.add_argument("--eval-every", type=int, default=1_000, help="eval cadence (games; 0=off)")
    parser.add_argument("--eval-pairs", type=int, default=100, help="CRN game-pairs per eval")
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed (numpy + torch)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/wp1"), help="checkpoint dir")
    args = parser.parse_args()

    # Heavy imports inside main so `--help` stays light and mp workers (if added
    # later) don't pull torch into every process. Mirrors scripts/benchmark_env.py.
    import numpy as np
    import torch

    from bgrl.agents import RandomAgent, ValueAgent
    from bgrl.agents.td_agent import TDAgent
    from bgrl.game import GameResult
    from bgrl.nets.value_net import MLPValueNet
    from bgrl.serialization import save_checkpoint
    from bgrl.training.evaluate import play_match
    from bgrl.training.loop import train

    # Seed torch so net initialisation is reproducible, and split the numpy seed
    # into two INDEPENDENT streams: training dice must never be perturbed by eval,
    # or "same seed -> same curve" breaks.
    torch.manual_seed(args.seed)
    train_rng, eval_rng = np.random.default_rng(args.seed).spawn(2)

    net = MLPValueNet(hidden=args.hidden)
    agent = TDAgent(net, lam=args.lam, lr=args.lr, gamma=args.gamma)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def on_game_end(n: int, result: GameResult) -> None:
        if args.eval_every and n % args.eval_every == 0:
            # Evaluate the current weights with a fresh, non-learning ValueAgent so
            # eval fires no learning hooks and never perturbs the trainer's traces.
            res = play_match(
                ValueAgent(net), RandomAgent(eval_rng), pairs=args.eval_pairs, rng=eval_rng
            )
            print(
                f"[game {n}] win-rate vs random: {res.win_rate_a:.3f} "
                f"(avg plies {res.avg_plies:.1f}, {res.truncated} truncated)"
            )
        if args.save_every and n % args.save_every == 0:
            path = args.out_dir / f"td_{n:07d}.pt"
            save_checkpoint(net, path, trained_with="td_lambda", metadata={"games_trained": n})
            print(f"[game {n}] saved {path}")

    print(f"training {args.games} games (hidden={args.hidden}, lam={args.lam}, lr={args.lr})")
    train(agent, games=args.games, rng=train_rng, on_game_end=on_game_end)
    print("done")


if __name__ == "__main__":
    main()
