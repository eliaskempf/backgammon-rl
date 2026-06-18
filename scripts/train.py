#!/usr/bin/env python
"""Train a TD(λ) value net by self-play (thin CLI; logic lives in ``bgrl``).

Periodically evaluates the live net against a benchmark opponent (``pubeval`` by
default — Tesauro's public-domain linear evaluator, the standard absolute yardstick),
logs the curve to ``{out-dir}/metrics.csv`` (and optionally Weights & Biases), and
checkpoints periodically plus a ``best.pt`` (best eval win-rate) and ``final.pt``.

Online TD(λ) is single-process; pin BLAS to one thread (the net is tiny, so intra-op
threading is pure overhead and would oversubscribe a shared node).

**Resumable & SIGTERM-safe.** A rolling ``latest.pt`` bundles the net plus the two RNG
states + game counter; ``--resume`` continues from it **bit-exactly** (traces are zero at
every game boundary, so weights + RNG + counter are the whole state). On SIGTERM (SLURM
time limit / preemption) or SIGINT the current game finishes, ``latest.pt`` is written,
and the process exits cleanly — so a requeued job continues seamlessly.

Example
-------
    uv run python scripts/train.py --games 1000000 --eval-opponent pubeval \
        --eval-every 25000 --save-every 50000 --seed 0 --out-dir runs/wp1 --wandb --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path


class _ResumeExit(Exception):
    """Raised at a game boundary after a stop signal — ``latest.pt`` is already saved."""

    def __init__(self, games_done: int) -> None:
        self.games_done = games_done


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play TD(λ) trainer.")
    parser.add_argument("--games", type=int, default=1_000_000, help="total self-play games")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from {out-dir}/latest.pt if it exists (else start fresh)",
    )
    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="backgammon-rl", help="W&B project name")
    args = parser.parse_args()

    # Heavy imports inside main so `--help` stays light and mp workers (if added
    # later) don't pull torch into every process. Mirrors scripts/benchmark_env.py.
    import csv
    import signal

    import numpy as np
    import torch

    from bgrl.agents import Agent, PubevalAgent, RandomAgent, ValueAgent
    from bgrl.agents.td_agent import TDAgent
    from bgrl.game import GameResult
    from bgrl.nets.value_net import MLPValueNet
    from bgrl.serialization import load_agent, load_checkpoint, load_net, save_checkpoint
    from bgrl.training.evaluate import play_match
    from bgrl.training.loop import train

    torch.set_num_threads(1)  # tiny net: intra-op threads only add overhead / oversubscribe;
    # also required for the bit-exact (single-thread-deterministic) resume guarantee.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.out_dir / "latest.pt"
    total = args.games

    # --- Resume from latest.pt, or start fresh ---------------------------------------
    resuming = args.resume and latest_path.exists()
    if resuming:
        meta = load_checkpoint(latest_path)["metadata"]
        net = load_net(load_checkpoint(latest_path))  # arch (incl. hidden) comes from the file
        start = int(meta["games_trained"])
        best_win_rate = float(meta.get("best_win_rate", -1.0))
        lam, lr, gamma = float(meta["lam"]), float(meta["lr"]), float(meta["gamma"])
        seed = int(meta["seed"])
        eval_opponent = meta.get("eval_opponent", args.eval_opponent)
        wandb_run_id = meta.get("wandb_run_id")
        for name, cli_v, stored_v in (
            ("lam", args.lam, lam),
            ("lr", args.lr, lr),
            ("gamma", args.gamma, gamma),
        ):
            if cli_v != stored_v:
                print(f"WARNING: resuming with stored {name}={stored_v} (CLI {cli_v} ignored)")
        torch.manual_seed(seed)  # harmless — weights are loaded, not re-initialised
        train_rng, eval_rng = np.random.default_rng(seed).spawn(2)
        train_rng.bit_generator.state = meta["train_rng_state"]
        eval_rng.bit_generator.state = meta["eval_rng_state"]
    else:
        seed, lam, lr, gamma = args.seed, args.lam, args.lr, args.gamma
        eval_opponent, wandb_run_id = args.eval_opponent, None
        start, best_win_rate = 0, -1.0
        torch.manual_seed(seed)  # reproducible net initialisation
        # Two INDEPENDENT streams: training dice must never be perturbed by eval.
        train_rng, eval_rng = np.random.default_rng(seed).spawn(2)
        net = MLPValueNet(hidden=args.hidden)

    agent = TDAgent(net, lam=lam, lr=lr, gamma=gamma)
    config = {
        "games": total,
        "hidden": net.arch_config()["hidden"],
        "lam": lam,
        "lr": lr,
        "gamma": gamma,
        "seed": seed,
        "eval_opponent": eval_opponent,
    }

    def make_opponent() -> Agent:
        if eval_opponent == "pubeval":
            return PubevalAgent()
        if eval_opponent == "random":
            return RandomAgent(eval_rng)
        return load_agent(load_checkpoint(eval_opponent))  # a checkpoint path

    opponent = make_opponent()

    def ckpt_metadata(n: int, **extra: object) -> dict:
        return {"games_trained": n, **config, **extra}

    # Stream metrics to disk (monitorable / crash-surviving); append when resuming.
    metrics_path = args.out_dir / "metrics.csv"
    write_header = not (resuming and metrics_path.exists())
    metrics_file = metrics_path.open("a" if resuming else "w", newline="")
    metrics = csv.writer(metrics_file)
    if write_header:
        metrics.writerow(["games", "win_rate", "avg_plies", "truncated"])
        metrics_file.flush()

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            # Name the run after its output-dir label (e.g. lr0.1_lam0.7_h128_s0) so the
            # W&B dashboard is readable instead of using a random adjective-noun name.
            init_kw = {
                "project": args.wandb_project,
                "name": args.out_dir.name,
                "config": config,
                "dir": str(args.out_dir),
            }
            if resuming and wandb_run_id:
                init_kw |= {"id": wandb_run_id, "resume": "allow"}
            wandb_run = wandb.init(**init_kw)
            wandb_run_id = wandb_run.id
        except Exception as exc:  # never let optional logging kill a long run
            print(f"WARNING: --wandb requested but unavailable ({exc!r}); continuing with CSV only")

    # --- Stop-signal handling: SLURM sends SIGTERM near the time limit -----------------
    stop = {"requested": False}

    def request_stop(signum: int, frame: object) -> None:
        stop["requested"] = True  # handlers must be minimal — save at the next boundary

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    last_win_rate: float | None = None

    def save_latest(games_done: int) -> None:
        """Atomically write the rolling resume bundle: net + RNG states + counter + best."""
        save_checkpoint(
            net,
            latest_path,
            trained_with="td_lambda",
            metadata=ckpt_metadata(
                games_done,
                best_win_rate=best_win_rate,
                wandb_run_id=wandb_run_id,
                train_rng_state=train_rng.bit_generator.state,
                eval_rng_state=eval_rng.bit_generator.state,
            ),
        )

    def run_eval(n: int) -> float:
        """Evaluate the live net vs the benchmark, record the row, and update best.pt.

        A fresh non-learning ``ValueAgent`` is used so eval fires no learning hooks and
        never perturbs the trainer's traces.
        """
        nonlocal best_win_rate
        res = play_match(ValueAgent(net), opponent, pairs=args.eval_pairs, rng=eval_rng)
        print(
            f"[game {n}] win-rate vs {eval_opponent}: {res.win_rate_a:.3f} "
            f"(avg plies {res.avg_plies:.1f}, {res.truncated} truncated)"
        )
        metrics.writerow([n, res.win_rate_a, res.avg_plies, res.truncated])
        metrics_file.flush()
        if wandb_run is not None:
            # commit=True is required: with an explicit `step`, wandb defaults to
            # commit=False, which buffers each eval's metrics until the *next* eval (so
            # the live dashboard lags one eval and shows only system metrics early on).
            wandb_run.log(
                {
                    "win_rate": res.win_rate_a,
                    "avg_plies": res.avg_plies,
                    "truncated": res.truncated,
                },
                step=n,
                commit=True,
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
        games_done = start + n  # global counter across resumes
        if args.eval_every and games_done % args.eval_every == 0:
            last_win_rate = run_eval(games_done)
        if args.save_every and games_done % args.save_every == 0:
            path = args.out_dir / f"td_{games_done:07d}.pt"
            save_checkpoint(net, path, trained_with="td_lambda", metadata=ckpt_metadata(games_done))
            save_latest(games_done)
            print(f"[game {games_done}] saved {path}")
        if stop["requested"]:
            save_latest(games_done)
            raise _ResumeExit(games_done)

    verb = "resuming" if resuming else "training"
    print(f"{verb} -> {total} games (start={start}, {config}) -> {args.out_dir}")
    try:
        if start >= total:
            print(f"already complete ({start} >= {total} games); nothing to do")
        else:
            train(agent, games=total - start, rng=train_rng, on_game_end=on_game_end)
            # Final eval unless the last game already landed on the eval cadence (avoids a
            # duplicate metrics row / wandb step), then always write final.pt + latest.pt.
            if args.eval_every and total % args.eval_every != 0:
                last_win_rate = run_eval(total)
            save_checkpoint(
                net,
                args.out_dir / "final.pt",
                trained_with="td_lambda",
                metadata=ckpt_metadata(total, win_rate=last_win_rate),
            )
            save_latest(total)
            print(f"done (best win-rate vs {eval_opponent}: {best_win_rate:.3f})")
    except _ResumeExit as exit_:
        print(f"signal received: checkpointed at game {exit_.games_done} for resume; exiting")
    finally:
        metrics_file.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
