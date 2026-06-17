#!/usr/bin/env python
"""Score an agent against GNU Backgammon's evaluation (thin CLI).

Plays recorded self-play games (or reads an existing ``.mat``), exports them to a
Jellyfish ``.mat``, drives gnubg to analyse every chequer play, and reports the mean
**equity loss** vs. gnubg's preferred move plus the move-agreement rate. Lower equity
loss = stronger play — a clean, standard strength metric.

Examples
--------
    # zero-config demo (pubeval vs pubeval, 1 game, gnubg 2-ply)
    uv run python scripts/eval_vs_gnubg.py

    # a trained checkpoint as WHITE vs pubeval as BLACK, 5 games
    uv run python scripts/eval_vs_gnubg.py --agent runs/wp1/td_0020000.pt \
        --opponent pubeval --games 5 --seed 1

    # analyse a .mat we (or anyone) already produced
    uv run python scripts/eval_vs_gnubg.py --mat game.mat

Requires gnubg on PATH (Ubuntu: ``sudo apt-get install -y gnubg``); if absent the
script prints an install hint and exits cleanly.

Ply numbering: ``--plies`` is gnubg's own count (raw net = 0-ply), which is also this
repo's internal convention (CLAUDE.md §9) — no translation needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        default="pubeval",
        help="WHITE seat: 'pubeval', 'random', or a checkpoint path (default: pubeval)",
    )
    parser.add_argument(
        "--opponent",
        default="pubeval",
        help="BLACK seat: 'pubeval', 'random', or a checkpoint path (default: pubeval)",
    )
    parser.add_argument("--games", type=int, default=1, help="self-play games to generate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (dice + random agents)")
    parser.add_argument("--plies", type=int, default=2, help="gnubg analysis ply (gnubg numbering)")
    parser.add_argument("--mat", type=Path, help="analyse this existing .mat instead of self-play")
    parser.add_argument("--keep-mat", type=Path, help="write the generated .mat here (else temp)")
    parser.add_argument("--per-move", action="store_true", help="print every move's equity loss")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent against gnubg (equity loss).")
    _add_args(parser)
    args = parser.parse_args()

    import tempfile

    import numpy as np

    from bgrl.agents import PubevalAgent, RandomAgent
    from bgrl.env import RandomDiceSource
    from bgrl.game import play_game
    from bgrl.serialization import (
        analyse_mat,
        gnubg_available,
        load_agent,
        load_checkpoint,
        match_to_mat,
        summarize,
    )

    if not gnubg_available():
        print("gnubg not found on PATH. Install it to run the analysis pipeline:")
        print("    sudo apt-get install -y gnubg")
        return

    def make_agent(spec: str, rng: np.random.Generator):
        if spec == "pubeval":
            return PubevalAgent()
        if spec == "random":
            return RandomAgent(rng)
        return load_agent(load_checkpoint(Path(spec)))

    if args.mat is not None:
        mat_path = args.mat
        label = str(mat_path)
        cleanup: tempfile.TemporaryDirectory | None = None
    else:
        rng = np.random.default_rng(args.seed)
        white = make_agent(args.agent, np.random.default_rng(args.seed + 1))
        black = make_agent(args.opponent, np.random.default_rng(args.seed + 2))
        games = []
        for _ in range(args.games):
            res = play_game(white, black, RandomDiceSource(rng), record=True)
            games.append((res.steps, res.outcome))
        mat_text = match_to_mat(games, white_name=args.agent, black_name=args.opponent)
        if args.keep_mat is not None:
            args.keep_mat.parent.mkdir(parents=True, exist_ok=True)
            args.keep_mat.write_text(mat_text)
            mat_path = args.keep_mat
            cleanup = None
        else:
            cleanup = tempfile.TemporaryDirectory()
            mat_path = Path(cleanup.name) / "eval.mat"
            mat_path.write_text(mat_text)
        label = f"{args.agent} (WHITE) vs {args.opponent} (BLACK), {args.games} game(s)"

    print(f"Analysing {label} with gnubg at {args.plies}-ply ...")
    moves = analyse_mat(mat_path, plies=args.plies)
    if cleanup is not None:
        cleanup.cleanup()

    if not moves:
        print("gnubg returned no analysed chequer plays.")
        return

    if args.per_move:
        header = f"\n{'g.ply':>6}  {'who':<5} {'dice':>4}  {'played':<22} {'best':<22} {'loss':>8}"
        print(header)
        for m in moves:
            flag = "" if m.agreed else " *"
            print(
                f"{m.game}.{m.ply:<4} {m.player:<5} {m.dice[0]}{m.dice[1]:>3}  "
                f"{m.move_made:<22} {m.best_move:<22} {m.equity_loss:8.4f}{flag}"
            )

    summary = summarize(moves)
    print("\nMean equity loss (lower = stronger) and gnubg-move agreement:")
    for side in ("White", "Black", "overall"):
        s = summary[side]
        if s.moves:
            print(
                f"  {side:<8} {s.moves:4d} moves   "
                f"mean loss {s.mean_equity_loss:7.4f}   agreement {s.agreement:6.1%}"
            )


if __name__ == "__main__":
    sys.exit(main())
