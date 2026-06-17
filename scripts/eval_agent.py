#!/usr/bin/env python
"""Evaluate a checkpoint's win-rate via common random numbers (thin CLI).

Examples
--------
    uv run python scripts/eval_agent.py --checkpoint runs/wp1/td_0020000.pt \
        --opponent random --pairs 200 --seed 1
    uv run python scripts/eval_agent.py --checkpoint new.pt --opponent old.pt --pairs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CRN win-rate evaluation for a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="agent A checkpoint (.pt)")
    parser.add_argument(
        "--opponent",
        default="pubeval",
        help="agent B: 'pubeval', 'random', or a path to another checkpoint",
    )
    parser.add_argument("--pairs", type=int, default=200, help="CRN game-pairs to play")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (dice + random opponent)")
    args = parser.parse_args()

    import numpy as np

    from bgrl.agents import PubevalAgent, RandomAgent
    from bgrl.serialization import load_agent, load_checkpoint
    from bgrl.training.evaluate import play_match

    rng = np.random.default_rng(args.seed)
    agent_a = load_agent(load_checkpoint(args.checkpoint))
    if args.opponent == "pubeval":
        agent_b = PubevalAgent()
        opp_name = "pubeval"
    elif args.opponent == "random":
        agent_b = RandomAgent(rng)
        opp_name = "random"
    else:
        agent_b = load_agent(load_checkpoint(Path(args.opponent)))
        opp_name = str(args.opponent)

    res = play_match(agent_a, agent_b, pairs=args.pairs, rng=rng)
    print(
        f"{args.checkpoint} vs {opp_name}: win-rate {res.win_rate_a:.3f} "
        f"over {res.games} games ({res.wins_a}-{res.wins_b}, "
        f"{res.truncated} truncated, avg plies {res.avg_plies:.1f})"
    )


if __name__ == "__main__":
    main()
