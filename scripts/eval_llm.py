#!/usr/bin/env python
"""Evaluate an LLM agent's win-rate vs a fixed opponent via CRN (thin CLI).

Offline by default: a network-free fake client that always plays the first legal
move, so the wiring runs with no key and no cost. Pass ``--live`` (with
``OPENROUTER_API_KEY`` available — exported or in a local dotenv file) to play with a
real model. A hard ``--budget-usd`` / ``--max-calls`` cap aborts mid-match with partial
stats rather than overspending, and ``--cache`` reuses paid responses across runs.

Reports the CRN win-rate (variance-cancelled, see ``bgrl.training.evaluate``) plus the
agent's harness stats: API calls, re-prompts, invalid-response and fallback rates, and
token/cost accounting. This is the headline "does it beat pubeval?" number; for a
lower-variance *move-quality* read, analyse the LLM's games with gnubg (eval_vs_gnubg).

Examples
--------
    # offline wiring check (no key, no cost)
    uv run python scripts/eval_llm.py --model dry --pairs 2

    # live: 10 CRN pairs (20 games) of haiku vs pubeval, capped at $2
    uv run python scripts/eval_llm.py --live --model anthropic/claude-haiku-4-5 \
        --renderer pip_list --template coach --format index_text \
        --opponent pubeval --pairs 10 --seed 1 \
        --cache runs/llm/play_cache.jsonl --budget-usd 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    args = _parse_args()

    import numpy as np

    from bgrl.llm.build import build_chat_client, build_llm_agent
    from bgrl.llm.client import BudgetExceededError
    from bgrl.training.evaluate import play_match

    client, guard = build_chat_client(
        live=args.live,
        cache_path=args.cache,
        budget_usd=args.budget_usd,
        max_calls=args.max_calls,
    )
    rng = np.random.default_rng(args.seed)
    agent = build_llm_agent(
        client,
        model=args.model,
        renderer=args.renderer,
        template=args.template,
        output_format=args.format,
        reasoning=args.reasoning,
        max_reprompts=args.max_reprompts,
        fallback=args.fallback,
        rng=np.random.default_rng(args.seed + 1),
    )
    opponent, opp_name = _build_opponent(args.opponent, rng)

    if not args.live:
        print("[dry run] offline fake client (first-legal moves); pass --live for a real model.\n")

    try:
        res = play_match(agent, opponent, pairs=args.pairs, rng=rng)
    except BudgetExceededError as exc:
        print(f"\nBUDGET CAP HIT mid-match: {exc}")
        _print_stats(agent.stats, guard)
        sys.exit(1)

    print(
        f"{args.model} vs {opp_name}: win-rate {res.win_rate_a:.3f} over {res.games} games "
        f"({res.wins_a}-{res.wins_b}, {res.truncated} truncated, avg plies {res.avg_plies:.1f})"
    )
    _print_stats(agent.stats, guard)


def _build_opponent(spec, rng):
    from bgrl.agents import PubevalAgent, RandomAgent

    if spec == "pubeval":
        return PubevalAgent(), "pubeval"
    if spec == "random":
        return RandomAgent(rng), "random"
    from bgrl.serialization import load_agent, load_checkpoint

    return load_agent(load_checkpoint(Path(spec))), spec


def _print_stats(stats, guard):
    print(
        f"  decisions={stats.decisions}  api_calls={stats.api_calls}  "
        f"reprompts={stats.reprompts}  invalid_rate={stats.invalid_rate:.3f}  "
        f"fallback_rate={stats.fallback_rate:.3f}"
    )
    print(
        f"  tokens: prompt={stats.total_prompt_tokens} completion={stats.total_completion_tokens} "
        f"reported_cost=${stats.total_cost:.4f}"
    )
    if guard is not None:
        print(f"  billed spend (cache-miss only): ${guard.spent:.4f} over {guard.calls} live calls")
    if stats.structured_unsupported:
        print("  note: model rejected structured output; downgraded to text indices")


def _parse_args():
    parser = argparse.ArgumentParser(description="CRN win-rate for an LLM agent vs an opponent.")
    parser.add_argument(
        "--model", required=True, help="OpenRouter model id (any string in dry-run)"
    )
    parser.add_argument(
        "--opponent", default="pubeval", help="'pubeval', 'random', or a checkpoint path"
    )
    parser.add_argument("--pairs", type=int, default=10, help="CRN game-pairs to play (2x games)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (dice + random opponent)")
    parser.add_argument(
        "--renderer",
        default="pip_list",
        choices=["ascii", "pip_list", "moves_only", "position_id"],
    )
    parser.add_argument("--template", default="coach", choices=["terse", "coach"])
    parser.add_argument("--format", default="index_text", choices=["index_text", "structured"])
    parser.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"])
    parser.add_argument("--max-reprompts", type=int, default=2)
    parser.add_argument(
        "--fallback", default="first_legal", choices=["first_legal", "random_legal"]
    )
    parser.add_argument(
        "--budget-usd", type=float, default=float("inf"), help="hard cost cap (live)"
    )
    parser.add_argument("--max-calls", type=int, default=None, help="hard model-call cap (live)")
    parser.add_argument("--cache", default=None, help="JSONL response cache path")
    parser.add_argument("--live", action="store_true", help="use the real OpenRouter API")
    return parser.parse_args()


if __name__ == "__main__":
    main()
