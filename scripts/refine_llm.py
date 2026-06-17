#!/usr/bin/env python
"""Sweep LLM prompt/format/model candidates and print a ranked report (thin CLI).

By default this runs OFFLINE against a fake client that always plays the first legal
move — it validates the plumbing and prints a report without spending money or hitting
the network. Pass ``--live`` (with ``OPENROUTER_API_KEY`` set) to evaluate real models;
responses are cached (``--cache``) and a hard ``--budget-usd`` cap aborts the sweep with
a partial report rather than overspending.

Examples
--------
    # offline dry run (no API calls): exercises the full sweep + report
    uv run python scripts/refine_llm.py --positions 5

    # live sweep over two models and two board renderings, capped at $1
    uv run python scripts/refine_llm.py --live --budget-usd 1.0 --positions 30 \
        --models anthropic/claude-sonnet-4-6 openai/gpt-5 \
        --renderers ascii pip_list --formats index_text structured

    # grade against a trained value net (equity-loss) instead of pubeval (agreement)
    uv run python scripts/refine_llm.py --live --checkpoint runs/wp1/td_0020000.pt \
        --scorer reference --mode equity_loss --positions 30
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    args = _parse_args()

    from bgrl.llm.client import CachingChatClient, ResponseCache
    from bgrl.llm.parse import OutputFormat
    from bgrl.llm.prompt import ALL_TEMPLATES
    from bgrl.llm.refine import BudgetExceeded, SweepAxes, SweepConfig, run_sweep
    from bgrl.llm.render import ALL_RENDERERS

    axes = SweepAxes(
        models=tuple(args.models),
        renderers=tuple(ALL_RENDERERS[name] for name in args.renderers),
        templates=tuple(ALL_TEMPLATES[name] for name in args.templates),
        output_formats=tuple(OutputFormat(value) for value in args.formats),
        reasoning_options=tuple(_reasoning(opt) for opt in args.reasoning),
    )
    scorer = _build_scorer(args)
    config = SweepConfig(
        axes=axes,
        scorer=scorer,
        n_positions=args.positions,
        seed=args.seed,
        budget_usd=args.budget_usd,
        max_calls=args.max_calls,
        max_reprompts=args.max_reprompts,
    )

    client = _build_client(args)
    if not args.live:
        print("[dry run] offline fake client (first-legal moves); pass --live for real models.\n")

    cache = ResponseCache(args.cache) if args.cache else None
    if cache is not None or args.live:
        client = CachingChatClient(client, cache)

    try:
        report = run_sweep(config, client)
    except BudgetExceeded as exc:
        print(f"\nBUDGET CAP HIT: {exc}\n")
        _print_report(exc.report)
        if args.json:
            _dump_json(exc.report, args.json)
        sys.exit(1)
    except NotImplementedError as exc:
        print(f"scorer not available: {exc}", file=sys.stderr)
        sys.exit(2)

    _print_report(report)
    if args.json:
        _dump_json(report, args.json)


def _reasoning(option: str) -> dict | None:
    return None if option == "off" else {"effort": option}


def _build_scorer(args):
    from bgrl.llm.scorer import GnubgPositionScorer, ReferenceAgentScorer, ScoreMode

    if args.scorer == "gnubg":
        return GnubgPositionScorer()
    mode = ScoreMode(args.mode)
    if args.checkpoint is not None:
        return ReferenceAgentScorer.from_checkpoint(args.checkpoint, mode=mode)
    if mode is ScoreMode.EQUITY_LOSS:
        raise SystemExit("--mode equity_loss requires --checkpoint (a value-net oracle)")
    return ReferenceAgentScorer.pubeval()


def _build_client(args):
    if args.live:
        from bgrl.llm.client import OpenRouterClient

        return OpenRouterClient()
    from bgrl.llm.client import FakeChatClient

    return FakeChatClient(responder=lambda messages, params: "0")


def _print_report(report) -> None:
    print(
        f"scorer={report.scorer_name}  positions={report.n_positions}  "
        f"total_cost=${report.total_cost_usd:.4f}"
    )
    header = f"{'rank':>4}  {'score':>7}  {'invalid':>7}  {'fallbk':>7}  {'cost$':>8}  candidate"
    print(header)
    print("-" * len(header))
    for rank, c in enumerate(report.candidates, start=1):
        flag = " [struct-unsupported]" if c.structured_unsupported else ""
        print(
            f"{rank:>4}  {c.mean_score:>7.3f}  {c.invalid_rate:>7.3f}  {c.fallback_rate:>7.3f}  "
            f"{c.total_cost_usd:>8.4f}  {c.label}{flag}"
        )


def _dump_json(report, path) -> None:
    import json
    from dataclasses import asdict

    with open(path, "w") as fh:
        json.dump(asdict(report), fh, indent=2)
    print(f"\nwrote {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep LLM move-selection candidates.")
    parser.add_argument("--models", nargs="+", default=["dry-run-model"], help="model id(s)")
    parser.add_argument(
        "--renderers",
        nargs="+",
        default=["pip_list"],
        choices=["ascii", "pip_list", "moves_only", "position_id"],
    )
    parser.add_argument("--templates", nargs="+", default=["terse"], choices=["terse", "coach"])
    parser.add_argument(
        "--formats", nargs="+", default=["index_text"], choices=["index_text", "structured"]
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=["off"],
        choices=["off", "low", "medium", "high"],
        help="reasoning effort per candidate (coarse off/on first)",
    )
    parser.add_argument("--positions", type=int, default=10, help="size of the frozen scoring set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-usd", type=float, default=float("inf"), help="hard cost cap")
    parser.add_argument("--max-calls", type=int, default=None, help="hard model-call cap")
    parser.add_argument("--max-reprompts", type=int, default=2)
    parser.add_argument("--scorer", default="reference", choices=["reference", "gnubg"])
    parser.add_argument(
        "--mode",
        default="agreement",
        choices=["agreement", "equity_loss"],
        help="reference scoring mode",
    )
    parser.add_argument("--checkpoint", default=None, help="value-net checkpoint for equity-loss")
    parser.add_argument("--cache", default=None, help="JSONL response cache path")
    parser.add_argument("--json", default=None, help="write the report as JSON to this path")
    parser.add_argument("--live", action="store_true", help="use the real OpenRouter API")
    return parser.parse_args()


if __name__ == "__main__":
    main()
