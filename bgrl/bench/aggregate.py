"""Aggregate per-host benchmark JSON into a comparison table + a decision block.

Prints Markdown to stdout (the human pastes the relevant part into DECISIONS.md);
it deliberately does not write DECISIONS.md itself.
"""

from __future__ import annotations

from pathlib import Path

from .schema import read_json

# Throughput thresholds (aggregate games/sec on the cluster CPU node).
_GREEN = 200.0  # ~1e6 games in ~1.4 h
_ACCEPTABLE = 50.0


def load_results(paths: list[str | Path]) -> list[dict]:
    return [read_json(p) for p in paths]


def _cell(x: object) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)


def _md_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def comparison_table(results: list[dict]) -> str:
    headers = [
        "tag",
        "host",
        "cores P/L",
        "workers",
        "games/s",
        "pos/s",
        "afterstates/s",
        "B",
        "scal.eff",
        "encode/s",
        "net_eval%",
    ]
    rows: list[list] = []
    for r in results:
        fp = r["fingerprint"]
        env = r["env"]
        nev = env.get("net_eval_share") or {}
        frac = nev.get("net_eval_fraction")
        net_pct = frac * 100 if frac is not None else None
        rows.append(
            [
                r.get("tag"),
                fp.get("hostname"),
                f"{fp.get('cores_physical')}/{fp.get('cores_logical')}",
                env.get("workers"),
                env.get("games_per_sec"),
                env.get("legal_moves_calls_per_sec"),
                env.get("afterstates_per_sec"),
                env.get("mean_afterstates_per_position"),
                env.get("scaling_efficiency"),
                env.get("encode_per_sec"),
                net_pct,
            ]
        )
    return _md_table(headers, rows)


def decision_block(results: list[dict]) -> str:
    best = max(results, key=lambda r: r["env"]["games_per_sec"])
    env = best["env"]
    fp = best["fingerprint"]
    g = env["games_per_sec"]
    verdict = (
        "GREEN — keep the DFS env"
        if g >= _GREEN
        else ("ACCEPTABLE" if g >= _ACCEPTABLE else "INVESTIGATE move-gen")
    )
    lines = [
        f"- **Env keep-vs-replace:** best aggregate **{g:,.0f} games/s** on `{best.get('tag')}` "
        f"({fp.get('cores_physical')} phys cores, {env['workers']} workers) → **{verdict}** "
        f"(green ≥ {_GREEN:.0f}, acceptable ≥ {_ACCEPTABLE:.0f}). Correctness is the hard gate "
        f"(differential oracle + property suite already green).",
        f"- **Workers:** scaling efficiency at {env['workers']} workers = "
        f"{env.get('scaling_efficiency')}; set self-play parallelism near the physical-core count.",
    ]
    nev = env.get("net_eval_share")
    if nev:
        lines.append(
            f"- **CPU vs GPU:** single-worker per-position split — move-gen "
            f"{nev['movegen_fraction'] * 100:.0f}%, encode {nev['encode_fraction'] * 100:.0f}%, "
            f"net-eval {nev['net_eval_fraction'] * 100:.0f}% (B={nev['branching_factor_B']}). "
            f"Online TD's largest batch is ~B (≪ the tiny-net GPU/CPU crossover), so **CPU**; "
            f"GPU is revisited only for WP2 expectimax / WP5 MCTS."
        )
    return "## Throughput Findings\n\n" + "\n".join(lines)


def render(results: list[dict]) -> str:
    return comparison_table(results) + "\n\n" + decision_block(results)
