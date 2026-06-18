# WP2 — n-ply Expectimax Lookahead (chance nodes)

**Status:** not started · **Depends on:** WP0 · **Parallel with:** WP1, WP3, WP4

**Branch:** this session creates branch `wp2-expectimax` off the latest `main` itself before any code (CLAUDE.md §10); it never asks the human to manage branches.

A pure agent-layer add-on: wrap any `ValueNet` in depth-limited expectimax search over
dice chance nodes. This is what real bots (gnubg/XG) do, and it's simpler and more
effective here than MCTS. The env and training loop are untouched.

---

## Scope

- `bgrl/agents/expectimax_agent.py`: given a `ValueNet` + equity module, evaluate moves
  by looking ahead `n` plies, averaging over all 21 distinct dice rolls at each chance
  node (weighted: 30/36 for the 15 non-doubles at 2/36 each, 6/36 for the 6 doubles at
  1/36 each — verify weighting in code + test).
- Reuses `Env.legal_moves` for expansion at decision nodes. No new env surface needed.
- Ply convention = gnubg's (raw net eval = 0-ply). Document and expose `--plies`.

## Design

- Decision node: enumerate legal afterstates (mover POV), recurse.
- Chance node: enumerate 21 rolls with weights, recurse into opponent's best response.
- Leaf (depth budget hit): `equity(net.evaluate(encode(afterstate)), cube)`.
- Cost grows ~×(legal_moves × 21) per ply — keep it shallow. 0/1/2-ply is the useful
  range; 2-ply is the sweet spot. Beyond that, rollouts would be the move (out of scope).
- Batch leaf evaluations for GPU efficiency where possible.
- Optional candidate pruning (evaluate top-k afterstates at 0-ply before deepening) —
  mirrors gnubg's pruning nets. Nice-to-have, gate behind a flag.

## Acceptance criteria

- Wrapping a WP1 checkpoint at 1-ply measurably **improves win rate vs. the same net at
  0-ply**, evaluated with common-random-numbers (expect a small but real gain — the
  TD-Gammon-lineage result is ~2pp vs. gnubg per added ply at the low end).
- Correct dice-roll weighting (unit-tested against the 36-outcome distribution).
- No changes required to `bgrl/env` or `bgrl/training`. If you need to touch them, the
  WP0 contract was wrong — escalate rather than patch around it.
- `uv run pytest` green, `uv run ruff check` clean.

## Notes

- This agent is also the natural strong opponent for evaluating WP1 and WP4 agents.
- Keep search depth a runtime parameter so the web UI (WP3) can trade strength vs.
  move latency.

---

## Status: search-only cut implemented (2026-06-18, branch `wp2-expectimax`)

Pure agent-layer, no `bgrl/env` or `bgrl/training` changes (acceptance criterion met).

**Delivered**
- `bgrl/agents/expectimax_agent.py` — `ExpectimaxAgent`: negamax over chance nodes,
  `eval_pov(s, depth)` = equity to `s.turn`, each level negates its children (the
  `ValueAgent` sign convention applied recursively). `plies=0` reproduces `ValueAgent`
  move-for-move; gnubg ply convention.
- Exact terminal scoring inside the search via a private `_terminal_equity` that builds
  the cumulative outcome 5-vector and reduces it through the shared `equity()` (so win
  magnitude stays consistent with the net's reduction; never the net on a finished game).
- 21-weighted-roll table (doubles 1/36, non-doubles 2/36, listed once as `a<=b`).
- Optional `top_k` candidate pruning (off by default — exact search) + a per-move
  transposition cache.
- `tests/agents/test_expectimax_agent.py` (15 tests), `--plies`/`--top-k` on
  `scripts/eval_agent.py`, and `scripts/bench_expectimax.py` (win-rate + ms/move sweep).

**Decisions / deviations**
- Terminal scoring is magnitude-aware (+/-1/2/3), reusing `equity()`, per CLAUDE.md §5
  (faithful gammon/backgammon). Single loss is the all-zeros vector (the -1 is the implied
  `p_lose`); pinned in a test.
- A forced pass (no legal reply for a roll) consumes a ply: recurse on
  `replace(s, turn=opponent)` at `depth-1`; pinned in a test.
- Pruning + transposition cache were pulled into this cut (beyond the original "exact only"
  first cut) specifically to make the deep-ply latency sweep measurable. The acceptance
  check and all correctness tests use exact search; pruning is an approximation used only
  in the benchmark.

**Feasibility finding (per-move latency, h64 net, CPU, avg branch ~18)**
- 0-ply ~0.3 ms; 1-ply exact ~313 ms; 1-ply top-8 ~114 ms; 2-ply top-8 ~13.6 s/move;
  3-ply infeasible (extrapolates to tens of min/move). The wall is pure-Python
  `legal_moves` (~28k calls/move at 2-ply), not just net overhead — global frontier
  batching cuts only the net portion (~half), so the next real lever for 2/3-ply is a
  faster (vectorised/native) move generator, not more batching. Full pubeval matches were
  run only at 0/1-ply; 2/3-ply matches are not feasible at this implementation's speed.
