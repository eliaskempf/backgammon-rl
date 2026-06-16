# WP2 — n-ply Expectimax Lookahead (chance nodes)

**Status:** not started · **Depends on:** WP0 · **Parallel with:** WP1, WP3, WP4

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
