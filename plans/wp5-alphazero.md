# WP5 — (Optional) AlphaZero-Style: Policy Head + Chance-Node MCTS + Self-Play

**Status:** not started · **Depends on:** WP0 · **Last / lowest priority**

**Branch:** this session creates branch `wp5-alphazero` off the latest `main` itself before any code (CLAUDE.md §10); it never asks the human to manage branches.

A "modern RL" capstone demo. **Explicitly overkill for the strength bar** — a WP1 net at
0-ply, or WP2 at 1-2 ply, already beats a non-pro human, and the field never adopted MCTS
for backgammon because the dice branching factor makes deep search far less rewarding than
in chess/Go. Build this for the demonstration value, not for strength. Reuses everything
below the agent layer.

---

## Scope

- **Policy head:** extend the net to `f(afterstate) -> (policy_prior, outcome_vector)`.
  Add as a second head; don't disturb the value head or the WP0 net interface for value
  agents (they ignore the policy head). Define the policy target space carefully — over
  the variable legal-move set; mask to legal moves.
- **Chance-node MCTS** (`bgrl/agents/mcts_agent.py`): tree alternates decision nodes
  (mover's move) and **chance nodes** (dice roll). Use PUCT descent at decision nodes;
  at chance nodes, **sample one roll per simulation** (cheap, higher variance) rather
  than expanding all 21 (expensive) — document the choice. Leaf evaluation uses the net
  (`equity`/outcome + policy prior), NOT rollouts — net-as-leaf is why AZ works and why
  long noisy backgammon rollouts are avoided.
- **Self-play training** (`bgrl/training/alphazero.py`): for each move run N sims; the
  **visit-count distribution is the improved policy target π**; sample a move from π with
  temperature; at game end label every position with outcome z (mover POV). Train: policy
  head → π (cross-entropy), value head → z (MSE / appropriate outcome loss). This reuses
  the WP0 env + the WP1 algorithm-agnostic loop scaffolding where possible (the self-play
  generator differs because move selection runs MCTS, but the env contract is unchanged).
- Intuition to encode in comments: **MCTS is a policy-improvement operator** — search
  sharpens the net's policy, the net distills the sharpened policy back into weights.

## Design constraints

- Must not require changes to `bgrl/env` or the WP0 value-agent interface. If it does,
  escalate — the contract was wrong.
- GPU genuinely helps here (batched leaf evals through the tree) unlike TD — but keep it
  optional and CPU-runnable for small N. bgsage's CPU-parallel multi-ply/rollout is a
  reference for the parallelism split.
- Ply/sim counts are runtime params (web latency vs. strength).

## Acceptance criteria

- Trains via self-play from a value-net init (or scratch) and reaches at least WP1-level
  strength (beating it is a bonus, not required — the point is a working AZ pipeline).
- Chance-node handling + PUCT are unit-tested (correct dice weighting if expanding;
  correct sampling distribution if sampling).
- Reuses the WP0 contract unchanged; selectable as a web opponent.
- `uv run pytest` green, `uv run ruff check` clean.

## Reality check

If time is short, **skip this**. WP1+WP2 deliver the stated goal. WP5 is the
"and here's the AlphaZero version" flourish that's cheap *given* the layered foundation,
but adds real complexity (policy target over variable actions, MCTS correctness, more
compute) for no strength gain at this bar.
