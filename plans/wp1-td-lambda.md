# WP1 — TD(λ) Baseline + Algorithm-Agnostic Training Loop

**Status:** not started · **Depends on:** WP0 (frozen contracts) · **Parallel with:** WP2, WP3, WP4

**Branch:** this session creates branch `wp1-td-lambda` off the latest `main` itself before any code (CLAUDE.md §10); it never asks the human to manage branches.

Delivers the first real agent and, more importantly, the **algorithm-agnostic training
loop** that later algorithms plug into. The loop must not know it's running TD.

---

## Execution model — SELF-IMPLEMENTED TD core (read first)

The human is implementing the TD(λ) math themselves to test their understanding. This
changes how this WP is built. Run it in three explicit phases; **do not collapse them.**

### Phase A — scaffold only (CC)
CC builds the class structure, method signatures, type hints, tensor shapes, and
docstrings describing *what* each piece must satisfy, plus `# TODO(human)` markers at the
exact spots the human implements. **Leave hollow:** the eligibility-trace update, the
TD-target construction, the gradient application, and the per-step net update.

Commenting rule (important): comments may describe **the invariant the code must
satisfy**, not **the line that satisfies it**.
- Allowed: "compute the TD error between successive afterstate values from the mover's
  POV; mind the sign flip between plies."
- NOT allowed: revealing the implementation, e.g. `delta = v_next - v_current`, or
  "use exactly this formula …".
The everything-else (loop wiring, net plumbing, episode bookkeeping, checkpointing) CC
implements normally.

### Phase B — human implements
Human fills the `TODO(human)` blocks. CC does not touch these.

### Phase C — explicit review gate (CC) — a distinct, named step
CC reviews the **human's filled-in code** for correctness + efficiency and proposes
adjustments *with* the human. **CC does not silently rewrite** — it presents findings,
discrepancies get discussed, not auto-fixed.

Review must substantively check:
- **Correctness:** trace decay (λ·γ); target = reward + γ·V(next afterstate); terminal
  bootstrap uses the *actual outcome*, not the net estimate; perspective/sign-flip
  handling between plies (the classic bug locus); trace reset at episode boundaries;
  gradient flows to the intended parameters only.
- **Efficiency:** no per-step Python loops where a vectorized op works; traces updated
  in place without reallocation; afterstate eval batched; no autograd graph retained
  across the episode; no needless CPU↔GPU transfers.

**Optional stronger variant:** run Phase C in a *fresh* CC session that sees only the
human's filled-in code + the WP0 contract, NOT the Phase-A scaffolding rationale. This
removes the "reviewer already knows the answer it steered toward" effect and is a cleaner
test of the human's understanding. Use if a more honest review is wanted.

---

## Scope

- `bgrl/training/loop.py`: a self-play game generator + driver that is independent of
  the update rule. It produces games (sequences of (state, dice, chosen move,
  afterstate, ...) ) and hands trajectories/steps to a `Trainer` via the agent's
  `observe_step` / `observe_game_end` hooks (or an explicit Trainer protocol — pick one
  and document; the hook approach keeps the loop thin).
- `bgrl/agents/td_agent.py`: afterstate value agent. Selection = argmax equity over
  legal afterstates (greedy; dice provide exploration, no ε needed — this is the
  TD-Gammon result, cite it in a comment).
- `bgrl/training/td_lambda.py`: TD(λ) update with eligibility traces, online,
  undiscounted, reward = outcome at terminal (0 elsewhere). v1 trains `p_win` head;
  leave the other outcome heads wired but inactive (or train them from the
  single/gammon/backgammon outcome label if cheap — preferred, since WP0 records it).
- `scripts/train.py`: thin CLI — config (hidden units, λ, lr, #games, save cadence,
  seed), runs the loop, writes checkpoints via WP0 checkpoint spec.
- `scripts/eval_agent.py`: win-rate of checkpoint vs. RandomAgent and vs. a previous
  checkpoint, using common-random-numbers (WP0 §7) for low variance.

## Design constraints

- The training loop must call only WP0 contract surfaces (`Env.legal_moves`,
  `Agent.act`, `Outcome`). No TD-specific assumptions in the loop itself.
- Selection consumes `equity(...)`, never raw net outputs (WP0 §5).
- Perspective invariant respected when forming TD targets (the value of the afterstate
  from the mover's POV; handle the sign flip between plies carefully — this is the
  classic bug locus).
- GPU optional. Net is tiny; CPU self-play is expected to dominate. Don't force CUDA.

## Acceptance criteria

- Trains from random init and **beats RandomAgent decisively** (e.g. >90% win rate)
  within a modest number of games; win rate vs. a frozen earlier checkpoint trends up.
- Reproducible: same seed → same training curve.
- `uv run pytest` green (include a smoke test: a few hundred games of training runs and
  loss/eval moves in the right direction), `uv run ruff check` clean.
- Loop is demonstrably reusable: a trivial second "trainer" (even a no-op) can be
  swapped in without touching `loop.py`.

## Notes / pitfalls

- Don't bake `episode == single cubeless game` into the loop's interface (WP0 §5).
- Keep self-play move-gen on the hot path fast (WP0 benchmark informs this); consider
  vectorized/batched afterstate eval even at 0-ply.
- This WP's strength bar (beat a non-pro human) is essentially already met by a
  well-trained net here; WP2 only widens the margin.

---

## Phase A — implemented (decisions & deviations)

**Status:** Phase A complete on branch `wp1-td-lambda`. `ruff check` clean;
`pytest` 71 passed, 1 skipped (the smoke test, gated until the core is filled).
The TD core is hollow and awaits the human (Phase B).

**Decisions (confirmed with the user):**
- **TD target = `p_win` head only** (index 0). The 5-vector shape is preserved;
  the gammon/backgammon heads stay wired but untrained.
- **Guided scaffold.** `td_lambda.py` provides the eligibility-trace storage (one
  zero tensor per net parameter), a carry-over slot, and a differentiable value
  helper `_value(afterstate, perspective)`; the human writes only `TDLambda.step`
  and `TDLambda.episode_end` (trace recurrence, TD error + perspective sign-flip,
  terminal bootstrap, manual trace-scaled weight update + reset). Comments there
  state invariants only — never `reward + γ·V(next)` or `λ·γ`.

**Files created:** `bgrl/training/{__init__,loop,evaluate,td_lambda}.py`,
`bgrl/agents/td_agent.py`, `scripts/{train,eval_agent}.py`,
`tests/training/{test_loop,test_evaluate,test_td_training_smoke}.py`,
`tests/agents/test_td_agent.py`.

**Deviations / notable choices:**
- **CRN uses seed-paired generators, not record-then-replay.** Swapping seats
  changes the trajectory, so the second game of a pair may need more rolls than
  the first produced → `ReplayDiceSource` exhausts. `play_match` instead seeds two
  generators identically per pair (seed drawn from the match `rng`), giving a
  shared on-demand dice stream that survives length divergence.
- **RNG separation for reproducibility.** `scripts/train.py` splits one seed via
  `np.random.default_rng(seed).spawn(2)` into independent train/eval streams (eval
  must not consume training dice), and calls `torch.manual_seed(seed)` so net
  initialisation is reproducible too. A Phase-A test asserts eval draws don't
  perturb the training dice.
- **Smoke test self-activates.** `test_td_training_smoke.py` is `@pytest.mark.slow`
  + `skipif(not _td_core_ready())`, where `_td_core_ready()` probes `TDLambda.step`
  for `NotImplementedError`. It turns green automatically once Phase B lands; no
  manual un-skip to forget. Bar: >90% vs RandomAgent after ~1000 games, reproducible.
- **Parallel-safety (§7a).** `TDAgent` is imported via `bgrl.agents.td_agent`
  directly; `bgrl/agents/__init__.py` is left untouched. Exporting `TDAgent` from
  the package `__init__` is a deferred post-merge one-liner.
- **`td_agent.py` has zero `torch` imports** — pure wiring, so no TD math can leak
  out of `td_lambda.py`.

**Next:** Phase B (human fills the two `TODO(human)` bodies), then Phase C (CC
reviews the human's code against the correctness/efficiency checklist above —
discussing, not auto-fixing — and the smoke test must pass).
