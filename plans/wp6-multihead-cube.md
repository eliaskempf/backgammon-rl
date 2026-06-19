# WP6 — Full 5-Head Outcome Training + Doubling Cube (money play)

**Status:** not started · **Depends on:** WP0 (contract) + WP1 (TD core it extends) + WP2 (equity/expectimax it reduces through) · **Parallel with:** WP5 (independent)

**Branch + worktree:** this session creates branch `wp6-multihead-cube` off the latest
`main` **in its own git worktree** and does all WP work only there (CLAUDE.md §10). Each WP
runs in a separate worktree so branches never collide on a shared checkout; the session
never asks the human to create, switch, merge, or share worktrees, and never edits another
WP's worktree.

Two coupled upgrades that finally use the cube-ready 5-vector the contract reserved from
WP0:

1. **Train all five outcome heads** `[p_win, p_win_gammon, p_win_bg, p_lose_gammon,
   p_lose_bg]`, not just `p_win`. v1 generates the labels every game and throws them away
   (`TDLambda.episode_end` discards `outcome.kind`); this turns them on.
2. **Add the doubling cube under money play** — cube state, cubeful money equity
   (Janowski), and a cube-decision surface on the agent. Cube decisions *consume* the
   gammon heads, which is why (1) is a prerequisite and the two ship together.

**Match play (score to 7/11, MET, Crawford) is explicitly out of scope — deferred to
WP7.** This WP delivers a cube-capable money-game agent; WP7 reuses the same net and adds
the score layer.

---

## Execution model — fully CC-implemented (no self-implementation gate)

Unlike WP1, there is **no `TODO(human)` / self-implementation phase**. CC implements the
multi-head TD math, the cubeful equity, and the cube decision logic end to end. The human
already wrote and reviewed the *scalar* TD core in WP1; this WP extends that reviewed code
to the vector case, so the explicit formulas below are CC's to implement directly.

---

## Part A — Train all five heads

### A1. The latent bug this also fixes (state it in the PR)

`ValueAgent.act` feeds the **full** 5-vector into `equity()` (`value_agent.py:52-53`), but
v1 trains only head 0. The other four heads are untrained — and because the net ends in
`Sigmoid()` with default init, they sit near **0.5, not 0** (verified empirically: a fresh
net outputs ~`[0.59, 0.55, 0.52, 0.40, 0.61]`). So today's `equity` is
`2·p_win−1 + (p_win_g + p_win_bg − p_lose_g − p_lose_bg)`, where the trailing term is
**position-dependent uncalibrated noise of the same order as the `p_win` signal it ranks
on** — i.e. v1 move selection is silently perturbed. The display path dodges it
(`win_probs` reads only head 0); `act` does not. Training the heads removes the leak as a
side effect. Pin a regression test: an all-zeros gammon/bg net must make `equity` ==
`2·p_win−1`.

### A2. The multi-head TD(λ) update (the real work — and the correctness locus)

Generalize the scalar negamax recursion in `bgrl/training/td_lambda.py` to the 5-vector.
This is the multi-head analogue of WP1's sign-flip; it is **the** thing to test.

- **Value is now a 5-vector.** `_value` returns the full `self.net(x)` (shape `(5,)`), not
  `self.net(x)[..., _PWIN]`. Delete `_PWIN`.
- **Bootstrap target = the perspective-flip permutation of the successor.** The scalar
  complement `1 − f(a_{t+1})` generalizes to a fixed linear involution. With
  `succ = V(a_{t+1})` (the opponent's afterstate value, opponent POV), the mover-POV target
  for `V(a_t)` is:

  ```
  target[0] (p_win)     = 1 − succ[0]      # win ⟺ opponent loses; complement on head 0
  target[1] (p_win_g)   = succ[3]          # win-gammon ⟺ opp lose-gammon
  target[2] (p_win_bg)  = succ[4]          # win-bg     ⟺ opp lose-bg
  target[3] (p_lose_g)  = succ[1]          # lose-gammon ⟺ opp win-gammon
  target[4] (p_lose_bg) = succ[2]          # lose-bg     ⟺ opp win-bg
  ```

  Head 0 keeps the `1−x` complement; the win/lose **gammon** and **backgammon** heads
  *swap*. Equivalently: 5 independent TD(λ) channels, head 0 a standard negamax channel,
  heads {1,3} a 2-cycle, heads {2,4} a 2-cycle.
- **Per-head eligibility traces.** A 5-vector value has a Jacobian, not a scalar gradient:
  maintain **one trace set per head** (≈5× trace storage). The TD error `δ` is a 5-vector;
  component `δ_k` scales head `k`'s trace. The cross-ply trace carry must respect the
  win/lose head pairing above (the swap), exactly as the scalar trace respected the sign
  flip. **This pairing is the primary correctness test.**
- **Terminal target now uses `outcome.kind`** (today discarded). `episode_end` regresses
  the last valued afterstate `a_{T-1}` (its to-move player is the **winner**) toward the
  realized cumulative one-hot from the winner's POV:

  ```
  target = [1, kind ≥ GAMMON, kind ≥ BACKGAMMON, 0, 0]
  ```

  (Single win → `[1,0,0,0,0]`; the loser's heads are 0 because the winner did not lose.)
- Keep the deferred-online structure, the terminal-afterstate skip, and the `gamma`
  generality from WP1 unchanged — only the value, target, traces, and terminal label
  become vector-valued.

### A3. Tests (generalize `tests/training/test_td_lambda.py`)

- The forward-view / Monte-Carlo equivalence checks become **per-head**: at `λ=1`, each
  head regresses toward its realized terminal indicator; the general-`λ` check pins the
  head-pairing trace carry (the analogue of the scalar sign-flip check).
- A two-ply hand-worked episode with a known gammon terminal, asserting all five heads
  move in the right direction.
- The `equity == 2·p_win−1` regression test from A1.

### A4. Calibration diagnostic (efficiency, not correctness)

Add a self-play diagnostic: predicted vs realized gammon/backgammon rate (reliability
curve) per head, logged during training. This is how you *know* the heads converged rather
than collapsed to ~0, and it's the signal for whether the optional rollout fine-tuning in
A5 is worth it. Expect `p_win` fastest, gammon a few× slower, **backgammon slowest and
noisiest** — that ordering is normal, not a bug.

### A5. Optional, gated: supervised rollout fine-tuning of the gammon/bg heads

**Trigger — do not build by default.** Only if the A4 diagnostic shows the backgammon (or
gammon) head **collapsed to ≈0 / its reliability curve is flat** — i.e. online TD never
accumulated enough rare-event signal. Otherwise skip entirely: the TD-only heads clear the
strength bar (TD-Gammon dropped backgammon outright; gnubg/XG treat it as a minor
correction).

**Method — decouple accuracy from waiting on rare online-TD terminals.** Generate
low-variance supervised targets and fine-tune on top of the Part-A net (a fine-tuning pass,
**not** a replacement for the TD training):

- Sample positions, **biased toward late-game / bearoff / deep-contact** where gammon and
  backgammon are actually decided — that's where the discriminative signal lives.
- From each, run **truncated multi-outcome rollouts** with the current net as the leaf
  policy; average the realized 5-vector to get a low-variance target for *all five heads at
  once*.
- Optional variance reduction: use the value net's own evaluation as a **control-variate
  baseline** in the rollout (gnubg-style) — this sharpens the rare heads fastest.
- Supervised-fit the heads to these targets (per-head sigmoid / cross-entropy).

**Acceptance (only if built).** The bg/gammon reliability curve from A4 measurably improves
vs the TD-only baseline, with **no regression** in `p_win` calibration or in playing
strength (win rate vs the WP2 / pubeval reference under common random numbers).

**Cost note.** Rollouts are the expensive part (many leaf evals per sampled position) — keep
the sampled set small and keep the whole pass behind the A4 trigger so it never runs by
default.

---

## Part B — Doubling cube (money play)

### B0. The load-bearing design decision: **train cubeless, apply the cube analytically**

Self-play used to *train* the net stays **cubeless, played to completion**, so every
terminal yields an honest 5-outcome label. Do **not** introduce dropping into training
self-play — a passed double has no played-out gammon outcome, which would corrupt the
multi-head labels. The cube is an **evaluation/agent capability**, layered on top via
formulas; the net keeps predicting the cubeless game-outcome distribution. (This is exactly
how gnubg factors it, and it's why Part A's training loop barely changes.)

### B1. Cube state

- Thread cube `(value, owner)` through `EnvState` for a played game (the slots are already
  reserved in the encoding and in `CubeContext`). `owner = None` = centered.
- Money cube: legal to double only by the player on roll who owns or shares the cube;
  doubling passes ownership and doubles `value`.

### B2. Cubeful money equity (Janowski)

- Extend `bgrl/nets/equity.py` with a **cubeful** path: given the cubeless 5-vector +
  `CubeContext` (value, owner) + a cube-life coefficient `x`, return cubeful money equity.
  Implement the standard Janowski dead/live-cube interpolation (`x` ≈ 2/3 is the usual
  default; expose it). The existing cubeless `equity` stays as the `x`-irrelevant / centered
  baseline and the `take/drop` reference.
- This consumes the gammon heads directly — take points and the doubling window shift with
  gammon rate, so Part A must land first.

### B3. Cube decisions: a pure module + a shared evaluator + a thin agent hook

The cube decision is **agent-independent** — don't conflate it with the checker-play agent.
Three separable pieces:

- **`CubeDecider` (pure, agent-independent).** Double/take/pass is a pure function of the
  position's cubeless outcome distribution + cube context: `decide(outcome_vector, cube) ->
  action` (Janowski windows, redouble/too-good handling). Lives in the cube/equity module,
  shared by every agent; it does not wrap or belong to any agent.
- **A shared `PositionEvaluator`, not an agent.** The decider's only input is an outcome
  distribution, produced by `evaluate_outcome(state, on_roll) -> 5-vector` (cubeless, mover
  POV). The raw net is the 0-ply evaluator; an expectimax wrapper is the n-ply evaluator.
  The checker selector and the cube decider read from the **same** evaluator; neither wraps
  the other. Cube decisions are made **pre-roll**, so ≥1-ply (average over the dice) is the
  natural floor for a meaningful on-roll distribution — a depth knob, *not* an agent choice.
- **A thin agent hook (protocol plumbing only).** The play loop must ask *someone* "do you
  double?", so add `should_double(state, cube) -> bool` / `should_take(state, cube) -> bool`
  to the agent interface, **defaulting to delegate to `CubeDecider`** over the agent's
  evaluator. The hook exists so an agent *can* override (e.g. the LLM's own cube policy, or a
  deliberately weak one), not because the logic is agent-specific. Existing agents
  (`RandomAgent`, LLM, expectimax) satisfy it unchanged via the default. This is the only
  addition to the WP0 agent contract; `act` is untouched (extend deliberately — we own the
  contract now, not in a parallel fan-out).

### B4. Money session loop + gnubg cross-check

- A money-game play/eval loop where games can be doubled, taken, or dropped (stakes scale;
  **no match score** — that's WP7). Training does not use this loop (B0).
- Validate cube equities and double/take/pass decisions against **gnubg** on a set of
  reference positions (we already drive gnubg for WP3 analysis). This is the acceptance
  oracle for the cube, the way pubeval was for WP1.

---

## Do the training parameters need to change?

Short answer: the **data source and loop are unchanged**; expect a modest net-size bump and
more total training, no new cube/match hyperparameters.

- **More games?** Not a new source — the labels are free in the games you already run. But
  the sparse heads need **more total positions** to calibrate: `p_win` converges first,
  gammon a few× later, backgammon slowest and stays lowest-precision (it's ~1–2% of games,
  so the head can low-loss-collapse to ≈0). Two responses: (a) accept bg as a small
  correction — TD-Gammon dropped it entirely, gnubg/XG treat it as minor; (b) optional
  **rollout fine-tuning** for gammon/bg — truncated multi-outcome rollouts give a
  low-variance target for all five heads at once, decoupling accuracy from waiting on rare
  online-TD terminals. Gate (b) behind the A4 diagnostic; don't build it unless calibration
  demands it — scoped as the optional **A5** above.
- **Bigger net?** Modest. The heads share the trunk, so marginal capacity is small, but the
  trunk must now encode gammon-relevant structure (loser containment / how trapped the
  loser is), not just the race. Recommend bumping `hidden` 64 → ~128; richer input features
  (Tesauro's extras) would help gammon discrimination more than raw width, but that's
  optional. Not a dramatic scale-up.
- **Hyperparameters?** `λ` unchanged (0.7); `lr` unchanged or marginally lower (the trunk
  now receives five heads' gradients, though the sparse heads contribute little). Output
  stays per-head sigmoid; note independent sigmoids do **not** enforce cumulative
  monotonicity `p_win ≥ p_win_g ≥ p_win_bg` — fine for gnubg-style nets, don't rely on it
  downstream. Resource note, not an HP: ~5× eligibility-trace memory.
- **Cube?** Adds **no net-training HP** — decisions are analytic (Janowski). The only knob
  is the cube-life coefficient `x`, a formula constant, not a learned parameter.

---

## Acceptance criteria

- All five heads train; the A4 calibration diagnostic shows `p_win` and gammon heads
  tracking realized rates (bg may stay noisy — documented, not failed).
- Per-head MC/general-`λ` equivalence tests green (the head-pairing trace carry is pinned).
- `equity == 2·p_win−1` regression test green for an all-zeros-gammon net (leak fixed).
- A cube agent makes double/take/pass decisions that **match gnubg** on a reference position
  set within tolerance; cubeful money equity validated against gnubg.
- Existing agents still satisfy the extended agent interface unchanged (safe defaults).
- No changes to `bgrl/env` move generation. `uv run pytest` green, `uv run ruff check` clean.

## Notes / open questions for the planner

- Confirm `EnvState` can carry cube state without breaking WP1/WP2/WP3 callers (the slots
  are reserved; verify nothing assumes a centered cube structurally).
- The cube decider consumes a shared `PositionEvaluator` (0-ply net or n-ply expectimax),
  **not** a specific agent — the open choice is the *evaluation depth*, tuned so cube
  decisions match gnubg in the B4 cross-check (≥1-ply is the natural floor, since cube
  decisions are pre-roll). This is an evaluator-depth choice, not an agent-wrapping choice.
- `x ≈ 2/3` default for Janowski is a starting point; the gnubg cross-check (B4) is what
  tunes it.

---

## Implementation notes (as built)

Branch `wp6-multihead-cube` in its own worktree. `uv run pytest` and `ruff` green
(the live-gnubg tests run here, since gnubg is installed).

**Part A.**
- A0 — one canonical `flip_outcome` + `FLIP_PERM`/`FLIP_SIGN` in `bgrl/nets/equity.py`,
  reused by the TD target *and* the cube evaluator so they cannot drift.
- A2 — multi-head `TDLambda`: full 5-vector `_value_vec`, per-head traces
  `list[list[Tensor]]`, the bootstrap target is `flip_outcome(V(a_t))`, and the trace
  carry is `λ·γ·FLIP_SIGN[k] · e_{t-1}[FLIP_PERM[k]] + ∇f_k`. **Confirmed: head 0 keeps
  the scalar `-λγ` carry; the swapped gammon/bg pairs carry `+λγ`** — pinned by the
  per-head MC/λ equivalence test at `λ=0.5` with forced GAMMON/BACKGAMMON outcomes.
  `episode_end` now consumes `outcome.kind`.
- A2b — `bgrl/training/td_lambda_exercise.py`: non-blocking self-study scaffold with
  `TODO(human)` on the three learning-critical pieces (bootstrap flip, trace carry,
  terminal target). Not imported anywhere; the real `td_lambda.py` is what trains.
- A4 — `bgrl/training/calibration.py` (per-head predicted-vs-realised + ECE), wired into
  `scripts/train.py` via `--calib-games` (own `calibration.csv` + W&B `cal/*`).
- A6 — hidden width is already a `train.py --hidden` knob (sweep `{64,128}` operationally);
  `scripts/aggregate_runs.py` now surfaces the win-gammon/bg head ECEs for selection.

**Part B.**
- B2 — `cubeful_equity` (Janowski), `CubeAccess`, `win_loss_magnitudes`, `cube_access`,
  `DEFAULT_CUBE_LIFE = 2/3` in `equity.py`.
- B3 — `CubeDecider`/`CubeAction`/`TakeAction` in `bgrl/nets/cube.py` (pure); the
  `PositionEvaluator` protocol + 1-ply `NetEvaluator` (Adapter A) + dispatchers
  (`wants_to_double`/`wants_to_take`/`evaluator_for`) in `bgrl/agents/cube_policy.py`.
  **Deviation:** the agent hook is a *separate* `CubeCapable` protocol, not an extension
  of `Agent`, because `isinstance(_, Agent)` is relied on widely and widening it would
  break existing agents. The n-ply vector evaluator (Adapter B) is deferred behind the B4
  trigger — 1-ply already matches gnubg.
- B4 — `bgrl/money.py` (`play_money_game` + `MoneyGameResult`/`CubeEvent`); the cube is a
  **loop-local `CubeContext` applied analytically and never fed into the net encoding**
  (B0: net stays cubeless; `EnvState`'s cube slots stay reserved/zero). `money_game_to_mat`
  emits `Doubles =>`/`Takes`/`Drops` — **verified to round-trip through gnubg**.
  `analyse_cube` extracts gnubg's `probs`/`nd-cubeful-eq`/`dt-cubeful-eq`; feeding gnubg's
  own probs into our formula reproduces its no-double equity to ~0.002 at `x = 2/3`, so
  **the 2/3 default is confirmed, no tuning needed**.
- Shared refactors (DRY): `WEIGHTED_ROLLS`/`weighted_rolls` moved to `bgrl/env/dice.py`
  and `outcome_to_vector` to `equity.py`; `ExpectimaxAgent` now consumes both.

**Open / deferred:** Adapter B (n-ply vector cube evaluator) only if a future cross-check
shows 1-ply disagreeing with gnubg; A5 rollout fine-tuning only if the A4 diagnostic shows
a collapsed bg/gammon head after a real training run.
