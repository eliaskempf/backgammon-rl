# Backgammon RL — Project Guide (CLAUDE.md)

This file is the entry point for any Claude Code session working on this repo.
Read it fully before starting a work package. Each work package has its own
detailed plan under `plans/`. **Do not start WP1+ until WP0's contract is frozen
and reviewed.**

---

## 1. Goal

Build an illustrative, modular RL pipeline for backgammon that:

1. Trains an agent (starting from TD(λ) self-play) strong enough to beat a decent
   but non-professional human player. This is a *modest* bar — a single afterstate
   value net at 0-ply already exceeds it (TD-Gammon-lineage results reach ~42-45%
   win rate vs. GNU Backgammon, a top bot).
2. Demonstrates a clean, algorithm-agnostic training loop and agent interface, so
   multiple RL algorithms of increasing complexity can be swapped in:
   **TD(λ) → + n-ply expectimax lookahead → (optional) AlphaZero-style MCTS.**
3. Provides an LLM baseline agent (via OpenRouter) with a prompt/format refinement
   harness.
4. Serves a web UI to play against any trained checkpoint (and the LLM agent), and
   serializes played games into a format GNU Backgammon can import and analyze, so
   agent moves can be compared against gnubg's evaluation.

## 2. Non-goals / explicit scope cuts (v1)

- **Cubeless, single games only** for v1. No doubling cube logic, no match play.
  BUT: every interface must be designed so the cube, gammon/backgammon scoring, and
  match play can be added later **without breaking the contract**. See §5.
- Not chasing SOTA strength. The SOTA (eXtreme Gammon, GNU Backgammon) is "great
  eval net + shallow expectimax + rollouts" — we deliberately reproduce the cheap,
  proven core, not the full commercial product.

## 3. Tech stack & conventions

- **Python 3.13.**
- **uv** for environment + dependency management (`uv venv`, `uv add`, `uv run`).
  Lockfile committed.
- **pytest** for tests. **ruff** for lint + format (ruff format, not black).
- **Type hints everywhere.** Run `ruff check` and `pytest` clean before any handoff.
- Production-quality code: no dead code, clear module boundaries, docstrings on
  public interfaces. Prefer pure functions for game logic.

### Repo layout

```
bgrl/                  # the package — ALL modular code, classes, logic lives here
  env/                 # board state, move generation, afterstate enumeration, encoding
  agents/              # Agent interface + implementations (random, td, mcts, llm, ...)
  training/            # algorithm-agnostic training loop + per-algorithm trainers
  nets/                # value/policy network(s) + equity reduction module
  serialization/       # checkpoint I/O + gnubg/.mat/.sgf export + position/match IDs
  web/                 # web server + UI glue
  llm/                 # OpenRouter client, prompt templates, refinement harness
scripts/               # executable entrypoints that LEVERAGE the package
  (train.py, play.py, benchmark_env.py, eval_vs_gnubg.py, refine_llm.py, ...)
plans/                 # per-work-package plan files (wp0..wp5)
tests/                 # pytest suite mirroring bgrl/ structure
```

**Rule:** `scripts/` contains only thin executable wrappers (arg parsing, wiring,
I/O). All reusable logic and classes live in `bgrl/`. If a script grows logic worth
testing, move that logic into the package and import it.

## 4. Architecture — the layered design (the whole point)

Three layers, talking only through frozen contracts (defined in WP0):

```
  env  (game logic, pure)
   │   legal_moves(state, dice) -> [(move_sequence, afterstate), ...]
   │   apply / is_terminal / outcome / encode(state) -> features
   ▼
  agent  (Agent.act(state, dice, legal) -> chosen move)
   │   value/policy nets sit behind agents; selection consumes EQUITY, never raw net out
   ▼
  trainer  (algorithm-agnostic loop generates games; per-algo update rule)
```

The env and training loop must be **as independent of the algorithm choice as
possible**. TD(λ), DQN-afterstate, expectimax, and MCTS all consume the same
`legal_moves` afterstate enumeration. Only the trainer's update rule and the agent's
selection logic differ.

**Afterstate-first.** We never parameterize a policy over the (wonky, variable)
action space. The net is `V(afterstate) -> outcome_vector`; agents enumerate legal
moves, evaluate resulting afterstates, and pick the best by equity. This is why
value-based methods (TD, DQN, expectimax, MCTS-with-net) all swap trivially. AlphaZero
adds a policy head but reuses everything below the agent layer.

## 5. Cube-ready invariants (design now, implement later)

Even though v1 is cubeless single games, **bake these in from WP0**:

- **Net output is always a fixed-length vector**, not a scalar. Target layout (mover's
  POV): `[p_win, p_win_gammon, p_win_bg, p_lose_gammon, p_lose_bg]` (p_lose_single
  implied). v1 may train only `p_win` and zero the rest, but the *shape* is fixed.
- **A separate `equity(outcome_vector, cube_context) -> float` module** does the
  reduction to a scalar. Move selection consumes equity, NEVER raw net outputs. v1's
  cube_context is trivial/centered; the argument exists from day one.
- **State representation reserves cube state** (value, ownership) even if always
  centered=1 in v1.
- **Game outcome recording captures how a game ended** (single / gammon / backgammon),
  even in v1 — so training data and gnubg export are already faithful.
- **Don't hardcode `episode == single game`** so tightly that match-to-N-points can't
  slot in later.
- Encoding follows Tesauro's 198-feature scheme (24 points × 4 units × 2 players +
  bar/off/turn). Reserve extra slots for cube state.

## 6. Perspective invariant (read this twice — sign-flip bugs live here)

Afterstates and net evaluations are **always canonicalized to the mover's point of
view**. The value net only ever learns one side. Every place that produces or consumes
an afterstate must respect this. This is a hard invariant in the WP0 contract, tested
explicitly, not an afterthought.

## 7. Work packages & dependency graph

| WP | Scope | Depends on | Parallelizable |
|----|-------|-----------|----------------|
| **WP0** | Foundation: env + frozen contracts + net/equity iface + checkpoint spec + test harness + env throughput benchmark | — | **No — solo, reviewed, gates everything** |
| WP1 | TD(λ) baseline: algo-agnostic training loop + agent iface + TD trainer | WP0 | after WP0 |
| WP2 | n-ply expectimax lookahead (chance nodes) | WP0 | after WP0, parallel to WP1 |
| WP3 | Web server + play UI + gnubg/.mat/.sgf export + analysis pipeline | WP0 (needs move-sequence contract + checkpoint spec) | after WP0, parallel (use a dummy/random agent until WP1 lands) |
| WP4 | LLM agent (OpenRouter) + prompt/format refinement harness + web opponent | WP0 | after WP0, parallel |
| WP5 | (optional) AlphaZero-style: policy head + chance-node MCTS + self-play | WP0 | last |

**Critical path:** WP0 is the integration risk. Freeze its contract before fan-out.
A wrong contract forces rework in every parallel package.

### 7a. Parallel execution (after the WP0 gate)

Once WP0's contract is **frozen and reviewed**, fan out to **four parallel CC sessions:
WP1, WP2, WP3, WP4.** They touch largely disjoint subtrees (`training/`+`agents/td_*`,
`agents/expectimax_*`, `web/`+`serialization/`, `llm/`+`agents/llm_*`), which is what
makes parallel work safe. Discipline:

- **One branch per session.** No two sessions edit the same file concurrently.
- **Shared files are off-limits to parallel edits:** `CLAUDE.md`, the WP0 contract
  modules, shared `__init__` exports. If a session finds the contract is wrong, it
  **escalates and pauses** — it does NOT patch around it or edit the contract on its
  branch. Contract changes force a re-sync across all sessions.
- **Integration seams are post-merge tasks, not parallel work:**
  - WP2 builds + unit-tests against a random-weight `ValueNet`; its real strength check
    (wrapping a WP1 checkpoint) happens after WP1 merges.
  - WP3 builds against `RandomAgent`; wiring real checkpoints is post-merge.
  - The LLM web-opponent lives at the WP3×WP4 seam — build both independently against
    the agent interface, wire together after both merge.
- **WP5** depends only on WP0 so is parallelizable in principle, but **hold it** — it's
  optional/lowest-priority and its policy head is cleaner to design once WP1's value net
  and loop are real.

## 8. Reference material (verified)

- **`dellalibera/td-gammon`** + **`dellalibera/gym-backgammon`** — canonical modern
  PyTorch TD-Gammon. Uses Tesauro's 198-feature encoding; `gym-backgammon` already
  generates all legal actions for (state, dice, player) using the highest number of
  dice possible, with no fixed action-space shape. Includes a module that drives gnubg
  and parses its responses. **Dated (Py3.6, old Gym API)** — treat as a reference to
  port/modernize under uv, not a dependency to pin. See WP0 for the build-vs-wrap call.
- **`bgsage`** (PyPI + github markbgsage/bgsage) — current NN backgammon library,
  self-play + supervised training, multi-ply/rollout, CUDA, benchmarked vs. XG. Uses
  the XG ply-numbering convention (their "1-ply" = raw net = gnubg's "0-ply"). Good
  architecture reference, especially for the net + multi-ply code.
- **GNU Backgammon** — free top-strength bot, our analysis oracle and benchmark
  opponent. Imports Jellyfish `.mat`, native `.sgf`, `.pos`, Snowie `.txt`. Batch
  analysis via CLI (`import mat ...; analyse match; save match ...`), depth set with
  `set analysis chequerplay evaluation plies N`. Has Position ID + Match ID strings and
  Python scripting. **One match per `.mat` file** (gnubg reads only the first match if
  multiple are concatenated). See WP3.
- TD-Gammon original: 198 inputs, 4 outputs (W win, W gammon, B win, B gammon —
  backgammon omitted as rare), TD(λ), self-play, greedy (dice provide exploration).

## 9. Ply numbering — pick one convention and document it

Two conventions exist in the wild and they're off by one: gnubg calls raw-net "0-ply",
XG/bgsage call raw-net "1-ply". **This repo uses gnubg's convention (raw net = 0-ply)**
internally for consistency with our primary analysis oracle. Translate when talking to
bgsage or XG. State this in any user-facing output.

## 10. Working agreement for sessions

- Break work into 2-3 steps and check in before large irreversible moves (big
  refactors, deleting/regenerating modules).
- Flag when guessing vs. confident. Search/verify external API surfaces (OpenRouter,
  gnubg `.mat` field layout) rather than assuming.
- Keep `ruff` + `pytest` green. Add tests with code, not after.
- Update the relevant `plans/wpN.md` with decisions made and deviations.

## 11. Self-implementation contract (applies wherever a plan marks `TODO(human)`)

Some learning-critical code (currently the TD(λ) core in WP1) is implemented by the
**human**, not CC, to test their understanding. Where a plan says so:

- **Scaffold, don't solve.** CC writes structure, signatures, types, shapes, and
  docstrings, with `TODO(human)` markers at the spots the human fills. Comments describe
  **the invariant the code must satisfy**, never **the line that satisfies it**. No
  "use exactly this formula", no giveaway expressions.
- **Then an explicit, named review gate.** After the human implements, CC reviews *their*
  code for correctness + efficiency and proposes changes **with** the human — it presents
  findings and discusses, it does **not** silently rewrite.
- Optionally run the review in a fresh session that sees only the filled-in code + the
  contract (not the scaffolding rationale), for a more honest check.

See WP1 for the concrete three-phase application.
