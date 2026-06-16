# WP0 Decisions

Records the build-vs-wrap call, the frozen representation choices, the correctness
methodology, the throughput evidence behind the CPU-vs-GPU / env decision, and
(§7) the **frozen contracts** that gate the WP1–WP4 fan-out.

**Status:** WP0 complete, pending human review. Env + benchmark done; cluster-CPU run
landed (sbatch job 29176287 on `dlc2cpu09`, a 192-core EPYC-9655 node) — **GREEN at
every worker count** (557 games/s @ 16 workers up to 4,360 games/s @ 192; see §5). Env
call: **keep the DFS generator**; recommended self-play parallelism ≈ 64 workers (the
scaling-efficiency knee). The net/equity, agent-lifecycle, checkpoint, and dice/CRN
contracts are now **frozen** (§7); `uv run ruff check` clean and `uv run pytest` green
(61 tests). Human review gate precedes WP1+ fan-out.

---

## 1. Environment: build vs. wrap — **build a DFS generator, oracle against gym-backgammon**

The plan's default was to port `dellalibera/gym-backgammon`'s move generator. On
inspection that move-gen is a **1526-line special-case thicket** whose bear-off and
doubles enumerators (`get_bear_off_play_double`, `get_bar_plays_double`, 500+ lines)
are high-risk to port faithfully. Instead:

- We wrote a compact **depth-first afterstate generator** (`bgrl/env/movegen.py`,
  ~120 lines). Keying the DFS on afterstates makes **dedup free**, and keeping only
  maximal-length branches makes **"use the maximum number of dice"** fall out by
  construction. A forced-higher-die post-filter covers the one case pure max-length
  misses.
- We **vendored** gym-backgammon's move-gen verbatim (`bgrl/_vendor/`, test-only) as
  a **differential oracle**.

**This is a deviation from "port their move-gen", and it turned out to be the more
correct choice:** over 384k random reachable positions our generator is a strict
**superset** of the reference — it produces 8 legal max-dice plays the reference's
doubles-bear-off enumerator *misses* (hand-verified; e.g. from `(3,3)` in a home
board, `5→2, 5→2, 2→off, 3→0`). We never miss a play the reference finds
(`dangerous=0`), and every returned `Move` replays to its afterstate
(`replay_fail=0`).

Fallback if throughput ever disappoints: port the enumerator (oracle still applies),
numba/Cython the hot path, or adopt `bgsage`'s C++ engine. The benchmark says this is
not needed (Section 5).

## 2. Representation & contract choices

- **Absolute signed board.** `EnvState.board` is 24 signed ints (`+n`=WHITE, `−n`=BLACK)
  in fixed absolute coordinates — one canonical truth that web/gnubg export read
  without un-flipping, and that the oracle compares apples-to-apples. WHITE moves
  toward index 0 (home 0..5), BLACK toward 23 (home 18..23).
- **Perspective at the encode boundary, not in the state.** `encode(state, perspective)`
  canonicalises to the mover's POV; the stored state stays absolute. The perspective
  invariant (CLAUDE.md §6) is tested explicitly (colour-mirror ⇒ identical encoding).
- **Encoding = Tesauro 198 + 2 reserved cube slots = 200 features**, emitted in
  **mover-relative order** with a side-to-move flag that encodes only *whether* the
  perspective player is on move (never colour). This **deviates from the reference's
  absolute WHITE-then-BLACK layout** on purpose — perspective invariance (the net
  learns one side) outranks byte-matching the reference. The cross-check is on
  *afterstates*, not encodings. `ENCODING_VERSION = 1`.
- **Outcome magnitude** (single/gammon/backgammon) is detected in `bgrl/env/outcome.py`
  — the reference only signals "WHITE won".
- **Net output is the fixed 5-vector** `[p_win, p_win_gammon, p_win_bg, p_lose_gammon,
  p_lose_bg]` (mover POV) even though v1 trains only `p_win` — cube-ready shape.

## 3. Correctness methodology (the gate)

`uv run pytest` (31 tests, incl. a `@slow` differential-oracle test) + `uv run ruff
check` are green. Coverage:

- **Golden** exact afterstate counts (openings cross-checked vs. the oracle: 3-1 → 16,
  6-6 → 11, 5-5 → 4) and crafted edge cases: bear-off exact / overshoot-allowed /
  overshoot-blocked-by-higher-checker, bar entry forced/blocked, closed-out pass,
  forced-higher-die, and the doubles-bear-off oracle-gap position.
- **Property (hypothesis)** over reachable positions: all plays maximal length,
  afterstates distinct, each `Move` replays to its afterstate, turn flips, cube fields
  untouched, 15-checker conservation per side.
- **Differential oracle:** every reference play ⊆ ours (containment, since we are a
  correct superset — see §1).
- **Perspective:** colour-mirror invariance of `encode`.

## 4. CPU-only this round; GPU deferred

WP1 is **classic online TD(λ)** (single game, eligibility traces). The only forward
passes are afterstate *evaluation* at batch = branching factor B ≈ 19 and the TD
*update* at batch 1 — both far below the tiny-net (200→64→5) GPU/CPU crossover
(~64–256). So GPU cannot help WP1; CPU-vs-GPU is settled analytically. GPU is revisited
only where batches get large: WP2 expectimax (leaf-batching) / WP5 MCTS, or a future
switch to batched/parallel self-play. No CUDA code or cluster-GPU run this round.

## 5. Throughput evidence

Method: random self-play across spawn workers (thread-pinned, barrier-started, rate =
summed work ÷ max-worker wall-time); a single-worker baseline for per-position cost and
scaling; an `encode` micro-bench; and a **single-threaded** CPU net forward-pass sweep
(single-thread = the per-worker cost in CPU multiprocessing self-play). See
`bgrl/bench/`. The net-eval *share* models a 0-ply value-agent step (enumerate B,
encode B, one forward pass of size ~B).

### Laptop (`DESKTOP-F7EPFT4`, AMD Ryzen 7 8845HS, 8 phys / 16 logical, torch 2.12.0+cpu), 2000 games

| metric | value |
|---|---|
| games/s (8 workers) | **188.9** |
| games/s (1 worker) | 33.9 |
| scaling efficiency @8 | 0.70 |
| legal_moves calls/s (8w) | 17,876 |
| afterstates/s (8w) | 332,299 |
| mean afterstates/position (B) | 18.6 |
| mean positions/game | 94.6 |
| encode/s (1 thread) | 150,355 |
| net fwd pos/s (1 thread) | b1 87.7k · b8 588k · b16 983k · b32 1.62M · b64 2.26M |
| per-position split | **move-gen 69% · encode 27% · net-eval 4%** |

Interpretation: move-gen dominates; net-eval is ~4% even at the realistic batch ≈ B,
and a single CPU thread already does 88k–2.3M forward passes/s — corroborating that
GPU offers nothing for online TD. ~189 games/s on a laptop ⇒ ~1M games in ~1.5 h;
a cluster CPU node with more cores scales further.

### Cluster CPU (`dlc2cpu09`, 2× AMD EPYC 9655 = 192 phys cores, SMT off, torch 2.12.0+cpu), 10000 games

Worker-count sweep on **one exclusive node** (clean scaling curve, no noisy neighbours),
run via `slurm/benchmark-cpu-sweep.sh` (gitignored; sbatch job 29176287, partition
`lmbdlc2_cpu-epyc9655`) — 4 runs + aggregate in 92 s.

| workers | games/s | scaling eff | games/s per worker | afterstates/s | net-eval % |
|---|---|---|---|---|---|
| 16 | 557.2 | 0.97 | 34.8 | 0.99M | 3.9 |
| 32 | 1,103.3 | 0.92 | 34.5 | 1.97M | 4.0 |
| 64 | 2,087.4 | 0.84 | 32.6 | 3.73M | 4.1 |
| 192 | 4,359.6 | 0.57 | 22.7 | 7.76M | 4.1 |

Single-worker baseline ~36–40 games/s; per-position split unchanged from the laptop
(move-gen ~69% · encode ~27% · net-eval ~4%, B≈18.6). Net forward-pass (1 thread):
b1 103k · b8 672k · b16 1.16M · b32 1.79M · b64 2.48M pos/s.

Interpretation: **GREEN at every worker count** — even 16 workers (557 g/s) is 2.8× the
200 g/s bar. Throughput scales near-linearly to ~64 workers (84% efficient); beyond that
NUMA/contention bites (192 workers yields 2.1× the games of 64 but at 0.57 efficiency, 22.7
vs 32.6 g/s per worker). **Sweet spot ≈ 64 workers** for efficient per-core use; use the
full 192 only when a node is dedicated and raw throughput is all that matters (4,360 g/s ⇒
~1M games in ~4 min). Intermediate points (96/128) were not sampled. Reproduce:

```bash
uv sync --frozen --group dev          # once on the login node
sbatch slurm/benchmark-cpu-sweep.sh   # exclusive node; sweeps 16/32/64/192 workers
uv run python scripts/aggregate_bench.py bench_results/cluster-cpu-*w-<jobid>.json
```

## 6. Decision rule

Applied by `scripts/aggregate_bench.py` over the combined JSON:

- **Keep DFS env vs. replace:** correctness is a hard gate (already green). Throughput
  bar on the **cluster CPU node**: aggregate **≥ 200 games/s = green**, ≥ 50 acceptable,
  < 50 investigate. If < 50 and move-gen > 70% of the loop, escalate (port enumerator /
  numba / bgsage engine). *Laptop is 189 (acceptable, sub-green only for lack of cores);
  cluster confirms green — 557→4,360 games/s across 16→192 workers (§5), so keep the DFS env.*
- **Workers:** set self-play parallelism near the physical-core count, trimmed to the
  scaling-efficiency knee — measured ≈ 64 workers on the 192-core node (eff 0.84, falling
  to 0.57 at 192).
- **CPU vs GPU:** CPU (see §4); net-eval share is the supporting evidence.

## 7. Frozen contracts (the final WP0 slice)

These are the seams WP1–WP4 bind to. **Off-limits to parallel edits** (CLAUDE.md §7a);
a change here forces a re-sync across all sessions. Each is documented in its module
docstring and covered by tests (`tests/{nets,agents,serialization,env}/`, `tests/test_game.py`).

**Outcome vector + net (`bgrl/nets/base.py`).** `ValueNet` Protocol = `evaluate(features)
-> (..., OUTCOME_DIM)`. `OUTCOME_DIM = 5`, the cube-ready mover-POV vector
`[p_win, p_win_gammon, p_win_bg, p_lose_gammon, p_lose_bg]`, interpreted **cumulatively,
gnubg-style** (`p_win_gammon` = P(win gammon *or better*), etc.; `p_lose = 1 - p_win`
implied). v1 trains only `p_win`; the shape never changes. `MLPValueNet` satisfies the
Protocol and adds `arch_config()`/`from_config()` for checkpointing.

**Equity (`bgrl/nets/equity.py`).** `equity(outcome, cube=CENTERED_CUBE) -> ndarray` is the
**only** thing move selection ranks. Cubeless money equity over the cumulative vector:
`(p_win + p_win_g + p_win_bg) - (p_lose + p_lose_g + p_lose_bg)`, which is `2*p_win - 1`
when the gammon/bg heads are zero. **Anti-symmetric** by construction (flipping win/loss
heads negates it) — that is what makes selection a single `argmax`. `CubeContext`
(value, owner) is accepted but ignored in cubeless v1.

**Agent lifecycle (`bgrl/agents/base.py`).** `Agent` = `act(state, dice, legal) -> Move`
(unchanged; non-learning agents stop here). `LearningAgent(Agent)` adds
`observe_step(state, dice, move, afterstate)` — fired **once per ply the agent makes** — and
`observe_game_end(outcome)` — fired **once per distinct learning agent** with the absolute
`Outcome`. In shared-net self-play one agent plays both seats and sees the whole afterstate
trajectory (consecutive afterstates alternate POV; the perspective flip is the WP1 update
rule, not this contract). The TD(λ) body is WP1's human task; only the signatures freeze here.

**Value selection (`bgrl/agents/value_agent.py`).** `ValueAgent` is the 0-ply greedy-by-equity
selector every value method shares (TD/expectimax/MCTS-leaf). **Sign convention (load-bearing,
CLAUDE.md §6):** an afterstate's `turn` is the *opponent*, so `encode(afterstate,
afterstate.turn)` + the net give the opponent's equity; the mover picks `argmax(-equity)` =
its own equity (valid because equity is anti-symmetric). Deterministic tie-break by default;
optional `rng` for random ties. WP1's `TDAgent` subclasses this to add the learning hook;
WP2 wraps it in n-ply search.

**Dice / CRN (`bgrl/env/dice.py`).** `roll_dice(rng)`; `DiceSource` Protocol; `RandomDiceSource`
(records `.history`); `ReplayDiceSource` (replays, raises when exhausted). A game is fully
determined by `(agents, dice source)`; replaying a recorded `history` to another config is
common-random-numbers comparison (WP1 eval, WP4 prompt A/B). The env never touches global RNG.

**Game driver (`bgrl/game.py`).** `play_game(white, black, dice, *, max_plies, record) ->
GameResult`. The algorithm-agnostic loop WP1's trainer is built around (same learning agent
in both seats = self-play). Lives at package root so `bgrl/training/` stays free for WP1.

**Checkpoint (`bgrl/serialization/checkpoint.py`).** `CHECKPOINT_FORMAT_VERSION = 1`; a
self-describing dict `{format_version, net_arch, weights, encoding_version, outcome_dim,
trained_with, metadata}`. `save_checkpoint` / `load_checkpoint` (validates `format_version`
and **guards `encoding_version`** — fail loudly, no migration yet) / `load_net` (via
`NET_REGISTRY` + `from_config`) / `load_agent(checkpoint) -> Agent` (returns a `ValueAgent`
wrapping the net — the generic factory WP3 loads any checkpoint through). **Seam for WP3:**
`bgrl/serialization/__init__.py` will also export the gnubg/.mat/.sgf functions — append
there, don't restructure.

### Still deferred (owned by later WPs)

Vectorized/batched env and all GPU / large-batch benchmarking → WP2 (expectimax leaf-batching)
/ WP5 (MCTS). Policy head on the net → WP5 (the `ValueNet` contract does not preclude a second
head). Encoding-version *migration* (vs. the current guard) → only if `ENCODING_VERSION` ever
bumps. Doubling cube / gammon-weighted equity heads → when those heads are trained (the
`CubeContext` argument and 5-vector shape already exist for it).
