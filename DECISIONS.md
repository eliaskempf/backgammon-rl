# WP0 Decisions

Records the build-vs-wrap call, the frozen-ish representation choices, the
correctness methodology, and the throughput evidence behind the CPU-vs-GPU / env
decision. This slice is **benchmark-first**: the core contracts (net/equity,
checkpoint, full Agent hooks, CRN) are deferred to a contract-freeze follow-up
(see end).

**Status:** env + benchmark complete and green on the laptop. The cluster-CPU run
is pending (run the benchmark command in §5 on a cluster CPU node); the final env
keep/replace call + recommended worker count are confirmed once that JSON lands.
Human review gate precedes WP1+ fan-out.

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

### Cluster CPU — PENDING

The benchmark is host-portable (no hardcoded paths/cores; CPU torch wheel only — no
CUDA). On a cluster CPU node, from the repo root:

```bash
uv sync --frozen --group dev
uv run python scripts/benchmark_env.py \
    --games 5000 --workers "${SLURM_CPUS_PER_TASK:-$(nproc)}" --bench-net --profile \
    --tag cluster-cpu --out bench_results/cluster-cpu.json
```

Wrap that in whatever the cluster uses (sbatch/srun); set workers to the node's
physical core count. Copy the JSON back next to `bench_results/laptop.json`, then:

```bash
uv run python scripts/aggregate_bench.py bench_results/*.json
```

## 6. Decision rule

Applied by `scripts/aggregate_bench.py` over the combined JSON:

- **Keep DFS env vs. replace:** correctness is a hard gate (already green). Throughput
  bar on the **cluster CPU node**: aggregate **≥ 200 games/s = green**, ≥ 50 acceptable,
  < 50 investigate. If < 50 and move-gen > 70% of the loop, escalate (port enumerator /
  numba / bgsage engine). *Laptop is 189 (acceptable, sub-green only for lack of cores);
  expected green on the cluster.*
- **Workers:** set self-play parallelism near the physical-core count, trimmed to the
  scaling-efficiency knee.
- **CPU vs GPU:** CPU (see §4); net-eval share is the supporting evidence.

## 7. Deferred to the contract-freeze follow-up

Freeze: net/equity Protocol + `equity(outcome, cube)`; checkpoint spec
(`save/load_checkpoint`, `load_agent`); full Agent hooks (`observe_step`/
`observe_game_end`); CRN replay; encoding-version migration. Plus any vectorized env and
all GPU/large-batch benchmarking (owned by WP2/WP5). Minimal versions exist now:
`Env`, `encode`, `RandomAgent`, injectable dice RNG, random-weight `MLPValueNet`.
