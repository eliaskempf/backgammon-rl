# WP0 — Foundation (gates everything)

**Status:** complete, pending human review. Env + correctness suite + throughput
benchmark green (laptop + cluster CPU); contracts §2,§3,§5,§6,§7 are now **frozen**
(`uv run ruff check` clean, `uv run pytest` green — 61 tests). Decisions & deviations in
`DECISIONS.md` (notably: a DFS afterstate generator with gym-backgammon as a differential
oracle instead of a literal port; mover-relative 200-feature encoding; gnubg-cumulative
5-vector with anti-symmetric equity and mover-POV afterstate selection; CPU-only — GPU
deferred to WP2/WP5).
· **Owner:** solo session, human-reviewed before fan-out
**Depends on:** nothing · **Blocks:** WP1–WP5

This package produces the frozen contracts, the env, the net/equity interfaces, the
checkpoint spec, the test harness, and the env throughput benchmark. **No other WP may
begin until the contracts here are reviewed and frozen.** Getting this wrong forces
rework everywhere.

---

## 0. Deliverables checklist

- [x] Repo scaffold: uv project (Python 3.13), `bgrl/` package, `scripts/`, `tests/`,
      ruff + pytest configured, CI-runnable `uv run pytest` / `uv run ruff check`.
- [x] `bgrl/env`: board state, dice, **legal-move + afterstate enumeration**, encoding,
      outcome detection. Build-vs-wrap decision made *on benchmark evidence* (§4).
- [x] Frozen **EnvState / move / afterstate contract** (§2).
- [x] Frozen **Agent interface** (§3).
- [x] Frozen **Net + equity-module interface** (§5), with cube-ready vector output.
- [x] **Checkpoint spec** (§6).
- [x] **RNG/seeding** contract for reproducible dice (§7).
- [x] **Test harness**: move-gen correctness (golden + property-based), perspective
      invariant, encoding round-trips (§8).
- [x] **Env throughput benchmark** runnable on laptop + remote CPU/GPU (§4).
- [x] Short `DECISIONS.md` recording the build-vs-wrap call and any contract deviations.

## 1. Encoding (fixed)

Tesauro 198-feature scheme as the baseline observation:
- For each of 24 points: 4 units per player encoding checker count (units 1-3 are
  threshold flags for 1/2/3 checkers, unit 4 = max(0, (n-3))/2 for the standard
  encoding), ×2 players = 192.
- Plus: checkers on bar (per player), checkers borne off (per player), and side-to-move
  flag → 198 total. Match the dellalibera/gym-backgammon layout so we can cross-check.
- **Reserve additional slots for cube state** (value + ownership) now, even if unused
  in v1. Document exact index layout in a module-level docstring; it is part of the
  contract.

Encoding lives in `bgrl/env/encoding.py` as a pure function
`encode(state: EnvState, perspective: Player) -> np.ndarray` (float32, shape fixed).

## 2. EnvState / move / afterstate contract (FREEZE)

Concrete signatures (adjust names if cleaner, but keep the shape of the contract):

```python
Player = Literal["white", "black"]            # or an IntEnum; pick one, document it
Dice   = tuple[int, int]                      # post-roll; doubles handled in move gen

@dataclass(frozen=True)
class EnvState:
    board: tuple[int, ...]      # signed checker counts per point, canonical indexing
    bar:   tuple[int, int]      # (white, black)
    off:   tuple[int, int]      # borne off (white, black)
    turn:  Player
    cube_value: int = 1         # reserved; always 1 in v1
    cube_owner: Player | None = None  # reserved; always None (centered) in v1
    # NOTE: equality/hash must make equivalent positions equal (canonical form).

@dataclass(frozen=True)
class SubMove:                  # one checker movement
    src: int                    # point index, or BAR sentinel
    dst: int                    # point index, or OFF sentinel

@dataclass(frozen=True)
class Move:                     # a full legal play for a (state, dice)
    submoves: tuple[SubMove, ...]   # 1..4 (4 for doubles); order canonicalized

class Env:
    @staticmethod
    def initial_state() -> EnvState: ...

    @staticmethod
    def legal_moves(state: EnvState, dice: Dice) -> list[tuple[Move, EnvState]]:
        """Return (move, afterstate) pairs. Afterstate is the deterministic result of
        applying `move` to `state` BEFORE the opponent rolls. Afterstate.turn is the
        OPPONENT (it's their move next). Must enforce: use highest number of dice
        possible; both dice if legal; doubles = up to 4 submoves; correct bear-off.
        Deduplicate afterstates that are reachable by different submove orderings, but
        keep ONE canonical Move per distinct afterstate (the web UI + gnubg export need
        a concrete legal submove sequence). Empty list = no legal move (turn passes)."""

    @staticmethod
    def is_terminal(state: EnvState) -> bool: ...

    @staticmethod
    def outcome(state: EnvState) -> Outcome | None:
        """None if not terminal. Else who won AND magnitude (single/gammon/backgammon),
        from a fixed POV. Needed for cube-ready training targets + gnubg export."""
```

**Perspective invariant (FREEZE):** afterstates and encodings are canonicalized to the
mover's POV. `encode(afterstate, perspective=afterstate.turn)` is what the net sees.
Test this explicitly (§8).

**Why `(Move, afterstate)` not just afterstate:** the agent only needs the afterstate
to choose, but the web UI must render the actual checker movements and gnubg export
must record the submove sequence + dice. Returning both from day one prevents a painful
retrofit. The LLM agent (WP4) also consumes the human-legible `Move`, not the afterstate.

## 3. Agent interface (FREEZE)

```python
class Agent(Protocol):
    def act(self, state: EnvState, dice: Dice,
            legal: list[tuple[Move, EnvState]]) -> Move:
        """Choose one Move from `legal`. Value agents score afterstates via a net +
        equity; the LLM agent picks from the human-legible Moves; random picks
        uniformly. Receiving `legal` precomputed keeps move-gen in the env, not the
        agent."""

    # Optional hooks (no-ops for non-learning agents):
    def observe_step(self, ...) -> None: ...     # for online TD updates
    def observe_game_end(self, outcome: Outcome) -> None: ...
```

Provide `RandomAgent` in WP0 as the reference implementation + a test/dev opponent and
as the dummy agent WP3 can build against before WP1 lands.

## 4. Env: build vs. wrap — DECIDE ON BENCHMARK EVIDENCE

Do **not** pre-commit. Evaluate, record in `DECISIONS.md`.

**Sensible default options, in rough order of preference to evaluate:**
1. **Port/modernize `dellalibera/gym-backgammon`'s move-gen** into `bgrl/env` under
   Python 3.13 (drop the old Gym API; we don't need the gym.Env wrapper, just the
   legal-action generation + encoding). Closest match to our afterstate-first design,
   proven correct, known encoding. **Likely default.**
2. **Reuse `bgsage`'s env** if its move-gen/afterstate API is cleanly importable and
   fast — it's actively maintained and already perf-minded.
3. **Wrap `gym-backgammon` as-is** as a dependency. Fastest to stand up, but old API +
   may not expose afterstates the way we want + Python-version friction. Fallback only.
4. **Fresh custom generator.** Cleanest fit, but move-gen has nasty edge cases
   (forced moves, must-use-both-dice, doubles, bear-off exact/overshoot). Only if 1-2
   are too slow or too awkward — and only with the full test suite (§8) green.

**Decision criterion:** correctness (passes §8 golden tests) first, then throughput.
Target: self-play game generation fast enough that ~10^5–10^6 games is hours-not-weeks
on a single modern CPU (move-gen is THE bottleneck, not gradients).

### Throughput benchmark (must run on heterogeneous hardware)

`scripts/benchmark_env.py` must:
- Measure **games/sec** and **legal_moves() calls/sec** for random self-play.
- Run on: (a) the dev laptop, (b) a remote CPU box, (c) a remote GPU box. Env move-gen
  is CPU-bound, so the GPU run is about confirming no accidental GPU dependency in the
  env and measuring net-eval throughput separately, NOT about env speed.
- Be **hardware-portable**: no hardcoded paths, no assumed core counts, no GPU
  requirement to run. Auto-detect cores; `--workers N` override. Print a machine
  fingerprint (CPU model, core count, RAM, torch CUDA availability) with results.
- Support `--games`, `--seed`, `--workers`, `--profile` (cProfile hot spots in move-gen).
- Emit a small JSON/markdown report so laptop vs. remote numbers are comparable.
- Separately, an optional `--bench-net` mode times batched net afterstate-eval on
  CPU vs. GPU (informs whether GPU helps for ply-search/MCTS later).

Document how to launch on a remote machine with uv (uv sync, uv run) so results are
reproducible across hosts.

## 5. Net + equity interface (FREEZE)

```python
OUTCOME_DIM = 5   # [p_win, p_win_gammon, p_win_bg, p_lose_gammon, p_lose_bg], mover POV

class ValueNet(Protocol):
    def evaluate(self, features: np.ndarray) -> np.ndarray:   # (..., OUTCOME_DIM)
        """Batched. Input = encode(afterstate, mover POV). Output = outcome probs.
        v1 may only train p_win and leave others ~0, but the shape is FIXED."""

def equity(outcome: np.ndarray, cube: CubeContext) -> float:
    """Reduce outcome vector to scalar equity. Cubeless v1: 2*p_win - 1 (or the
    gammon-weighted cubeless formula once those heads train). Move selection consumes
    THIS, never raw net outputs. cube arg exists from day one (trivial in v1)."""
```

Net implementation (`bgrl/nets/`): start with the TD-Gammon-style shallow MLP
(198→hidden(40–80)→OUTCOME_DIM, sigmoid). Keep architecture swappable. Policy head for
WP5 is added later as a second head; don't build it now, but don't preclude it.

## 6. Checkpoint spec (FREEZE)

`bgrl/serialization/checkpoint.py`. A checkpoint must be loadable by the web server
(WP3) and eval scripts without knowing which algorithm produced it.

```
checkpoint = {
  "format_version": int,
  "net_arch": {...},          # enough to reconstruct the module
  "weights": ...,             # state_dict
  "encoding_version": int,    # which feature layout (so old ckpts stay loadable)
  "outcome_dim": 5,
  "trained_with": str,        # "td_lambda" | "mcts_az" | ...  (informational)
  "metadata": {games_trained, created_at, git_sha, notes, ...},
}
```

Provide `save_checkpoint` / `load_checkpoint` + a `load_agent(checkpoint) -> Agent`
factory so WP3 can load "any checkpoint" generically.

## 7. RNG / seeding (FREEZE)

- Dice rolls go through an injectable RNG (`numpy.random.Generator` or explicit seed),
  never global random. Self-play reproducibility requires this.
- Support **common-random-numbers**: the ability to replay the *same dice sequence*
  across two agents/configs (needed for low-variance agent-vs-agent and LLM prompt
  comparison in WP4).
- `Env` must never touch global RNG state.

## 8. Test harness (build with the code, not after)

- **Golden tests:** known positions → exact expected legal-move counts and a few exact
  afterstates. Cover: opening rolls, doubles (4 submoves), forced single-die plays,
  must-use-higher-die, bar entry, bear-off exact vs. overshoot, no-legal-move pass.
- **Property-based** (hypothesis): generated reachable states never produce illegal
  moves; afterstate checker conservation (15 per side accounting for bar/off);
  must-use-both-dice-if-possible holds.
- **Perspective invariant:** encode/eval is POV-consistent; flipping the side and
  mirroring the board yields the mirrored encoding.
- **Encoding round-trip / shape** stability; encoding_version guard.
- **Determinism:** same seed → same self-play game.
- The agent cannot catch its own move-gen bugs (it just learns to exploit them), so
  this suite is the real correctness backstop. Treat coverage here as load-bearing.

## 9. Acceptance criteria

- `uv run pytest` green; `uv run ruff check` clean.
- `scripts/benchmark_env.py` runs on laptop + a remote host and emits comparable reports.
- A `RandomAgent` can play full legal games end-to-end via the training-loop stub.
- Contracts (§2,§3,§5,§6,§7) documented in module docstrings AND summarized in
  `DECISIONS.md`, reviewed by a human before WP1+ start.
