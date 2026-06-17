# WP3 — Web Server, Play UI, and GNU Backgammon Export/Analysis

**Status:** not started · **Depends on:** WP0 (move-sequence contract + checkpoint spec)
**Parallel with:** WP1, WP2, WP4 — build against `RandomAgent` until real checkpoints exist.

**Branch:** this session creates branch `wp3-web-gnubg` off the latest `main` itself before any code (CLAUDE.md §10); it never asks the human to manage branches.

A reasonably nice web UI to play backgammon against any checkpoint (or the LLM agent),
plus serialization of played games into a GNU-Backgammon-importable format and a batch
analysis pipeline that compares agent moves against gnubg's evaluation.

---

## Part A — Web server + UI

- **Backend:** thin FastAPI (or Flask) app in `bgrl/web/`, agent loaded server-side via
  the WP0 `load_agent(checkpoint)` factory. The agent/contract is the load-bearing
  piece; the frontend is disposable/regenerable.
- **Stable API contract** (nail this down first; the UI is built against it and can be
  regenerated freely):
  - `POST /new_game` → game id, initial state, whose turn.
  - `POST /roll` → dice (server-side RNG, seedable for repro).
  - `GET  /legal_moves?game&dice` → list of legal Moves (human-legible submoves +
    resulting afterstate id) — straight from `Env.legal_moves`.
  - `POST /move` → apply a human or agent Move; return new state + terminal/outcome.
  - `POST /agent_move` → agent picks + applies; return Move + new state.
  - State serialization = a JSON view of `EnvState` + render hints.
- **Frontend:** prefer reusing a maintained open-source JS backgammon board component
  (rendering, drag-drop, dice, legal-destination highlighting, bear-off) over rendering
  from scratch — board interaction has many fiddly edge cases. **Verify which board
  libs are currently maintained before committing** (flag: don't guess). Wire it to the
  API above. Claude Code may build/replace the frontend at will as long as the API holds.
- **Checkpoint picker:** UI dropdown to select among available checkpoints; show
  metadata (games trained, trained_with). Allow choosing agent search depth (WP2) to
  trade strength vs. latency.
- **LLM opponent (from WP4):** selectable as an agent. Note UX: frontier-model moves can
  take seconds (more with reasoning) — show loading state, timeout, maybe a cheaper
  default model for interactive play vs. the strong model for offline eval.

## Part B — GNU Backgammon export + analysis (verified pipeline)

Goal: write played games to a gnubg-importable file, batch-analyze with gnubg, and
surface "agent move vs. gnubg's preferred move + equity loss."

- **Serialize to Jellyfish `.mat`** (and/or native `.sgf`) in `bgrl/serialization/`.
  gnubg imports `.mat`, `.sgf`, `.pos`, Snowie `.txt`. `.mat` is simple text and
  well-supported. **One match per file** — gnubg reads only the first match if multiple
  are concatenated.
- **First task: validate a round-trip.** Emit a known game to `.mat`, import into gnubg,
  confirm the position/moves match (the exact `.mat` field layout must be verified
  empirically — the import/analyse pipeline is confirmed, the byte-level spec is not).
  Position ID + Match ID strings are the lightweight path for single-position checks.
- **Batch analysis via gnubg CLI** (no GUI needed): prepare an input script like
  ```
  set analysis chequerplay evaluation plies 2
  set analysis cubedecision evaluation plies 3
  import mat game.mat
  analyse match
  save match game.sgf
  ```
  Run headless gnubg against it, then parse the resulting `.sgf`/analysis to extract per-
  move equity loss and gnubg's preferred move. (gnubg also has Python scripting — viable
  alternative to text parsing; evaluate which is more robust.)
- `scripts/eval_vs_gnubg.py`: given a game record or a set of self-play games, produce a
  report of agent vs. gnubg move agreement and mean equity loss per move (a clean,
  standard strength metric — lower equity loss = stronger).
- Requires gnubg installed on the host; document install (Ubuntu pkg / build) and make
  the pipeline skip gracefully with a clear message if gnubg is absent.

## Acceptance criteria

- A human can play a full game vs. a checkpoint in the browser, including bar entry and
  bear-off, with legal-move enforcement.
- A played game exports to `.mat`, imports cleanly into gnubg, and round-trips (same
  position/moves).
- `eval_vs_gnubg.py` produces a per-move equity-loss report for at least one real game.
- API is stable and documented; frontend is regenerable without backend changes.
- `uv run pytest` green (serialization round-trip tests with a fixture game),
  `uv run ruff check` clean.

## Pitfalls

- Cube-ready but cubeless v1: `.mat`/`.sgf` must still record game-end magnitude
  (single/gammon/backgammon) faithfully (WP0 §5).
- Don't let the frontend hold game logic — all legality comes from the backend
  (`Env.legal_moves`). The browser only renders + collects intent.
- gnubg ply numbering is its own convention (raw = 0-ply); keep our internal convention
  consistent and translate at the boundary.

---

## Status & decisions

Split into **two PRs** (disjoint subtrees; independently reviewable):
- **PR1 — Part A (web + UI): DONE** on branch `wp3-web-ui` (off `main`, rebased onto WP1).
- **PR2 — Part B (gnubg export + analysis): not started**, to be done in its own session
  on `wp3-gnubg-export`. gnubg will be installed (`apt-get install gnubg`) and the live
  round-trip + equity-loss report validated in-session.

### Part A decisions (implemented)
- **Backend:** FastAPI app factory `create_app()` in `bgrl/web/` (`schemas`, `views`,
  `session`, `agents`, `app`). `GameSession`/`SessionStore` hold server-side state and are
  the sole legality boundary — handlers never reason about legality, the env does.
- **API (frozen contract, pydantic):** `POST /new_game`, `POST /roll`,
  `GET /legal_moves`, `POST /move` (by `move_id`; stale/illegal → 409),
  `POST /agent_move`, `GET /checkpoints`, `POST /export_mat` (501 until PR2). State is
  projected in **absolute** coordinates; never mover-relative.
- **Incremental human move:** the frontend filters the full legal-`Move` list by the
  chosen submove prefix (no partial-legality endpoint); a complete prefix submits its
  `move_id`. A legal-moves list (notation buttons) is the always-works fallback.
- **Frontend:** the human chose the **npm path**. Verification found *no maintained,
  play-capable* JS board component, so we use the npm toolchain only for tooling +
  rendering: a **custom Vite + React + TS SVG board** we control. Source in `frontend/`;
  the built bundle is **committed to `bgrl/web/static/`** (Node 18 only needed to
  rebuild; FastAPI serves the bundle). `scripts/play_web.py` runs the server.
- **Deps:** new `web` dependency-group (`fastapi`, `uvicorn[standard]`); `httpx` added to
  `dev` for the `TestClient`. Narrow `filterwarnings` ignore for Starlette's
  `httpx`-vs-`httpx2` TestClient deprecation (we stay on `httpx`).
- **Tests:** `tests/web/` — view projections, `GameSession` mechanics (incl. a genuinely
  blocked forced-pass position and terminal detection), and a `TestClient` suite (full
  game loop, 409 enforcement, session isolation, checkpoint opponent). All green; ruff
  clean. Live `uvicorn` smoke confirmed static serving + the full game flow.
- **Verified facts:** WHITE enters from the bar at absolute index `24 − die` (into BLACK's
  home, 18–23); doubles render 4 submoves (e.g. `6/2 6/2 6/2 6/2`).
