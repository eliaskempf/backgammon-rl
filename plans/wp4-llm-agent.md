# WP4 — LLM Agent (OpenRouter) + Prompt/Format Refinement Harness

**Status:** not started · **Depends on:** WP0 · **Parallel with:** WP1, WP2, WP3

An LLM baseline agent that picks moves by prompting a frontier model via OpenRouter,
plus a harness that sweeps prompt/format variants to find good settings. Also exposed as
a web opponent (WP3).

---

## Scope

- `bgrl/llm/client.py`: OpenRouter client. **Verify the current OpenRouter API surface
  before committing** — model availability, whether native structured-output / tool-use
  / JSON-mode is uniformly supported across the models you want to compare, rate limits,
  cost, reasoning-effort knobs. Don't guess; search/read docs. Secrets via env var
  (`OPENROUTER_API_KEY`), never hardcoded.
- `bgrl/agents/llm_agent.py`: implements the WP0 `Agent` interface. Given `legal`
  (human-legible Moves + afterstates), render a prompt, call the model, parse the choice.
  - **Selects from the legal Move list** (e.g. return an index/ID of the chosen legal
    move) — the LLM does not score afterstates numerically. This validates that the WP0
    contract exposes concrete Moves, not just afterstates.
  - **Validity + fallback policy (correctness AND fairness issue):** model may return an
    illegal/malformed/prose response. Pipeline: constrained output format (prefer native
    structured output / index-from-enumerated-list) → parser → on invalid, re-prompt up
    to N times → final fallback (random-legal or first-legal). **The reported metric is
    "model + harness," not the model in isolation** — document this; a weak parser
    understates a model.
- **Board serialization is a pluggable function** (`bgrl/llm/render.py`): ASCII board /
  pip list / XGID-or-position-ID / move-list-only. This is one of the swept knobs, so
  keep it a strategy, not hardcoded. (Frontier models may have seen XGID/position-ID
  formats in training — worth including as a candidate.)

## Refinement harness (a small eval harness)

- `bgrl/llm/refine.py` + `scripts/refine_llm.py`: sweep over
  {board representation, prompt template, output format, reasoning-effort coarse on/off,
   model} and score each candidate.
- **Scoring — prefer cheap move-matching over full self-play for the sweep:** evaluate
  candidates by agreement / equity-loss vs. gnubg on a **fixed position set** (cheap,
  low variance, reuses WP3's gnubg pipeline). Reserve full games vs. a fixed opponent
  (RandomAgent / WP1 checkpoint / WP2 / gnubg) for final validation only.
- **Cost + latency are first-class:** a full game is ~50-60 plies ≈ 25-30 model calls
  per side. Sweeping × many candidates × games gets expensive fast. Budget it; cache
  responses; make game-count and candidate-count CLI params. Default to the cheap
  position-set scorer.
- **Variance control:** dice make single games noisy. Use common-random-numbers (WP0 §7)
  so candidates face identical roll sequences; or use the deterministic position-set
  scorer to sidestep dice variance entirely for the sweep.
- Reasoning-effort: treat as coarse (off / on) first, not a fine sweep — it multiplies
  cost.

## Web opponent (integrates with WP3)

- Selectable LLM agent in the web UI. Surface model choice. Handle latency: loading
  state, timeout, possibly a cheaper default model for interactive play vs. the strong
  model for offline baseline eval. Cache where sensible.

## Acceptance criteria

- LLM agent plays legal full games end-to-end via the WP0 interface, with a robust
  parser + fallback and logged invalid-response rate.
- Refinement harness runs a sweep and emits a ranked report (candidate → score, cost,
  invalid-rate), using the cheap gnubg position-set scorer by default.
- Selectable as a web opponent (with WP3).
- `uv run pytest` green (mock the OpenRouter client in tests — no live API calls in CI),
  `uv run ruff check` clean.

## Pitfalls

- Live API calls must never run in CI/tests — mock the client.
- Report "model + harness" performance honestly; don't conflate parser failures with
  model weakness.
- Keep board rendering and prompt templates as swappable strategies from the start;
  retrofitting the sweep over hardcoded prompts is painful.
- Cost can balloon silently — log token usage + $ per run, fail loudly on a budget cap.
