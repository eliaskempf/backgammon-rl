# WP1 — TD(λ) core cleanup & validation — HANDOFF

**For:** the next planning session. **Goal:** clean up `bgrl/training/td_lambda.py`
(`step` + `episode_end`), validate the update rule against a reference / ground
truth, and verify end-to-end. **Do not trust the inline hints from the prior
session** — some were wrong (see "Correction" below). Trust the Monte-Carlo
equivalence test described here; implement, then let that test confirm correctness.

---

## 0. Status & how we got here

- WP1 Phase A (scaffold + algorithm-agnostic loop + CRN eval + scripts + tests) is
  merged on branch `wp1-td-lambda` and green (commit `c20878b`). The only hollow
  piece was the TD(λ) math in `bgrl/training/td_lambda.py` (`TDLambda.step` /
  `episode_end`).
- That math was being self-implemented by the human (CLAUDE.md §11). **The human has
  now released that constraint and asked CC to finish + validate it.** So the next
  session implements the core itself.
- During the hint phase CC made an error worth flagging: it told the user to **remove
  the eligibility-trace sign-flip**. That was wrong. The correct mover-relative
  update needs **both** a bootstrap complement **and** a trace sign-flip (§3). The
  reason the error was plausible: at `λ=0` the sign-flip is invisible (no trace
  memory), so a complement-only version looks fine — but the real setting is
  `λ=0.7`, where the sign-flip matters.

---

## 1. Current code state (has bugs — to be replaced)

`bgrl/training/td_lambda.py` currently holds a partial attempt. Provided correctly
already: `__init__` (net, `lam`, `gamma`, `lr`, `self._traces` = one zero tensor per
param, `self._prev = None`) and `_value(afterstate, perspective)` (differentiable
`p_win` via `net.forward`, **not** `evaluate`). The two methods below are buggy:

`step` (current):
```python
v = self._prev if self._prev is not None else self._value(state, state.turn)
v_next = self._value(afterstate, afterstate.turn)
v_next.backward()
with torch.no_grad():
    td_error = v_next - v
    for i, (trace, param) in enumerate(zip(self._traces, self.net.parameters())):
        self._traces[i] = self.lam * self.gamma * trace + param.grad   # NO sign flip
        param += self.lr * td_error * trace
self._prev = v_next.detach()
```
Bugs: (a) `td_error = v_next - v` lacks the **complement** (should bootstrap toward
`1 - v_next`); (b) the trace lacks the **sign-flip** (`-λγ·e + ∇f`); (c) relies on a
fragile Python aliasing trick (`trace` keeps the pre-reassignment tensor) for the
deferred ordering — works, but should be made explicit.

`episode_end` (current):
```python
r = outcome.winkind          # AttributeError: attribute is `outcome.kind`
td_error = r - v             # uses the WinKind magnitude (1/2/3) as a probability
...
```
Bugs: `outcome.winkind` doesn't exist (it's `outcome.kind`), **and** the target must
be a win/lose probability in `[0,1]`, not the win magnitude (§3). Reset (zero traces,
`_prev = None`) is correct.

---

## 2. The design (fixed by WP0, do not change)

- Net is **mover-relative**: `_value(a, persp) = P(persp wins | a)` over the Tesauro
  encoding; the agent (`ValueAgent.act`) ranks afterstates by `_value(after,
  after.turn)` then negates equity. So the quantity to learn is
  **`f(a) = _value(a, a.turn)`** — the to-move player's win prob at afterstate `a`,
  the same query the agent makes (side-to-move flag always "is mover").
- This differs from classic TD-Gammon / `dellalibera/td-gammon`, which use a **fixed**
  White-perspective value (`td_error = p_next - p`, plain trace `λ·e + grad`, no flip;
  selection by argmax for White / argmin for Black). That reference is correct *for
  the fixed-perspective convention* but does **not** directly validate our
  mover-relative version — see §4.
- v1 trains the **`p_win` head only** (index 0). Backprop through `f` leaves the other
  4 output heads with zero gradient (they're untrained — fine; the 5-vector shape is
  preserved). γ defaults to 1.0 (undiscounted episodic). No `torch.optim` (manual
  trace-scaled update).

---

## 3. The correct update (implement this, then prove it with §4)

Learn `f(a) = _value(a, a.turn)`. Because consecutive afterstates belong to **opposite
movers**, two perspective adjustments are needed **together**:

- **Bootstrap complement.** The successor of `a_t` belongs to the opponent, so the
  target for the previous afterstate is `1 - f(a_t)`, not `f(a_t)`.
- **Trace sign-flip.** The value's POV flips every ply, so the eligibility trace
  negates its carry-over: `e_t = -λγ·e_{t-1} + ∇f(a_t)`. (Complement handles the
  one-step target; sign-flip handles multi-step credit for `λ>0`.)

Recommended **deferred** online structure (fold the current afterstate's gradient,
apply the correction to the previous afterstate, handle the terminal in
`episode_end`). Per `step(state, dice, move, afterstate)` — only `afterstate` is
needed:

```
zero_grad
v_cur = _value(afterstate, afterstate.turn)      # f(a_t)
v_cur.backward()                                  # p.grad = ∇f(a_t)  (fresh graph)
with no_grad:
    if _prev is not None:                         # _prev = f(a_{t-1}) (detached scalar)
        δ = (1 - v_cur.detach()) - _prev          # complement bootstrap
        for p, e in zip(params, traces): p.add_(lr * δ * e)     # apply over e_{t-1}
    for i, p in enumerate(params):
        traces[i] = -lam*gamma*traces[i] + p.grad # sign-flipped fold of ∇f(a_t)
    _prev = v_cur.detach()
```

`episode_end(outcome)`:
```
with no_grad:
    δ = 0.0 - _prev                               # terminal target 0 (see below)
    for p, e in zip(params, traces): p.add_(lr * δ * e)
for e in traces: e.zero_()
_prev = None
```

**Terminal target = 0** (not `outcome.kind`): in backgammon you only end the game by
bearing off your own last checker, so the player who makes the final move is the
**winner** ⇒ the to-move player at the terminal afterstate `a_T` is the **loser** ⇒
`f(a_T) = P(loser wins)` has realised target `0`. (For v1 `p_win` the target is
0/1 only; `outcome.kind`'s gammon/backgammon magnitude is for the other heads later,
so `outcome` is otherwise unused in v1 — that's expected.)

Autograd gotcha (this bit the prior session twice): **never cache a graph-bearing
tensor in `_prev` and `.backward()` it on a later ply** — the in-place weight update
mutates the saved weights and PyTorch raises *"a variable needed for gradient
computation has been modified by an inplace operation"* on the 2nd ply. Cache only the
**detached scalar**; compute a fresh forward each ply (one forward + one backward per
ply — negligible for this 200→h→5 net; net-eval is ~4% of the loop per the WP0 bench).

---

## 4. Validation — the reference and the proof

**Primary ground truth: Monte-Carlo equivalence at λ=1.** This holds for *any* net
(no symmetry assumptions) and pins the complement, the sign-flip, and the terminal
target simultaneously.

> With `λ=1, γ=1`, the **offline** (weights frozen across the episode) total
> backward-view update equals Monte-Carlo regression of each afterstate value toward
> the realised outcome:
> `Σ_t (y_t − f(a_t)) · ∇f(a_t)`, where `y_t = 1 if a_t.turn == outcome.winner else 0`.

Proof sketch (telescoping): with `δ_t = (1 − f_{t+1}) − f_t` (t<T), `δ_T = y_T − f_T`,
and the sign-flipped trace `e_t = Σ_{k≤t} (−1)^{t−k} ∇f_k` at λ=1, one gets
`Σ_t δ_t e_t = Σ_k (Σ_{t≥k} (−1)^{t−k} δ_t) ∇f_k`. The inner sum telescopes to
`y_k − f_k` using `y_k + y_{k+1} = 1` (consecutive afterstates have opposite movers
and exactly one side wins). Hence the total = the MC update. ∎

Perspective bookkeeping for the test (the error-prone part — get it exactly right):
- afterstates: `a_t = step.afterstate` for each recorded `Step`.
- maker of `a_t` = `step.state.turn`; `a_t.turn = step.afterstate.turn` = opponent of
  the maker.
- `f(a_t) = _value(a_t, a_t.turn)`; MC target `y_t = (a_t.turn == outcome.winner)`.
  (Sanity: for the terminal `a_T`, `a_T.turn` is the loser ⇒ `y_T = 0`.)

**Exact test design (tests the production class, not a re-impl):** snapshot `θ0`;
for each recorded step call `trainer.step(...)`, record `Δ = θ − θ0`, then **restore
`θ = θ0`** (keep `_traces` and `_prev`); same after `episode_end`. The summed `Δ` is
the offline backward update at `θ0`; assert it ≈ `lr · MC_update(θ0)` (compute the MC
update independently with `torch.autograd.grad`/`backward` on `f` at `θ0`). Use
`λ=1, γ=1`, a small `MLPValueNet(hidden=16)`, `torch.allclose(atol≈1e-4)`. A missing
complement, a missing/added sign-flip, or a wrong terminal target all break this test
by an order-1 margin.

**Secondary reference (optional cross-check):** `dellalibera/td-gammon`
(`td_gammon/model.py`) — fixed-perspective vanilla TD(λ). Useful to confirm the *trace
mechanics* (decay, `lr·δ·e` update) but note the convention difference (§2): our
mover-relative version is equivalent to it only if the net satisfied
`_value(a,W) = 1 − _value(a,B)`, which it doesn't in general — so don't use it as a
numeric equality test.

**End-to-end:** `tests/training/test_td_training_smoke.py` already exists and
**self-activates** once `step` stops raising (it probes `_td_core_ready()`). It trains
~1000 games at `λ=0.7` and asserts >90% vs `RandomAgent`, reproducibly. Run with
`uv run pytest tests/training/test_td_training_smoke.py -m slow -v`.

---

## 5. Concrete tasks for the next session

1. Rewrite `TDLambda.step` and `episode_end` per §3 (clean, explicit ordering — no
   aliasing trick; in-place `p.add_`; separate apply-then-fold loops).
2. Update the module/class/method docstrings: it's **implemented now**, not
   `TODO(human)` — document the complement + sign-flip + terminal-target-0 reasoning
   and the MC-equivalence guarantee. Remove the stale `# TODO(human)` / commented
   `raise` lines.
3. Add `tests/training/test_td_lambda.py`: the **MC-equivalence test** (§4, the
   restore-weights offline trick) plus a small "traces reset / `_prev=None` after
   `episode_end`" test.
4. Run `uv run ruff check` + `uv run ruff format`, then `uv run pytest` (incl.
   `-m slow`). The smoke test must clear >90% and be reproducible.
5. Phase C review pass against the WP1 checklist (trace decay `λγ`; bootstrap; terminal
   uses actual outcome; perspective/sign-flip; reset; gradient to net params only; no
   retained graph; vectorised in-place trace ops).
6. Decide whether to keep `bgrl/agents/__init__.py` untouched (export `TDAgent`
   post-merge) — current code imports `TDAgent` via submodule for parallel-safety.

## 6. Files

- `bgrl/training/td_lambda.py` — the core (rewrite `step` + `episode_end`).
- `tests/training/test_td_lambda.py` — **new**, the MC-equivalence + reset tests.
- `tests/training/test_td_training_smoke.py` — exists; auto-activates; end-to-end gate.
- `bgrl/agents/td_agent.py`, `bgrl/training/{loop,evaluate}.py`, `scripts/*` — done in
  Phase A; do not touch (loop/eval already split train/eval RNG via `spawn(2)` and seed
  torch for reproducible init).
- Reference: `github.com/dellalibera/td-gammon` (fixed-perspective; convention note in §2/§4).
