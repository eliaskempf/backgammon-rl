# WP6 training runbook (read this on the training machine)

WP6 turns on the four previously-untrained outcome heads (`p_win_gammon`, `p_win_bg`,
`p_lose_gammon`, `p_lose_bg`). That means a **fresh self-play run from scratch** — this is
*not* a checkpoint-extension (heads 1–4 were never trained, so there is nothing to resume
into). The doubling cube (Part B) needs **no training**: it is applied analytically on top
of the cubeless net, so this runbook is only about the Part-A net.

## The command

```bash
uv run python scripts/train.py \
  --games 1000000 \
  --eval-opponent pubeval --eval-every 25000 --eval-pairs 100 \
  --calib-games 500 \
  --save-every 50000 \
  --hidden 64 --lr 0.1 --lam 0.7 --seed 0 \
  --out-dir runs/wp6_h64_lr0.1_lam0.7_s0 \
  --wandb --resume
```

- `--games 1000000` — the actual training games, from scratch. Unchanged from WP1.
- `--calib-games 500` — **diagnostic only** (see below). Orthogonal to `--games`. `0` = off.
  Adds ~500 short self-play games per eval (every 25k games) — cheap vs. training.
- `--resume` — safe to re-launch the same command after a crash / SLURM preemption;
  `latest.pt` bundles weights + RNG + game counter and continues bit-exactly. Traces are
  zero at every game boundary, so nothing else needs persisting.
- Resource note: WP6 keeps ~5× eligibility-trace storage vs. WP1 — still tiny, CPU-only.

## What `--calib-games` measures

Each eval it plays `--calib-games` greedy self-play games with the current net and, for
every non-terminal afterstate, compares the net's predicted outcome vector to the realized
game outcome from that position's POV. Per head it logs (to `{out-dir}/calibration.csv`,
W&B `cal/*`, and stdout):

- `cal/<head>_pred` — mean predicted probability
- `cal/<head>_real` — mean realized rate (the base rate)
- `cal/<head>_ece`  — expected calibration error (0 = perfect; lower is better)

The heads are `p_win, p_win_g, p_win_bg, p_lose_g, p_lose_bg`.

## What to watch during the run

1. **`win_rate` vs pubeval must not regress.** This is the strength bar. The multi-head net
   should match or *beat* a `p_win`-only v1 net — WP6 also fixes a latent bug where the
   untrained heads (~0.5) leaked uncalibrated noise into move selection, so the leak fix
   should help, not hurt. If win-rate is clearly worse than v1, stop and flag — something is
   wrong with the multi-head update, not just calibration.

2. **Head convergence order is `p_win` → gammon → backgammon.** Expect, over the run:
   - `p_win`: calibrates fastest; `cal/p_win_pred ≈ cal/p_win_real`, ECE small early.
   - gammon (`p_win_g`, `p_lose_g`): a few× slower; pred should track real by mid-run.
   - backgammon (`p_win_bg`, `p_lose_bg`): slowest and **noisiest** (bg is ~1–2% of games).
   A bg head that stays a bit noisy is **normal, not a failure** (TD-Gammon dropped bg
   entirely; gnubg/XG treat it as a minor correction).

3. **The collapse signal (the only real "bad" outcome).** A magnitude head where
   `cal/<head>_pred ≈ 0` while `cal/<head>_real` is clearly positive and *stays* that way
   late in the run means that head never accumulated enough rare-event signal — it collapsed
   to zero. Watch the **gammon** heads especially; bg collapsing is tolerable, gammon
   collapsing is not (cube decisions consume it).
   - This is the **gate for A5** (optional supervised rollout fine-tuning). Only build A5 if
     a gammon (or, if you care about it, bg) head is genuinely collapsed at the end of a real
     run. Otherwise skip A5 — see `plans/wp6-multihead-cube.md` §A5.

4. **Sanity:** all `cal/*_pred` in [0,1]; `cal/p_win_real ≈ 0.5` over self-play (both seats
   are the same net). Gammon realized rates in the low tens of percent, bg in low single
   digits.

## The sweep (hidden width was answered "sweep it")

Run the same command across the grid and pick the winner by **win-rate, with calibration as
the tie-/quality-breaker**:

- `--hidden {64, 128}` × `--lr {…}` × `--lam 0.7` (λ anchored), distinct `--out-dir` per run
  (e.g. `runs/wp6_h128_lr0.1_lam0.7_s0`) and distinct `--seed`.
- Rank with: `uv run python scripts/aggregate_runs.py runs/wp6_* --reeval-pairs 500`
  — it prints win-rate plus the **win-gammon / win-backgammon head ECEs** (`gam_ece`,
  `bg_ece`) per run, so you can prefer a run that is both strong *and* well-calibrated on the
  magnitude heads. Lower ECE = better-calibrated.

## After training — wiring the cube (no retraining)

The trained cubeless checkpoint is everything Part B needs. The money-game cube path
(`bgrl.money.play_money_game`, `CubeDecider`, `cubeful_equity`) consumes the net's outcome
distribution analytically; `x = 2/3` is already confirmed against gnubg (no tuning needed).
Nothing about the cube requires another training run.
