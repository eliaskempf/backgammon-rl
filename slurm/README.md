# Cluster training (SLURM)

Online TD(λ) self-play is **single-process** (~33 games/s on one core; ~8 h for 1M
games). A compute node does not make one run faster — its value is running many
**independent** runs in parallel. So we launch a **seed × net-size sweep** as a SLURM
array and pick the strongest checkpoint afterwards.

Strength is measured live against **pubeval** (Tesauro's public-domain benchmark
evaluator, the standard absolute yardstick — `bgrl/agents/pubeval_agent.py`). Win-rate
vs random saturates in a few thousand games and is useless for a long run; vs-pubeval
is meaningful (a fresh net ≈ 0, climbing toward 0.5+; gnubg-level comparison comes in
WP3).

## Launch

```bash
# 1. One-time on the login node: materialise the exact env (incl. wandb).
uv sync --frozen --group train

# 2. Submit the array (edit slurm/train_sweep.sbatch first: --account/--qos, and the
#    SEEDS/HIDDEN grid + matching --array range). SLURM won't create the --output dir.
mkdir -p runs/sweep
sbatch slurm/train_sweep.sbatch

# 3. Watch it.
squeue --me
tail -f runs/sweep/slurm-*_*.out          # stdout: per-eval win-rate vs pubeval
```

Each array task writes to `runs/sweep/h<HIDDEN>_s<SEED>/`:

- `metrics.csv` — `games, win_rate, avg_plies, truncated, wall_seconds` (one row per
  eval; the reliable record).
- `td_<games>.pt` — periodic checkpoints; `best.pt` — best eval win-rate so far;
  `final.pt` — end of run.
- `wandb/` — offline W&B run (see below).

## Pick the best run

```bash
uv run python scripts/aggregate_runs.py runs/sweep/*                 # rank by metrics.csv
uv run python scripts/aggregate_runs.py runs/sweep/* --reeval-pairs 500   # lower-variance re-eval
```

It prints a ranked table and names the winning `best.pt`. Spot-check it:

```bash
uv run python scripts/eval_agent.py --checkpoint runs/sweep/<best>/best.pt \
    --opponent pubeval --pairs 500
```

## Weights & Biases

The sbatch logs to W&B in **offline** mode (`WANDB_MODE=offline`) since compute nodes
usually lack internet. Sync afterwards from a host with network access:

```bash
wandb login            # once
wandb sync runs/sweep/*/wandb/offline-run-*
```

For online logging instead, `export WANDB_MODE=online` (and ensure `WANDB_API_KEY` is
set) before `sbatch`. W&B is optional — `metrics.csv` always captures the curve, and a
missing/unconfigured wandb only prints a warning (it never aborts a run).
