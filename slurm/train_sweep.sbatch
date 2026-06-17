#!/bin/bash
# TD(λ) self-play training sweep — one single-core run per array task.
#
# Online TD(λ) is single-process, so the cluster's value is running independent runs
# in parallel (seeds x net size), not speeding up one run. Each task trains ~1M games
# (~8 h single-core) and writes checkpoints + metrics.csv to its own out-dir; pick the
# strongest afterwards with scripts/aggregate_runs.py.
#
# Submit (after a ONE-TIME `uv sync --frozen --group train` on the login node):
#     sbatch slurm/train_sweep.sbatch
#
#SBATCH --job-name=bgrl-td-sweep
#SBATCH --partition=lmbdlc2_cpu-epyc9655
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=14:00:00
#SBATCH --output=runs/sweep/slurm-%A_%a.out
## --account / --qos: set to match your allocation, e.g.
## #SBATCH --account=<your-account>

set -euo pipefail

# Pin to a single thread: the net is tiny (intra-op threading is pure overhead) and
# many array tasks share a node, so threads would oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Cluster nodes usually have no outbound internet; log offline and `wandb sync` later
# from the login node. Override by exporting WANDB_MODE=online before sbatch.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Config grid: array index -> (seed, hidden). Edit freely; keep --array in sync.
SEEDS=(0 1 2 0 1 2)
HIDDEN=(64 64 64 96 96 96)
i="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$i]}"
H="${HIDDEN[$i]}"
OUT="runs/sweep/h${H}_s${SEED}"
mkdir -p "${OUT}"

echo "task ${i}: seed=${SEED} hidden=${H} -> ${OUT}"

# --frozen: never mutate the committed lock on a compute node. --group train: wandb.
uv run --frozen --group train python scripts/train.py \
    --games 1000000 --hidden "${H}" --lam 0.7 --lr 0.1 \
    --eval-opponent pubeval --eval-every 25000 --eval-pairs 100 --save-every 50000 \
    --seed "${SEED}" --out-dir "${OUT}" \
    --wandb --wandb-project backgammon-rl-wp1
