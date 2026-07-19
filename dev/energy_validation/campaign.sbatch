#!/bin/bash
# Ground-state-energy validation campaign for the parallel-chains driver.
# One array task = one full basic node (64 cores); each task runs its share of the
# 48 systems in ref/cf_ground_state_energies.csv via run_node.jl (node-harvest style:
# the node is held for the whole run, chains fill all cores).
#
# Submit from the worktree root on Roar:
#   sbatch --array=1-<K> dev/energy_validation/campaign.sbatch
# Tunables come from the environment (set by the submit wrapper):
#   CFVAL_NODES, CFVAL_CHAINS, CFVAL_THERM, CFVAL_STEPS, CFVAL_WORKTREE, CFVAL_SCRATCH, CFVAL_FINAL
#
#SBATCH --account=open
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --job-name=cf_energy_val
#SBATCH --output=/storage/home/mvg6042/scratch/cf_energy_validation/logs/%x_%a.out

set -u

JULIA=/storage/home/mvg6042/work/julia-1.12.2/bin/julia
WORKTREE=${CFVAL_WORKTREE:-/storage/home/mvg6042/work/CompositeFermions-parallel-chains}
SCRATCH_OUT=${CFVAL_SCRATCH:-/storage/home/mvg6042/scratch/cf_energy_validation}
FINAL_OUT=${CFVAL_FINAL:-/storage/home/mvg6042/work/cf_energy_validation}

N_NODES=${CFVAL_NODES:?set CFVAL_NODES to the array size}
N_CHAINS=${CFVAL_CHAINS:-63}
THERM=${CFVAL_THERM:-500000}
STEPS=${CFVAL_STEPS:-2000000}

mkdir -p "$SCRATCH_OUT/logs" "$FINAL_OUT"

export JULIA_PROJECT="$WORKTREE"
# One BLAS thread per worker (the package enforces this, but belt and suspenders).
export OPENBLAS_NUM_THREADS=1

"$JULIA" -p $((SLURM_CPUS_PER_TASK - 1)) \
    "$WORKTREE/dev/energy_validation/run_node.jl" \
    "$WORKTREE/ref/cf_ground_state_energies.csv" \
    "$SLURM_ARRAY_TASK_ID" "$N_NODES" \
    "$SCRATCH_OUT" "$N_CHAINS" "$THERM" "$STEPS"

status=$?

# Last step: copy results off scratch to work storage (always, even on partial failure).
rsync -a "$SCRATCH_OUT/" "$FINAL_OUT/"

exit $status
