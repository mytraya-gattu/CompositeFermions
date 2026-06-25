#!/bin/bash
#SBATCH --job-name=cf_mc
#SBATCH --account=jkj2_cr_default
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# --- Expected environment variables (set by run_mc_dft_loop.sh via --export=ALL) ---
# CF_MC_STEPS      : production steps per chain
# CF_MC_THERM      : thermalization steps per chain
# CF_MC_N          : number of electrons
# CF_MC_TWOQ       : 2Q (monopole strength)
# CF_MC_P          : flux attachment (2p)
# CF_MC_NMAX       : Lambda level cutoff
# CF_MC_CHAINS_DIR : directory to write chain_<id>.jld2 output files
# CF_MC_PROJECT    : path to CompositeFermions project root
# CF_ORBITAL_FILE  : path to DFT orbital JLD2 file (empty string to use standard Ψproj)

module load julia  # adjust to your module system

CHAIN_ID=${SLURM_ARRAY_TASK_ID}

ORBITAL_ARGS=""
if [ -n "${CF_ORBITAL_FILE}" ] && [ "${CF_ORBITAL_FILE}" != "none" ]; then
    ORBITAL_ARGS="--orbital-file ${CF_ORBITAL_FILE}"
fi

julia --project="${CF_MC_PROJECT}" \
    "${CF_MC_PROJECT}/scripts/mc_chain_worker.jl" \
    --N      "${CF_MC_N}" \
    --twoQ   "${CF_MC_TWOQ}" \
    --p      "${CF_MC_P}" \
    --nmax   "${CF_MC_NMAX}" \
    --chain  "${CHAIN_ID}" \
    --steps  "${CF_MC_STEPS}" \
    --therm  "${CF_MC_THERM}" \
    --out    "${CF_MC_CHAINS_DIR}/chain_${CHAIN_ID}.jld2" \
    ${ORBITAL_ARGS}
