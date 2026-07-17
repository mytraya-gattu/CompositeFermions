# # Parallel multi-chain sampling of a CF ground state with convergence diagnostics
#
# This script runs several independent Metropolis–Hastings–Gibbs chains of the ν = n/(pn+1)
# composite-fermion ground state across Julia `Distributed` worker processes, monitors
# convergence live (split-R̂, effective sample size, acceptance), and pools the chains into a
# combined density and pair correlation. The per-chain sampling is identical to
# examples/sampler_single_state.jl — `run_chains` only orchestrates many copies of it and
# handles checkpointing/resumption.
#
# Run it (four worker processes) with, e.g.:
#
#     julia --project -p 4 examples/sampler_parallel_chains.jl
#
# or start workers from inside the script via addprocs (as below). Either way the package must be
# loaded on every worker with `@everywhere using CFsOnSphere`.

using Distributed

# Start worker processes if none were launched with `-p`. Adjust to the cores you have.
if nprocs() == 1
    addprocs(4)
end

# Load the package (and single-threaded BLAS) on the master AND every worker. The wavefunction
# objects are built worker-side inside the driver, so every process needs the package.
@everywhere begin
    using CFsOnSphere
    using LinearAlgebra
    LinearAlgebra.BLAS.set_num_threads(1)
end

# Choose the state: ν = 1/3 ground state (n = 1 filled Λ-level, p = 2 attached vortices) for N
# electrons. The convenience wrapper `sample_density_chains` builds the density + pair-correlation
# observables of examples/sampler_single_state.jl (same 500-point θ mesh and 5000-point r grid).
N, n, p = 30, 1, 2

# Output directory: one checkpoint file chain_<id>.jld2 per chain is written here as the run
# progresses (and re-read by monitor_chains / combine_chains).
outdir = joinpath(@__DIR__, "parallel_chains_run")

# Run the chains. `seed` fixes the per-chain RNGs (Xoshiro(hash((seed, chain_id)))), so the whole
# campaign is reproducible. The master prints a status table every `monitor_every` seconds.
combined = sample_density_chains(outdir, N, n, p;
                                 n_chains          = nworkers(),
                                 seed              = 2025,
                                 num_thermalization = 500_000,
                                 num_steps          = 2_000_000,
                                 record_every       = 1_000,
                                 checkpoint_every   = 500_000,
                                 monitor_every      = 15.0)

# `sample_density_chains` returns `combine_chains(outdir)`. Inspect convergence and the pooled
# result. `combined["diagnostics"]` carries the cross-chain split-R̂ and multi-chain ESS of the
# logpdf traces; a well-converged run has R̂ ≈ 1.
println("\nFinal status:")
print(monitor_chains(outdir))

diag = combined["diagnostics"]
println("\nsplit-R̂ = ", diag.rhat, "   multi-chain ESS = ", diag.ess)
println("pooled over ", combined["n_chains"], " chains, ", combined["total_steps"], " total steps")

# The pooled density and pair correlation, directly comparable to the single-chain reference.
density = combined["observables"]["density"]
θgrid   = combined["grids"]["theta grid"]
pairg   = combined["observables"]["pair densities"]
rgrid   = combined["grids"]["r grid"]

# Save the pooled observables for downstream plotting (e.g. a separate Julia plotting script).
save(joinpath(outdir, "combined.jld2"),
     Dict("density"        => density,
          "theta grid"     => θgrid,
          "pair densities" => pairg,
          "r grid"         => rgrid,
          "split rhat"     => diag.rhat,
          "ess"            => diag.ess,
          "n chains"       => combined["n_chains"],
          "total steps"    => combined["total_steps"]))

println("\nWrote pooled observables to ", joinpath(outdir, "combined.jld2"))

# To resume a run that was cut short (e.g. by a Slurm wall-clock limit), call the same command
# with `resume = true`: each chain restarts from its checkpoint (thermalization skipped) and
# continues until `num_steps` total, merging the observables of both legs.
#
#     sample_density_chains(outdir, N, n, p; n_chains = nworkers(), seed = 2025,
#                           num_steps = 2_000_000, resume = true)
