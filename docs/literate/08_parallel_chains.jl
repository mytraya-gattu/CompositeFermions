# # 8. Parallel chains: distributed sampling and convergence diagnostics
#
# A single Metropolis–Hastings–Gibbs chain gives one estimate of a density or a correlator, with
# no independent handle on whether it has converged or how large its Monte Carlo error is. The
# standard fix is to run **several independent chains** from different random starts: comparing
# them diagnoses convergence (do they agree?), and pooling them both shrinks the error bar and
# uses many cores at once.
#
# This tutorial uses the distributed driver [`run_chains`](@ref) (built on Julia's `Distributed`
# standard library) to run the ``\nu = 1/3`` ground state on several worker processes, monitors
# convergence live with [`monitor_chains`](@ref), and pools the result with [`combine_chains`](@ref).
# The per-chain sampling is *exactly* the loop of Tutorial 1 — the driver only orchestrates many
# copies of it and handles checkpointing, resumption, and diagnostics.

using CFsOnSphere
using Random
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Plots
gr()

# ## Running chains across worker processes
#
# In a real run you add worker processes and load the package on each of them before calling the
# driver. Each chain is built and sampled **on its worker** — the wavefunction objects hold large
# caches and are never serialized across processes, so `setup` constructs them worker-side. The
# convenience wrapper [`sample_density_chains`](@ref) packages the density + pair-correlation
# `setup` for the projected CF ground state:
#
# ```julia
# using Distributed
# addprocs(4)                       # four worker processes
# @everywhere using CFsOnSphere     # load the package on every worker (required!)
#
# # ν = 1/3 ground state, N = 30, four chains, one per worker.
# combined = sample_density_chains("run_nu13", 30, 1, 2;
#                                  n_chains = 4, seed = 2025,
#                                  num_thermalization = 500_000, num_steps = 2_000_000,
#                                  record_every = 1_000, checkpoint_every = 500_000)
# ```
#
# While the chains run, the master prints a plain-text status table every `monitor_every` seconds
# (readable in a Slurm log — no terminal control codes). A typical tick looks like:
#
# ```
# Chain status  (dir = run_nu13,  4 chain(s) read)
#   chain             steps       accept            σ      mean logpdf
#   ----------------------------------------------------------------
#   1               1500000       0.4998      0.28713        -412.6631
#   2               1500000       0.5013      0.28840        -412.5902
#   3               1500000       0.4991      0.28691        -412.7188
#   4               1500000       0.5007      0.28775        -412.6449
#   ----------------------------------------------------------------
#   split-R̂ (logpdf) = 1.0009     multi-chain ESS = 5871.3
# ```
#
# The acceptance rate sits at the ARM target of `0.50`, the four `mean logpdf` values agree, and
# the split-R̂ is essentially 1 — the chains have mixed.
#
# If you forget `@everywhere using CFsOnSphere`, the workers cannot see the package; `run_chains`
# detects this up front and raises an informative error rather than failing deep inside `pmap`.

# ## A small in-process run
#
# For this page we keep everything in one process (`nprocs() == 1`), where the driver runs the
# chains sequentially in-process — the same code path, just without workers. We use a short run so
# the page builds quickly; production runs use far more steps.

dir = mktempdir()
combined = sample_density_chains(dir, 10, 1, 2;
                                 n_chains = 4, seed = 2025,
                                 num_thermalization = 20_000, num_steps = 60_000,
                                 record_every = 500, checkpoint_every = 30_000)
print(monitor_chains(dir))

# ## Split-R̂ and effective sample size
#
# Two scalar diagnostics summarize convergence, both computed here from the recorded `logpdf`
# trace (any scalar observable works).
#
# **Split-R̂** ([`split_rhat`](@ref)) is the Gelman–Rubin ratio after splitting each chain in half
# (to also catch *within*-chain drift). With ``2M`` half-chains of length ``n``, between- and
# within-chain variances
#
# ```math
# B = \frac{n}{2M-1}\sum_m (\bar\theta_m - \bar\theta)^2, \qquad
# W = \frac{1}{2M}\sum_m s_m^2,
# ```
#
# combine into ``\widehat{\mathrm{var}}^+ = \frac{n-1}{n}W + \frac{1}{n}B`` and
#
# ```math
# \hat R = \sqrt{\widehat{\mathrm{var}}^+ / W}.
# ```
#
# ``\hat R \to 1`` as the chains mix; values ``\gtrsim 1.01`` flag non-convergence.
#
# **Effective sample size** ([`effective_sample_size`](@ref)) discounts autocorrelation. With
# normalized autocorrelations ``\rho_t``, Geyer's initial-positive-sequence truncation sums the
# consecutive pairs ``\Gamma_m = \rho_{2m} + \rho_{2m+1}`` while they stay positive, giving the
# integrated autocorrelation time ``\tau = -1 + 2\sum_m \Gamma_m`` and
# ``\mathrm{ESS} = Mn/\tau`` for ``M`` pooled chains. A short, well-mixed run has ``\mathrm{ESS}``
# close to the number of recorded samples; a sticky chain has far fewer.

report = monitor_chains(dir)
@show report.rhat report.ess;

# ## Pooling the chains
#
# [`combine_chains`](@ref) reads the per-chain checkpoints and pools the observables with a
# step-weighted mean (equivalent to summing raw counts and dividing by the pooled step count). It
# returns the pooled observables, the per-chain values, cross-chain standard errors for scalar
# observables, and the diagnostics.

density = combined["observables"]["density"]
θc      = combined["grids"]["theta grid"]
@show combined["n_chains"] combined["total_steps"];

# We plot the pooled density as the local filling ``\nu(\theta) = 2\pi\, n(\theta)/Q_{\text{shift}}``
# (bulk value = ``\nu``), exactly as in Tutorial 1 but now averaged over all chains.

N, ν   = 10, 1/3
Qshift = N / (2ν)
νθ     = density .* (2π / Qshift)

plot(θc, νθ; lw = 2, label = "pooled (4 chains)", ylims = (0.0, 0.5),
     xlabel = "θ", ylabel = "local filling  ν(θ)", title = "ν = 1/3 ground state, pooled chains")
hline!([ν]; ls = :dash, label = "ν = 1/3")

# ## The measurement callback (arbitrary observables)
#
# `sample_density_chains` is a thin wrapper; the general driver lets you accumulate *any*
# observable through the `measure!` callback of [`gibbs_sampling!`](@ref). It is called once per
# step, and its `accepted` flag lets you do work only on accepted moves. The density's
# pair-correlation histogram uses exactly this: it is maintained **incrementally on acceptance**
# (only the moved particle's ``N-1`` distances change) yet **accumulated every step**. Because
# `measure!` fires before the driver re-synchronizes its two wavefunction buffers, an accumulator
# that captures both buffers sees the *new* column in `Ψnext` and the *pre-move* column in
# `Ψcurrent` on an accepted step:
#
# ```julia
# measure! = function (Ψ, θ, ϕ, accepted, sampling_iter)
#     if accepted
#         for i in 1:N-1
#             cdd[bin(Ψnext.dist_matrix[i, sampling_iter])]    += 1.0   # new distances
#             cdd[bin(Ψcurrent.dist_matrix[i, sampling_iter])] -= 1.0   # pre-move distances
#         end
#     end
#     accumulated_pair_density .+= cdd            # accumulate every step
#     update_density!(θmesh, θ, accumulated_density)
# end
# ```
#
# To build your own driver, pass a `setup(chain_id, RNG)` returning
# `(Ψcurrent, Ψnext, θ, ϕ, logpdf, measure!, observables)` to [`run_chains`](@ref); see its
# docstring for the precise contract, including the `observables` finalizer layout that lets
# [`combine_chains`](@ref) pool your quantities and resume merge them correctly.

# ## Resuming a killed run
#
# Each chain checkpoints to `dir/chain_<id>.jld2` (positions, `σ`, sampling index, step count,
# thinned trace, wall time, and the user observables). Passing `resume = true` restarts each chain
# from its checkpoint — thermalization is skipped and sampling continues until `num_steps` total
# is reached, with the observables of both legs merged. This makes long campaigns robust to
# wall-clock limits and preemption on a cluster.
