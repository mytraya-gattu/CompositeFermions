# Multi-chain distributed MCMC driver. `gibbs_sampling!` factors the measurement loop of
# examples/sampler_single_state.jl into a reusable, observable-agnostic function; `run_chains`
# spreads independent chains over `Distributed` workers (one checkpoint file per chain);
# `monitor_chains`/`combine_chains` read those checkpoints back for live/offline convergence
# diagnostics and pooled observables. The sampling algorithm is UNCHANGED from the example:
# same sphere proposal, same ARM-tuned σ from `gibbs_thermalization!`, same one-particle Gibbs
# sweep order `sampling_iter = mod(sampling_iter, N) + 1`, same accept rule
# `logpdf_next - logpdf_current >= log(rand(RNG))`.

"""
    gibbs_sampling!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, sampling_iter,
                    logpdf, num_steps;
                    measure! = (Ψ, θ, ϕ, accepted, sampling_iter) -> nothing,
                    record_every = 1000,
                    checkpoint  = (step, acceptance_rate, σ, trace) -> nothing,
                    checkpoint_every = typemax(Int))
        -> (num_accepted, trace, sampling_iter)

Run the Metropolis–Hastings–Gibbs **measurement** phase for `num_steps` single-particle moves,
starting from the (already thermalized) state carried by `Ψcurrent`, `θcurrent`, `ϕcurrent` and
the tuned step size `σ`. This is the loop of `examples/sampler_single_state.jl` factored out
verbatim in its algorithm; only the observable bookkeeping is delegated to callbacks. Works for
any wavefunction type (`Ψproj`, `Ψparton`, `ΨoneLL`, `Ψunproj`) via the same `Union` as
[`gibbs_thermalization!`](@ref).

Each step: propose with [`proposal`](@ref), update `Ψnext`, evaluate `logpdf(Ψnext)`, accept iff
`logpdf_next - logpdf_current >= log(rand(RNG))`, then advance `sampling_iter`.

# Callbacks
- `measure!(Ψ, θcurrent, ϕcurrent, accepted::Bool, sampling_iter)` is called **once per step**,
  after the accept/reject decision and the `θ/ϕ` update but **before** the internal `Ψcurrent` /
  `Ψnext` buffers are re-synchronized. The first argument `Ψ` is the buffer that holds the
  **current accepted configuration** (`Ψnext` on an accepted step, `Ψcurrent` on a rejected one),
  and `θcurrent`, `ϕcurrent` are the current accepted angles. Because the buffers are not yet
  synchronized, an accumulator that captures *both* buffers can read the new column from `Ψnext`
  and the pre-move column from `Ψcurrent` on an accepted step — this is exactly what the example's
  incremental pair-correlation update needs (update only when `accepted`, accumulate every step).
  See the parallel-chains tutorial for that pattern.
- `checkpoint(step, acceptance_rate, σ, trace)` is called every `checkpoint_every` steps and once
  at the end, for the driver to persist state.

`logpdf_current` is recorded into the returned `Float64` `trace` every `record_every` steps
(pass `record_every <= 0` to disable). Returns `(num_accepted, trace, sampling_iter)` with the
final `sampling_iter` (the next particle to move), so a caller can resume the sweep seamlessly.
"""
function gibbs_sampling!(RNG, Ψcurrent::T, Ψnext::T,
                         θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64},
                         θnext::Vector{Float64}, ϕnext::Vector{Float64},
                         σ::Float64, sampling_iter::Int64,
                         logpdf::Function, num_steps::Int64;
                         measure!::Function = (Ψ, θ, ϕ, accepted, sampling_iter) -> nothing,
                         record_every::Int64 = 1000,
                         checkpoint::Function = (step, acceptance_rate, σ, trace) -> nothing,
                         checkpoint_every::Int64 = typemax(Int64)) where {T <: Union{Ψproj, Ψparton, ΨoneLL, Ψunproj}}

    N::Int64 = Ψcurrent.system_size
    num_samples_accepted::Int64 = 0

    logpdf_current::Float64 = logpdf(Ψcurrent)
    logpdf_next::Float64 = logpdf_current

    trace = Float64[]

    for monte_carlo_iter in 1:num_steps

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        logpdf_next = logpdf(Ψnext)

        accepted::Bool = logpdf_next - logpdf_current >= log(rand(RNG))

        if accepted
            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]
        else
            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]
        end

        # Measure BEFORE re-syncing the buffers: on an accepted step Ψnext holds the new
        # configuration and Ψcurrent still holds the pre-move one (see docstring).
        measure!(accepted ? Ψnext : Ψcurrent, θcurrent, ϕcurrent, accepted, sampling_iter)

        if accepted
            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next
            num_samples_accepted += 1
        else
            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current
        end

        if record_every > 0 && mod(monte_carlo_iter, record_every) == 0
            push!(trace, logpdf_current)
        end

        sampling_iter = mod(sampling_iter, N) + 1

        if monte_carlo_iter == num_steps || mod(monte_carlo_iter, checkpoint_every) == 0
            checkpoint(monte_carlo_iter, num_samples_accepted / monte_carlo_iter, σ, trace)
        end

    end

    return num_samples_accepted, trace, sampling_iter
end

const CHAIN_SCHEMA_VERSION = 1

"""
    _merge_observables(prior, prior_steps, new, new_steps) -> Dict

Step-weighted merge of two user-observable Dicts (see the `observables` contract in
[`run_chains`](@ref)). The `"observables"` sub-dict holds per-step averages, combined as
`(prior_steps·prior + new_steps·new) / (prior_steps + new_steps)`; the `"grids"` sub-dict is
static and carried over from `new`.
"""
function _merge_observables(prior::AbstractDict, prior_steps::Int, new::AbstractDict, new_steps::Int)
    tot = prior_steps + new_steps
    merged = Dict{String,Any}()
    pobs = prior["observables"]; nobs = new["observables"]
    for (k, v) in nobs
        pv = pobs[k]
        merged[k] = (prior_steps .* pv .+ new_steps .* v) ./ tot
    end
    out = Dict{String,Any}("observables" => merged)
    haskey(new, "grids") && (out["grids"] = new["grids"])
    return out
end

"""
    _run_one_chain(chain_id, setup, dir, seed, num_thermalization, num_steps,
                   record_every, checkpoint_every, resume) -> NamedTuple

Worker-side driver for a single chain: build the per-chain RNG `Xoshiro(hash((seed, chain_id)))`,
call `setup(chain_id, RNG)`, thermalize (or restore from `dir/chain_\$(chain_id).jld2` when
`resume` and a checkpoint exists), run [`gibbs_sampling!`](@ref) for the remaining steps, and
persist checkpoints. Returns a summary NamedTuple `(chain, steps, acceptance, sigma, mean_logpdf)`.
Not exported; called through `run_chains`.
"""
function _run_one_chain(chain_id::Int, setup::Function, dir::String, seed,
                        num_thermalization::Int, num_steps::Int,
                        record_every::Int, checkpoint_every::Int, resume::Bool)

    chain_seed = hash((seed, chain_id))
    RNG = Xoshiro(chain_seed)

    cfg = setup(chain_id, RNG)
    Ψcurrent = cfg.Ψcurrent
    Ψnext    = cfg.Ψnext
    θcurrent = cfg.θ
    ϕcurrent = cfg.ϕ
    logpdf   = cfg.logpdf
    measure! = cfg.measure!
    observables = cfg.observables

    N = Ψcurrent.system_size
    θnext = copy(θcurrent)
    ϕnext = copy(ϕcurrent)

    path = joinpath(dir, "chain_$(chain_id).jld2")

    prior_obs = nothing
    prior_trace = Float64[]
    prior_steps = 0
    prior_duration = 0.0
    σ = π / sqrt(12.0)
    sampling_iter = 1
    therm_accept = NaN
    δt_therm = NaN
    num_thermalization_done = num_thermalization

    if resume && isfile(path)
        d = load(path)
        θcurrent .= d["theta vector"]
        ϕcurrent .= d["phi vector"]
        θnext .= θcurrent
        ϕnext .= ϕcurrent
        σ = d["step size"]
        sampling_iter = d["sampling iter"]
        prior_steps = d["number of steps"]
        prior_obs = get(d, "user observables", nothing)
        prior_trace = Float64.(get(d, "logpdf trace", Float64[]))
        prior_duration = get(d, "monte carlo duration", 0.0)
        num_thermalization_done = get(d, "number of thermalization steps", num_thermalization)
        therm_accept = get(d, "thermalization acceptance rate", NaN)
        δt_therm = get(d, "thermalization duration", NaN)
        update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
        copy!(Ψnext, Ψcurrent)
    else
        update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
        copy!(Ψnext, Ψcurrent)
        sampling_iter, σ, δt_therm, therm_accept =
            gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, num_thermalization)
    end

    si0 = sampling_iter                       # initial index for this run's sweep
    remaining = num_steps - prior_steps

    t0 = time()

    write_checkpoint = function (step, acceptance_rate, σcur, trace)
        total_steps = prior_steps + step
        # Next particle to move after `step` sweeps from si0 (each advances the index once).
        next_iter = mod(si0 - 1 + step, N) + 1
        newobs = observables(step)
        obs = prior_obs === nothing ? newobs : _merge_observables(prior_obs, prior_steps, newobs, step)
        full_trace = isempty(prior_trace) ? trace : vcat(prior_trace, trace)
        data = Dict{String,Any}(
            "schema version"                => CHAIN_SCHEMA_VERSION,
            "chain id"                      => chain_id,
            "chain seed"                    => chain_seed,
            "theta vector"                  => copy(θcurrent),
            "phi vector"                    => copy(ϕcurrent),
            "step size"                     => σcur,
            "sampling iter"                 => next_iter,
            "number of steps"               => total_steps,
            "number of thermalization steps"=> num_thermalization_done,
            "thermalization acceptance rate"=> therm_accept,
            "thermalization duration"       => δt_therm,
            "acceptance rate"               => acceptance_rate,
            "logpdf trace"                  => full_trace,
            "record every"                  => record_every,
            "monte carlo duration"          => prior_duration + (time() - t0),
            "user observables"              => obs,
        )
        save(path, data)
        return
    end

    if remaining <= 0
        # Nothing to do (already complete); rewrite the checkpoint to normalize metadata.
        num_accepted = 0
        trace = Float64[]
        acc_rate = NaN
    else
        num_accepted, trace, sampling_iter = gibbs_sampling!(
            RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, si0, logpdf, remaining;
            measure! = measure!, record_every = record_every,
            checkpoint = write_checkpoint, checkpoint_every = checkpoint_every)
        acc_rate = num_accepted / remaining
    end

    total_steps = prior_steps + max(remaining, 0)
    mean_lp = isempty(trace) ? NaN : sum(trace) / length(trace)
    return (chain = chain_id, steps = total_steps, acceptance = acc_rate, sigma = σ, mean_logpdf = mean_lp)
end

"""
    run_chains(setup; n_chains, dir, seed = 0, num_thermalization = 500_000, num_steps = 1_000_000,
               record_every = 1000, checkpoint_every = 500_000, resume = false,
               monitor_every = 10.0) -> ChainStatusReport

Run `n_chains` independent MCMC chains in parallel over the available `Distributed` workers,
writing one checkpoint file `dir/chain_<id>.jld2` per chain, and print a live convergence table
while they run. Returns the final [`monitor_chains`](@ref) report.

The sampling algorithm is identical to `examples/sampler_single_state.jl` — this only *drives*
many independent copies of it.

# The `setup` contract
`setup(chain_id::Int, RNG)` runs **on the worker** and returns a `NamedTuple` with fields:
- `Ψcurrent`, `Ψnext` — the two wavefunction buffers, **constructed worker-side** (they hold
  large caches; never serialize them across workers — build them here instead).
- `θ`, `ϕ` — initial position vectors of length `N = Ψcurrent.system_size` (e.g.
  `rand_θ_ϕ_gen(RNG, N)`).
- `logpdf::Function` — `ψ -> ::Real`, the log sampling weight.
- `measure!::Function` — the per-step accumulator with the [`gibbs_sampling!`](@ref) signature.
- `observables::Function` — a finalizer `observables(n::Int) -> Dict{String,Any}` returning the
  user observables accumulated over `n` measurement steps. Use the reserved layout
  `Dict("observables" => Dict(name => per_step_average), "grids" => Dict(name => static_grid))`:
  entries under `"observables"` are per-step averages that [`combine_chains`](@ref) pools by a
  step-weighted mean; entries under `"grids"` are static (identical across chains) and are not
  pooled. This layout also lets a resumed run merge its observables correctly.

# Determinism
Per-chain RNG is `Xoshiro(hash((seed, chain_id)))`, so a given `(seed, chain_id)` reproduces the
exact same chain regardless of how chains are scheduled onto workers.

# Distribution
Chains are dispatched with `pmap` over a `CachingPool(workers())`. If `nprocs() == 1` there are
no worker processes, so the chains run **sequentially in-process** (a note is printed). Otherwise
you **must** load the package on every worker first:

```julia
using Distributed
addprocs(4)
@everywhere using CFsOnSphere
run_chains(setup; n_chains = 4, dir = "run")
```

`run_chains` checks that every worker has `CFsOnSphere` loaded and raises an informative error if
`@everywhere using CFsOnSphere` was forgotten (which otherwise surfaces as an `UndefVarError` on
the workers).
"""
function run_chains(setup::Function; n_chains::Int, dir::String, seed = 0,
                    num_thermalization::Int = 500_000, num_steps::Int = 1_000_000,
                    record_every::Int = 1000, checkpoint_every::Int = 500_000,
                    resume::Bool = false, monitor_every::Real = 10.0)

    mkpath(dir)

    run_one(cid) = _run_one_chain(cid, setup, dir, seed, num_thermalization, num_steps,
                                  record_every, checkpoint_every, resume)

    if nprocs() == 1
        println("[run_chains] nprocs() == 1: no worker processes; running $(n_chains) chains " *
                "sequentially in-process. Call addprocs(...) and `@everywhere using CFsOnSphere` " *
                "for parallel execution.")
        for cid in 1:n_chains
            run_one(cid)
            print(monitor_chains(dir))
        end
    else
        # Fail loudly and early if the package is not loaded on the workers.
        for w in workers()
            loaded = try
                remotecall_fetch(() -> isdefined(Main, :CFsOnSphere), w)
            catch err
                error("Could not reach worker $w while checking for CFsOnSphere: $err")
            end
            loaded || error("Worker $w does not have CFsOnSphere loaded. Run " *
                            "`@everywhere using CFsOnSphere` before calling run_chains.")
        end

        pool = CachingPool(workers())
        try
            task = @async pmap(run_one, pool, 1:n_chains)
            while !istaskdone(task)
                sleep(monitor_every)
                try
                    print(monitor_chains(dir))
                catch
                    # Checkpoints may be mid-write or not yet created; skip this tick.
                end
            end
            fetch(task)   # rethrows any worker exception
        finally
            clear!(pool)
        end
    end

    return monitor_chains(dir)
end

"""
    ChainStatusReport

Snapshot of a multi-chain run read from the per-chain checkpoints in a directory: per-chain
`(steps, acceptance, σ, mean logpdf)` plus the cross-chain split-R̂ and multi-chain ESS of the
logpdf traces. `print`/`show` render a plain-text table readable in a Slurm log. Produced by
[`monitor_chains`](@ref).
"""
struct ChainStatusReport
    dir::String
    chain_ids::Vector{Int}
    steps::Vector{Int}
    acceptance::Vector{Float64}
    sigma::Vector{Float64}
    mean_logpdf::Vector{Float64}
    rhat::Float64
    ess::Float64
    n_read::Int
end

function Base.show(io::IO, r::ChainStatusReport)
    println(io, "Chain status  (dir = ", r.dir, ",  ", r.n_read, " chain(s) read)")
    if r.n_read == 0
        println(io, "  (no readable checkpoints yet)")
        return
    end
    @printf(io, "  %-8s %14s %12s %12s %16s\n", "chain", "steps", "accept", "σ", "mean logpdf")
    println(io, "  ", repeat("-", 64))
    for i in eachindex(r.chain_ids)
        @printf(io, "  %-8d %14d %12.4f %12.5f %16.4f\n",
                r.chain_ids[i], r.steps[i], r.acceptance[i], r.sigma[i], r.mean_logpdf[i])
    end
    println(io, "  ", repeat("-", 64))
    @printf(io, "  split-R̂ (logpdf) = %.4f     multi-chain ESS = %.1f\n", r.rhat, r.ess)
    return
end

Base.show(io::IO, ::MIME"text/plain", r::ChainStatusReport) = show(io, r)

"""
    _read_chain_checkpoints(dir) -> Vector{Dict}

Load every `chain_*.jld2` in `dir`, sorted by chain id, skipping files that fail to load (a
checkpoint may be momentarily unreadable while a worker is mid-write). Each load is retried once
after a short pause before being skipped.
"""
function _read_chain_checkpoints(dir::String)
    isdir(dir) || return Dict{String,Any}[]
    files = filter(f -> occursin(r"^chain_\d+\.jld2$", basename(f)),
                   readdir(dir; join = true))
    out = Dict{String,Any}[]
    for f in sort(files)
        d = nothing
        for attempt in 1:2
            try
                d = load(f)
                break
            catch
                attempt == 1 ? sleep(0.05) : (d = nothing)
            end
        end
        d === nothing || push!(out, d)
    end
    return out
end

"""
    monitor_chains(dir) -> ChainStatusReport

Read all `chain_*.jld2` checkpoints in `dir` and report per-chain `(steps, acceptance, σ, mean
logpdf)` together with the cross-chain **split-R̂** ([`split_rhat`](@ref)) and multi-chain
**ESS** ([`effective_sample_size`](@ref)) of the recorded logpdf traces (trimmed to their common
length). Works on a partially written run: files that are unreadable (e.g. mid-write) are skipped,
and R̂/ESS are `NaN` until at least two chains with usable traces are present. The returned report
`print`s as a plain-text table.
"""
function monitor_chains(dir::String)
    ds = _read_chain_checkpoints(dir)

    chain_ids = Int[]
    steps = Int[]
    acceptance = Float64[]
    sigma = Float64[]
    mean_logpdf = Float64[]
    traces = Vector{Float64}[]

    for d in ds
        push!(chain_ids, get(d, "chain id", -1))
        push!(steps, get(d, "number of steps", 0))
        push!(acceptance, get(d, "acceptance rate", NaN))
        push!(sigma, get(d, "step size", NaN))
        tr = get(d, "logpdf trace", Float64[])
        push!(mean_logpdf, isempty(tr) ? NaN : sum(tr) / length(tr))
        isempty(tr) || push!(traces, Float64.(tr))
    end

    rhat = NaN
    ess = NaN
    if length(traces) >= 1
        L = minimum(length, traces)
        trimmed = [t[1:L] for t in traces]
        rhat = split_rhat(trimmed)
        ess = effective_sample_size(trimmed)
    end

    return ChainStatusReport(dir, chain_ids, steps, acceptance, sigma, mean_logpdf,
                             rhat, ess, length(ds))
end

"""
    combine_chains(dir) -> Dict

Pool the per-chain checkpoints in `dir` into combined observables and diagnostics. Returns a
`Dict` with:
- `"observables"` — pooled user observables: each per-step-average entry is combined across
  chains by a **step-weighted mean** `Σ steps_c · obs_c / Σ steps_c` (equivalent to pooling raw
  counts and dividing by the pooled step count).
- `"grids"` — the static grids, taken from the first chain.
- `"per_chain"` — the per-chain observable values, so you can inspect chain-to-chain scatter.
- `"standard_error"` — for **scalar** observables, the cross-chain standard error
  `std(per-chain means)/√(n_chains)`.
- `"n_chains"`, `"total_steps"`, and `"diagnostics"` — a `(rhat, ess)` NamedTuple from the
  logpdf traces (as in [`monitor_chains`](@ref)).
"""
function combine_chains(dir::String)
    ds = _read_chain_checkpoints(dir)
    isempty(ds) && error("combine_chains: no readable chain_*.jld2 checkpoints in $dir")

    steps_per = [get(d, "number of steps", 0) for d in ds]
    total_steps = sum(steps_per)
    n_chains = length(ds)

    obs_dicts = [d["user observables"]["observables"] for d in ds]
    names = collect(keys(obs_dicts[1]))

    pooled = Dict{String,Any}()
    per_chain = Dict{String,Any}()
    standard_error = Dict{String,Float64}()

    for name in names
        vals = [obs_dicts[c][name] for c in 1:n_chains]
        per_chain[name] = vals
        # Step-weighted pool (works elementwise for arrays and for scalars).
        acc = steps_per[1] .* vals[1]
        for c in 2:n_chains
            acc = acc .+ steps_per[c] .* vals[c]
        end
        pooled[name] = acc ./ total_steps
        # Cross-chain standard error for scalar observables.
        if all(v -> v isa Number, vals) && n_chains >= 2
            standard_error[name] = std(Float64.(vals)) / sqrt(n_chains)
        end
    end

    grids = haskey(ds[1]["user observables"], "grids") ? ds[1]["user observables"]["grids"] : Dict{String,Any}()

    rep = monitor_chains(dir)

    return Dict{String,Any}(
        "observables"    => pooled,
        "grids"          => grids,
        "per_chain"      => per_chain,
        "standard_error" => standard_error,
        "n_chains"       => n_chains,
        "total_steps"    => total_steps,
        "diagnostics"    => (rhat = rep.rhat, ess = rep.ess),
    )
end

"""
    sample_density_chains(dir; Qstar, l_m_list, p, n_chains, kwargs...) -> Dict
    sample_density_chains(dir, N, n, p; n_chains, kwargs...) -> Dict

Convenience multi-chain driver reproducing the density and pair-correlation observables of
`examples/sampler_single_state.jl` for the projected CF state `Ψproj(Qstar, p, N, l_m_list)`,
using the same meshes and normalizations: a 500-point equal-area `acos(LinRange(1, -1, 500))`
polar-angle mesh for the density `n(θ)` (divided by the per-bin area) and a 5000-point
`LinRange(0, 2, 5000)` chord-distance grid for the pair density (each accumulated exactly as in
the example — the pair histogram updated only on acceptance, accumulated every step). Runs
`n_chains` chains via [`run_chains`](@ref) and returns [`combine_chains`](@ref)`(dir)`; the pooled
`"observables"["density"]` is directly comparable to the single-chain reference data.

The `(N, n, p)` method builds `(Qstar, l_m_list)` with [`cf_ground_state_lm`](@ref). Keyword
arguments (`seed`, `num_thermalization`, `num_steps`, `record_every`, `checkpoint_every`,
`resume`, `monitor_every`) are forwarded to [`run_chains`](@ref).
"""
function sample_density_chains(dir::String; Qstar::Rational{Int64},
                               l_m_list::Vector{NTuple{2,Rational{Int64}}}, p::Int64,
                               n_chains::Int, kwargs...)

    setup = function (chain_id::Int, RNG)
        N = length(l_m_list)
        Ψcurrent = Ψproj(Qstar, p, N, l_m_list)
        Ψnext    = Ψproj(Qstar, p, N, l_m_list)
        θ, ϕ = rand_θ_ϕ_gen(RNG, N)

        logpdf(ψ::Ψproj) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

        # Density: equal-area (uniform in cosθ) polar-angle histogram.
        θmesh = map(x -> acos(x), LinRange(1.0, -1.0, 500))
        Agrid = 2.0 * pi .* (cos.(θmesh[begin:end-1]) .- cos.(θmesh[begin+1:end]))
        accumulated_density = zeros(Float64, length(θmesh) - 1)
        θgrid = 0.50 .* (θmesh[1:end-1] .+ θmesh[2:end])

        # Pair correlation: chord-distance histogram, maintained incrementally on acceptance.
        rgrid = collect(LinRange(0.0, 2.0, 5_000))
        dr = rgrid[2] - rgrid[1]
        accumulated_pair_density = zeros(Float64, length(rgrid) - 1)
        current_distance_distribution = zeros(Float64, length(rgrid) - 1)
        rgrid_centers = 0.50 .* (rgrid[1:end-1] .+ rgrid[2:end])
        initialized = Ref(false)

        measure! = function (Ψ, θc, ϕc, accepted::Bool, sampling_iter)
            if !initialized[]
                # First measurement: build the full pair histogram of the current config.
                for i in 1:N-1
                    for j in i+1:N
                        current_distance_distribution[ceil(Int64, Ψ.dist_matrix[j-1, i] / dr)] += 1.0
                    end
                end
                initialized[] = true
            elseif accepted
                # Only particle `sampling_iter` moved: swap its N-1 distances in the histogram.
                for i in 1:N-1
                    @inbounds current_distance_distribution[ceil(Int64, Ψnext.dist_matrix[i, sampling_iter] / dr)] += 1.0
                    @inbounds current_distance_distribution[ceil(Int64, Ψcurrent.dist_matrix[i, sampling_iter] / dr)] -= 1.0
                end
            end
            accumulated_pair_density .+= current_distance_distribution
            update_density!(θmesh, θc, accumulated_density)
            return
        end

        observables = function (n::Int)
            return Dict{String,Any}(
                "observables" => Dict{String,Any}(
                    "density"        => accumulated_density ./ n ./ Agrid,
                    "pair densities" => accumulated_pair_density ./ n,
                ),
                "grids" => Dict{String,Any}(
                    "theta grid" => θgrid,
                    "r grid"     => rgrid_centers,
                ),
            )
        end

        return (Ψcurrent = Ψcurrent, Ψnext = Ψnext, θ = θ, ϕ = ϕ,
                logpdf = logpdf, measure! = measure!, observables = observables)
    end

    run_chains(setup; n_chains = n_chains, dir = dir, kwargs...)
    return combine_chains(dir)
end

function sample_density_chains(dir::String, N::Int64, n::Int64, p::Int64; n_chains::Int, kwargs...)
    Qstar, l_m_list = cf_ground_state_lm(N, n, p)
    return sample_density_chains(dir; Qstar = Qstar, l_m_list = l_m_list, p = p,
                                 n_chains = n_chains, kwargs...)
end
