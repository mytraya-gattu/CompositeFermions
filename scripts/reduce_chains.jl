#!/usr/bin/env julia
# Average spherical-harmonic density coefficients across MC chains.
#
# Usage:
#   julia --project=<CF_PROJECT> scripts/reduce_chains.jl \
#       --chains-dir ./chains_step_1 \
#       --num-chains 100 \
#       --out density_step_1.jld2 \
#       [--N 19]
#
# Each chain file is expected to have keys l, m, n_lm (and optionally N).
# Outputs a JLD2 readable by CF-DFT's load_density_lm.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2

function parse_args(args)
    d = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("expected --flag, got $key")
        i == length(args) && error("missing value for $key")
        d[key[3:end]] = args[i + 1]; i += 2
    end
    return d
end

args       = parse_args(ARGS)
chains_dir = get(args, "chains-dir", ".")
nchains    = parse(Int, get(args, "num-chains", "1"))
outfile    = get(args, "out", "density_combined.jld2")
N_override = haskey(args, "N") ? parse(Int, args["N"]) : nothing

# ---------------------------------------------------------------- load and average

# Collect all (l,m) → sum of n_lm across chains
mode_sums = Dict{Tuple{Int,Int}, ComplexF64}()
mode_count = 0
N_total = 0.0
acc_rates = Float64[]
nloaded = 0

for chain in 1:nchains
    f = joinpath(chains_dir, "chain_$(chain).jld2")
    isfile(f) || (println("WARNING: chain file $f not found, skipping"); continue)
    d = JLD2.load(f)
    ls = d["l"]::Vector{Int}
    ms = d["m"]::Vector{Int}
    nlm = d["n_lm"]::Vector{ComplexF64}
    for i in eachindex(ls)
        key = (ls[i], ms[i])
        mode_sums[key] = get(mode_sums, key, 0.0+0.0im) + nlm[i]
    end
    if haskey(d, "N"); N_total += Float64(d["N"]); end
    if haskey(d, "acceptance_rate"); push!(acc_rates, d["acceptance_rate"]); end
    nloaded += 1
end

nloaded > 0 || error("no chain files found in $chains_dir")
println("loaded $nloaded / $nchains chains")

# Average
for key in keys(mode_sums)
    mode_sums[key] /= nloaded
end

N_val = if !isnothing(N_override)
    Float64(N_override)
elseif N_total > 0
    N_total / nloaded
else
    error("cannot determine N; pass --N explicitly")
end

# Enforce normalization: n_00 must equal N / sqrt(4π)
expected_n00 = N_val / sqrt(4 * pi)
actual_n00   = real(get(mode_sums, (0, 0), 0.0+0.0im))
if abs(actual_n00) > 1e-15
    scale = expected_n00 / actual_n00
    println("rescaling n_lm by $(round(scale, digits=6)) to enforce normalization")
    for key in keys(mode_sums)
        mode_sums[key] *= scale
    end
end

# Enforce reality: n_{l,-m} = (-1)^m conj(n_{l,m})
lmax = maximum(first, keys(mode_sums))
for l in 1:lmax, m in 1:l
    vp = get(mode_sums, (l, m),  0.0+0.0im)
    vm = get(mode_sums, (l, -m), 0.0+0.0im)
    sym = ((-1)^m * conj(vp) + vm) / 2
    mode_sums[(l, -m)] = sym
    mode_sums[(l,  m)] = (-1)^m * conj(sym)
end

# Build output arrays sorted by (l, m)
sorted_keys = sort(collect(keys(mode_sums)), by = x -> (x[1], x[2]))
ls_out  = [k[1] for k in sorted_keys]
ms_out  = [k[2] for k in sorted_keys]
nlm_out = [mode_sums[k] for k in sorted_keys]

# ---------------------------------------------------------------- save

avg_acc = isempty(acc_rates) ? NaN : sum(acc_rates) / length(acc_rates)
JLD2.jldsave(outfile;
    l    = ls_out,
    m    = ms_out,
    n_lm = nlm_out,
    N    = N_val,
    metadata = "reduced from $nloaded chains in $chains_dir",
    num_chains_loaded = nloaded,
    mean_acceptance   = avg_acc,
)
println("wrote combined density to $outfile")
println("  lmax=$(lmax)  N=$(round(N_val, digits=4))  n_00=$(round(nlm_out[1], digits=6))  mean_acc=$(round(avg_acc, digits=3))")
