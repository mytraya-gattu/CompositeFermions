#!/usr/bin/env julia
# Single-chain CFD worker: samples |Ψ_guide|² and accumulates the overlap
# matrix S and potential matrix W over K determinant configurations sharing
# the walker (see cf_full_dft/notes/cfd_design.md).
#
# Usage:
#   julia --project=<CF_PROJECT> scripts/cfd_chain_worker.jl \
#       --N 6 --twoQ 15 --p 1 --nmax 2 \
#       --orbital-file orbitals.jld2 --configs-file configs.jld2 \
#       [--disorder-file disorder.jld2] [--stream-file stream.jld2] \
#       --chain 1 --steps 1000000 --therm 200000 --stride 0 \
#       --out cfd_chain_1.jld2
#
# --stride 0 (default) means one sweep (N single-particle moves) between
# accumulated samples.
#
# --stream-file: ideal (Aharonov-Casher) band. The file holds the stream
# function chi in the cf-psiformer real-harmonic layout (key `chi_real`,
# unit-sphere harmonics, chi in magnetic-length units). Every configuration
# carries the common dress exp(sum_i chi(Omega_i)); it cancels in all ratios
# r_c and enters only the sampling weight, |Psi_guide|^2 exp(2 sum chi).
# No one-body energy is added: the AC-tuned one-body energy is constant on
# the band.
#
# The orbital file must contain norb ≥ N eigenvector columns (run
# run_orbital_step.jl with --nev norb). The guide ΨprojDFT uses the columns of
# the guide configuration; all K configurations are evaluated from the guide's
# raw_slater_det.
#
# Output JLD2: S_sum, W_sum (K×K), n_samples, sum_r2, sum_r4, configs, guide,
# norb, params, acceptance, timings. Reduce with
# cf_full_dft/scripts/reduce_cfd_chains.jl.

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CFsOnSphere
using Random
using JLD2

# ---------------------------------------------------------------- arg parsing

function parse_args(args)
    d = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("expected --flag, got $key")
        i == length(args) && error("missing value for $key")
        d[key[3:end]] = args[i + 1]
        i += 2
    end
    return d
end

function req(d, k); haskey(d, k) || error("missing --$k"); return d[k]; end

args    = parse_args(ARGS)
N       = parse(Int, req(args, "N"))
twoQ    = parse(Int, req(args, "twoQ"))
p       = parse(Int, req(args, "p"))
nmax    = parse(Int, req(args, "nmax"))
chain   = parse(Int, req(args, "chain"))
nsteps  = parse(Int, get(args, "steps",  "1000000"))
ntherm  = parse(Int, get(args, "therm",  "200000"))
stride  = parse(Int, get(args, "stride", "0"))
stride  = stride <= 0 ? N : stride
outfile = get(args, "out", "cfd_chain_$(chain).jld2")
orbital_file  = req(args, "orbital-file")
configs_file  = req(args, "configs-file")
disorder_file = get(args, "disorder-file", "")
stream_file   = get(args, "stream-file", "")

Qstar = twoQ // 2 - p * (N - 1)

RNG = Random.MersenneTwister(chain)
# gibbs_thermalization! draws its acceptance tests from the global RNG: seed it too, so a
# chain is reproducible from --chain alone.
Random.seed!(1_000_003 * chain)

# ---------------------------------------------------------------- orbitals & configs

println("loading DFT orbitals from: $orbital_file")
Qstar_orb, p_orb, N_orb, nmax_orb, C_all = load_dft_orbitals(orbital_file)
@assert Qstar_orb == Qstar && p_orb == p && N_orb == N "orbital file params mismatch"
@assert nmax_orb == nmax "nmax mismatch: orbital file has $nmax_orb, requested $nmax"

cfgdata = JLD2.load(configs_file)
norb    = cfgdata["norb"]::Int
guide   = cfgdata["guide"]::Int
configs_mat = cfgdata["configs"]::Matrix{Int}   # K × N
configs = [sort(configs_mat[k, :]) for k in axes(configs_mat, 1)]
K = length(configs)

size(C_all, 2) >= norb || error("orbital file has $(size(C_all, 2)) orbitals, configs need norb=$norb (rerun run_orbital_step.jl with --nev $norb)")
C_all = C_all[:, 1:norb]

cfg = CFDConfigs(norb, configs, guide)
println("K = $K configurations over norb = $norb orbitals, guide = configuration $guide")

# ---------------------------------------------------------------- disorder

dis_l, dis_m, dis_V_lm = Int[], Int[], ComplexF64[]
dis_meta = ""
if !isempty(disorder_file)
    d = JLD2.load(disorder_file)
    println("loaded disorder potential from: $disorder_file")
    dis_l    = Vector{Int}(d["l"])
    dis_m    = Vector{Int}(d["m"])
    dis_V_lm = Vector{ComplexF64}(d["V_lm"])
    dis_meta = get(d, "metadata", "")
end
fe = HarmonicFieldEvaluator(dis_l, dis_m, dis_V_lm)

# ---------------------------------------------------------------- AC dress

stream_meta = Dict{String, Any}()
log_weight = nothing
if !isempty(stream_file)
    sd = JLD2.load(stream_file)
    chi_field = RealHarmonicField(Vector{Float64}(sd["chi_real"]))
    log_weight = (θ, ϕ) -> 2.0 * field_value(chi_field, θ, ϕ)
    for k in ("xi", "lmax", "field_scale", "R", "texture_sha256", "texture_path")
        haskey(sd, k) && (stream_meta[k] = sd[k])
    end
    println("loaded AC stream from: $stream_file  ($(stream_meta))")
end

# ---------------------------------------------------------------- guide wavefunction

guide_coeffs = Matrix{ComplexF64}(C_all[:, cfg.configs[guide]])
Ψcurrent = ΨprojDFT(Qstar, 2p, N, nmax, guide_coeffs)
Ψnext    = ΨprojDFT(Qstar, 2p, N, nmax, guide_coeffs)
logpdf(ψ::ΨprojDFT) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

acc = CFDAccumulator(cfg, Ψcurrent.dim_basis, N)

# ---------------------------------------------------------------- positions & thermalization

θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, N)
θnext = copy(θcurrent)
ϕnext = copy(ϕcurrent)
σ = pi / sqrt(12.0)

println("thermalizing ($ntherm steps) ...")
sampling_iter, σ, δt_therm, acc_therm = gibbs_thermalization!(
    RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, ntherm;
    log_weight = log_weight)
println("  acceptance rate: $(round(acc_therm, digits=3))  step size: $(round(σ, digits=4))  time: $(round(δt_therm, digits=1))s")

# ---------------------------------------------------------------- production

logpdf_current = logpdf(Ψcurrent)
wcache = log_weight === nothing ? zeros(N) : [log_weight(θcurrent[i], ϕcurrent[i]) for i in 1:N]
num_accepted   = 0
t0 = time()

for mc_iter in 1:nsteps
    global sampling_iter, logpdf_current, num_accepted

    θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
    update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)
    logpdf_next = logpdf(Ψnext)
    wnew = log_weight === nothing ? 0.0 : log_weight(θnext[sampling_iter], ϕnext[sampling_iter])

    if (logpdf_next - logpdf_current) + (wnew - wcache[sampling_iter]) >= log(rand(RNG))
        θcurrent[sampling_iter] = θnext[sampling_iter]
        ϕcurrent[sampling_iter] = ϕnext[sampling_iter]
        copy!(Ψcurrent, Ψnext, sampling_iter)
        logpdf_current = logpdf_next
        wcache[sampling_iter] = wnew
        num_accepted  += 1
    else
        θnext[sampling_iter] = θcurrent[sampling_iter]
        ϕnext[sampling_iter] = ϕcurrent[sampling_iter]
        copy!(Ψnext, Ψcurrent, sampling_iter)
    end

    if mod(mc_iter, stride) == 0
        V_loc = coulomb_local_energy(Ψcurrent, twoQ) + field_local_energy(fe, θcurrent, ϕcurrent)
        accumulate_cfd!(acc, Ψcurrent, C_all, V_loc)
    end

    sampling_iter = mod(sampling_iter, N) + 1

    if mc_iter == nsteps || mod(mc_iter, 500_000) == 0
        println("  step $mc_iter / $nsteps  acceptance=$(round(num_accepted/mc_iter, digits=3))  samples=$(acc.n_samples)")
    end
end

δt_prod = time() - t0
println("production done in $(round(δt_prod, digits=1))s  acceptance=$(round(num_accepted/nsteps, digits=3))  samples=$(acc.n_samples)")

# effective sample size per configuration (guide-coverage diagnostic)
ess = [acc.sum_r2[k]^2 / max(acc.sum_r4[k], eps()) for k in 1:K]
println("min/median ESS over configs: $(round(minimum(ess), digits=1)) / $(round(sort(ess)[cld(K, 2)], digits=1))  (n_samples=$(acc.n_samples))")

# ---------------------------------------------------------------- save

JLD2.jldsave(outfile;
    S_sum = acc.S,
    W_sum = acc.W,
    n_samples = acc.n_samples,
    sum_r2 = acc.sum_r2,
    sum_r4 = acc.sum_r4,
    ess = ess,
    configs = configs_mat,
    guide = guide,
    norb = norb,
    N = N, twoQ = twoQ, p = p, nmax = nmax,
    stride = stride,
    disorder_file = disorder_file,
    disorder_l = dis_l,
    disorder_m = dis_m,
    disorder_V_lm = dis_V_lm,
    disorder_metadata = dis_meta,
    stream_file = stream_file,
    stream_metadata = stream_meta,
    orbital_file = orbital_file,
    acceptance_rate = num_accepted / nsteps,
    thermalization_acceptance = acc_therm,
    production_time = δt_prod,
    therm_time = δt_therm,
    metadata = "cfd_chain_worker chain=$chain nsteps=$nsteps stride=$stride K=$K norb=$norb",
)
println("wrote CFD chain to $outfile")
