#!/usr/bin/env julia
# Single-chain MC worker for the MC-DFT self-consistency loop.
#
# Usage:
#   julia --project=<CF_PROJECT> scripts/mc_chain_worker.jl \
#       --N 19 --twoQ 55 --p 1 --nmax 5 \
#       --chain 1 --steps 1000000 --therm 500000 \
#       --out chain_1.jld2 \
#       [--orbital-file orbitals_step_k.jld2]
#
# If --orbital-file is supplied the wavefunction uses ΨprojDFT (general
# superpositions of Lambda levels as returned by CF-DFT); otherwise the
# standard Ψproj with the ground-state l_m_list is used (useful for the
# first iteration bootstrap).
#
# Output: a JLD2 file readable by CF-DFT's load_density_lm, containing
# the spherical-harmonic density coefficients n_lm accumulated online
# during the production run.

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
ntherm  = parse(Int, get(args, "therm",  "500000"))
outfile = get(args, "out", "chain_$(chain).jld2")
orbital_file = get(args, "orbital-file", "")

# ---------------------------------------------------------------- derived params

Qstar = twoQ // 2 - p * (N - 1)   # Rational{Int64}
qabs  = abs(Qstar)
density_lmax = Int(2 * (numerator(qabs) + nmax * denominator(qabs)))
# For half-integer Qstar: qabs = a//2, so 2*qabs = a, density_lmax = a + 2*nmax
density_lmax = 2 * (Int(numerator(2 * qabs)) ÷ denominator(2 * qabs) ÷ 2 + nmax)
# simpler: density_lmax = 2*(|Qstar| + nmax) computed as rational and converted
density_lmax = round(Int, 2 * (Float64(qabs) + nmax))

RNG = Random.MersenneTwister(chain)  # reproducible per chain

# ---------------------------------------------------------------- wavefunction

use_dft = !isempty(orbital_file)

local Ψcurrent, Ψnext, logpdf

if use_dft
    println("loading DFT orbitals from: $orbital_file")
    Qstar_orb, p_orb, N_orb, nmax_orb, coeffs = load_dft_orbitals(orbital_file)
    @assert Qstar_orb == Qstar && p_orb == p && N_orb == N "orbital file params mismatch"
    @assert nmax_orb == nmax "nmax mismatch: orbital file has $nmax_orb, requested $nmax"
    # orbital files made for CFD carry norb > N eigenvectors; the DFT density
    # loop always fills the N lowest
    size(coeffs, 2) >= N || error("orbital file has fewer than N eigenvectors")
    coeffs = coeffs[:, 1:N]
    Ψcurrent = ΨprojDFT(Qstar, 2p, N, nmax, coeffs)
    Ψnext    = ΨprojDFT(Qstar, 2p, N, nmax, coeffs)
    logpdf(ψ::ΨprojDFT) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)
else
    println("no orbital file; using ground-state l_m_list for $N particles at Q*=$Qstar")
    l_m_list = NTuple{2, Rational{Int64}}[(L, M)
        for L in qabs:1:(qabs + nmax)
        for M in -L:1:L][1:N]
    Ψcurrent = Ψproj(Qstar, 2p, N, l_m_list)
    Ψnext    = Ψproj(Qstar, 2p, N, l_m_list)
    logpdf(ψ::Ψproj) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)
end

# ---------------------------------------------------------------- positions

θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, N)
θnext = copy(θcurrent)
ϕnext = copy(ϕcurrent)
σ = pi / sqrt(12.0)

# ---------------------------------------------------------------- thermalization

println("thermalizing ($ntherm steps) ...")
sampling_iter, σ, δt_therm, acc_therm = gibbs_thermalization!(
    RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, ntherm)
println("  acceptance rate: $(round(acc_therm, digits=3))  step size: $(round(σ, digits=4))  time: $(round(δt_therm, digits=1))s")

# ---------------------------------------------------------------- production run

acc = HarmonicAccumulator(density_lmax)

logpdf_current = logpdf(Ψcurrent)
num_accepted   = 0
t0 = time()

for mc_iter in 1:nsteps
    θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
    update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)
    logpdf_next = logpdf(Ψnext)

    if logpdf_next - logpdf_current >= log(rand(RNG))
        θcurrent[sampling_iter] = θnext[sampling_iter]
        ϕcurrent[sampling_iter] = ϕnext[sampling_iter]
        copy!(Ψcurrent, Ψnext, sampling_iter)
        logpdf_current = logpdf_next
        num_accepted  += 1
    else
        θnext[sampling_iter] = θcurrent[sampling_iter]
        ϕnext[sampling_iter] = ϕcurrent[sampling_iter]
        copy!(Ψnext, Ψcurrent, sampling_iter)
    end

    # accumulate n_lm on every step using the current (accepted) configuration
    accumulate_density!(acc, θcurrent, ϕcurrent)

    sampling_iter = mod(sampling_iter, N) + 1

    if mc_iter == nsteps || mod(mc_iter, 500_000) == 0
        println("  step $mc_iter / $nsteps  acceptance=$(round(num_accepted/mc_iter, digits=3))")
    end
end

δt_prod = time() - t0
println("production done in $(round(δt_prod, digits=1))s  acceptance=$(round(num_accepted/nsteps, digits=3))")

# ---------------------------------------------------------------- save density

ls, ms, n_lm = finalize_n_lm(acc)
JLD2.jldsave(outfile;
    l  = ls,
    m  = ms,
    n_lm = n_lm,
    N  = N,
    metadata = "mc_chain_worker chain=$chain nsteps=$nsteps twoQ=$twoQ p=$p nmax=$nmax orbital_file=$orbital_file",
    acceptance_rate  = num_accepted / nsteps,
    thermalization_acceptance = acc_therm,
    production_time  = δt_prod,
    therm_time       = δt_therm,
)
println("wrote density to $outfile  (lmax=$density_lmax, n_00=$(round(n_lm[1], digits=6)))")
