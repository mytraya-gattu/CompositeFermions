# Ground-state Coulomb-energy validation for the multi-chain driver (`run_chains`).
#
# Provides `energy_setup` — a `run_chains` setup closure measuring the configuration Coulomb
# sum V = Σ_{i<j} 1/r_ij (chord distances on the UNIT sphere, from `ψ.dist_matrix`) — and
# `run_energy_system`, which runs the chains for one (N, n, p) ground state and reduces ⟨V⟩
# to the per-particle background-subtracted energy on the sphere of radius R = √Q l_B:
#
#     E/N = ( ⟨V⟩ − N²/2 ) / (R N),        Q = Qstar + p(N−1)/2,   R = √Q.
#
# Two density-corrected variants (× √(2Qν/N) and × √(N/(2Qν)), ν = n/(pn+1)) are reported
# alongside the raw value; which convention matches ref/cf_ground_state_energies.csv is
# established by the local benchmark BEFORE any cluster submission — the campaign then quotes
# that convention only.

using CFsOnSphere
using Random
using Statistics
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)

"Physical monopole strength Q = Qstar + p(N-1)/2 for the Jastrow factor J^p."
physical_Q(Qstar::Rational{Int64}, p::Int64, N::Int64) = Qstar + p * (N - 1) // 2

"""
    energy_setup(Qstar, l_m_list, p) -> setup

`run_chains` setup closure for the Ψproj ground state accumulating the per-step Coulomb sum
`V mean` = ⟨Σ_{i<j} 1/r_ij⟩ (unit sphere) and `V2 mean` = ⟨V²⟩ (both scalars, so
`combine_chains` pools them step-weighted and attaches cross-chain standard errors).
"""
function energy_setup(Qstar::Rational{Int64}, l_m_list::Vector{NTuple{2,Rational{Int64}}}, p::Int64)

    N = length(l_m_list)

    return function (chain_id::Int, RNG)

        Ψcurrent = Ψproj(Qstar, p, N, l_m_list)
        Ψnext    = Ψproj(Qstar, p, N, l_m_list)
        θ, ϕ = rand_θ_ϕ_gen(RNG, N)

        logpdf(ψ::Ψproj) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

        # Full O(N²) recomputation of V every step: ~N²/2 divisions is negligible next to the
        # O(N³) logdet in `logpdf`, and it avoids any incremental floating-point drift.
        coulomb(ψ) = sum(1.0 / ψ.dist_matrix[k, i] for i in 1:N-1 for k in i:N-1)

        Vsum  = Ref(0.0)
        V2sum = Ref(0.0)

        measure! = function (Ψ, θc, ϕc, accepted::Bool, sampling_iter)
            V = coulomb(Ψ)
            Vsum[]  += V
            V2sum[] += V * V
            return
        end

        observables = function (n::Int)
            return Dict{String,Any}(
                "observables" => Dict{String,Any}(
                    "V mean"  => Vsum[] / n,
                    "V2 mean" => V2sum[] / n,
                ),
                "grids" => Dict{String,Any}(),
            )
        end

        return (Ψcurrent = Ψcurrent, Ψnext = Ψnext, θ = θ, ϕ = ϕ,
                logpdf = logpdf, measure! = measure!, observables = observables)
    end
end

"""
    reduce_energy(combined, N, n, p, Qstar) -> Dict

Reduce the pooled `combine_chains` output to per-particle energies in units of e²/(ε l_B):
`E_raw` (background-subtracted, no density correction) and the two candidate density-corrected
values, each with the cross-chain standard error propagated linearly.
"""
function reduce_energy(combined::Dict, N::Int64, n::Int64, p::Int64, Qstar::Rational{Int64})

    Q = physical_Q(Qstar, p, N)
    R = sqrt(Float64(Q))
    ν = n // (p * n + 1)

    Vmean = combined["observables"]["V mean"]
    σV    = get(combined["standard_error"], "V mean", NaN)

    E_raw  = (Vmean - N^2 / 2.0) / (R * N)
    σE     = σV / (R * N)

    c1 = sqrt(2.0 * Float64(Q) * Float64(ν) / N)   # ρ_∞/ρ_N direction 1
    c2 = 1.0 / c1                                  # and its inverse

    return Dict{String,Any}(
        "N" => N, "n" => n, "p" => p,
        "Qstar" => Qstar, "Q" => Q, "nu" => ν,
        "V mean" => Vmean, "V stderr" => σV,
        "E_raw" => E_raw, "E_raw stderr" => σE,
        "E_corr1" => E_raw * c1, "E_corr1 stderr" => σE * c1,
        "E_corr2" => E_raw * c2, "E_corr2 stderr" => σE * c2,
        "corr1 factor" => c1,
        "n_chains" => combined["n_chains"],
        "total_steps" => combined["total_steps"],
        "rhat" => combined["diagnostics"].rhat,
        "ess"  => combined["diagnostics"].ess,
    )
end

"""
    run_energy_system(dir, N, n, p; n_chains, kwargs...) -> Dict

Run the multi-chain energy measurement for one (N, n, p) ground state: chains checkpoint into
`dir/`, the reduced summary is saved to `dir/energy_summary.jld2` and returned. `kwargs` pass
through to `run_chains` (num_thermalization, num_steps, resume, ...).
"""
function run_energy_system(dir::String, N::Int64, n::Int64, p::Int64; n_chains::Int, kwargs...)

    Qstar, l_m_list = cf_ground_state_lm(N, n, p)

    run_chains(energy_setup(Qstar, l_m_list, p); n_chains = n_chains, dir = dir, kwargs...)

    combined = combine_chains(dir)
    summary = reduce_energy(combined, N, n, p, Qstar)

    save(joinpath(dir, "energy_summary.jld2"), summary)

    return summary
end
