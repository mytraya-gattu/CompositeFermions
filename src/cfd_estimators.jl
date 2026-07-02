module CFDEstimators

# CF diagonalization (CFD) estimators over K determinant configurations that
# share one Monte Carlo walker.
#
# The guide wavefunction is a ΨprojDFT whose occupied orbitals are the guide
# configuration's columns of the full KS coefficient matrix C_all
# (dim_basis × norb, norb ≥ N). ΨprojDFT maintains raw_slater_det
# (dim_basis × N): the JK-projected value of every basis orbital at every
# particle, which depends only on positions. For configuration c (a sorted
# N-subset of 1:norb) the projected many-body amplitude is
#
#   Ψ_c ∝ J^{2p} det( (C_all' * raw_slater_det)[c, :] )
#
# so the Jastrow cancels in every ratio r_c = Ψ_c/Ψ_guide and the overlap and
# potential matrices are accumulated from outer products of the ratio vector:
#
#   S_cc' = E[ conj(r_c) r_c' ],   W_cc' = E[ conj(r_c) r_c' V_loc ]
#
# sampled against |Ψ_guide|². Kinetic energy is exactly quenched in the LLL
# (constant), so V_loc = chord Coulomb + external (disorder) potential only.

export CFDConfigs, CFDAccumulator, accumulate_cfd!,
    coulomb_local_energy, HarmonicFieldEvaluator, field_local_energy

using LinearAlgebra

using ..DFTWavefunction: ΨprojDFT

# -------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------

struct CFDConfigs
    norb::Int
    configs::Vector{Vector{Int}}   # K sorted N-subsets of 1:norb
    guide::Int                     # index into configs of the sampled guide

    function CFDConfigs(norb::Int, configs::Vector{Vector{Int}}, guide::Int)
        isempty(configs) && throw(ArgumentError("need at least one configuration"))
        n = length(first(configs))
        for c in configs
            length(c) == n || throw(ArgumentError("all configurations must have the same particle number"))
            issorted(c) || throw(ArgumentError("configurations must be sorted"))
            (first(c) >= 1 && last(c) <= norb) || throw(ArgumentError("orbital index outside 1:norb"))
            allunique(c) || throw(ArgumentError("repeated orbital in configuration"))
        end
        (1 <= guide <= length(configs)) || throw(ArgumentError("guide index out of range"))
        return new(norb, configs, guide)
    end
end

# -------------------------------------------------------------------
# Accumulator
# -------------------------------------------------------------------

mutable struct CFDAccumulator
    cfg::CFDConfigs
    S::Matrix{ComplexF64}          # Σ conj(r_i) r_j
    W::Matrix{ComplexF64}          # Σ conj(r_i) r_j V_loc
    sum_r2::Vector{Float64}        # Σ |r_k|²   (for effective sample size)
    sum_r4::Vector{Float64}        # Σ |r_k|⁴
    n_samples::Int

    Φ::Matrix{ComplexF64}          # workspace: norb × N projected orbitals
    sub::Matrix{ComplexF64}        # workspace: N × N row selection
    logr::Vector{ComplexF64}
    r::Vector{ComplexF64}
end

function CFDAccumulator(cfg::CFDConfigs, dim_basis::Int, system_size::Int)
    K = length(cfg.configs)
    return CFDAccumulator(cfg,
        zeros(ComplexF64, K, K), zeros(ComplexF64, K, K),
        zeros(Float64, K), zeros(Float64, K), 0,
        zeros(ComplexF64, cfg.norb, system_size),
        zeros(ComplexF64, system_size, system_size),
        zeros(ComplexF64, K), zeros(ComplexF64, K))
end

# log det of Φ[rows, :] via in-place LU (complex log; 2πi branch is irrelevant
# because only exponentiated differences are used).
function _logdet_rows!(work::Matrix{ComplexF64}, Φ::Matrix{ComplexF64}, rows::Vector{Int})
    n = length(rows)
    @inbounds for j in 1:n, i in 1:n
        work[i, j] = Φ[rows[i], j]
    end
    F = lu!(work; check = false)
    z = zero(ComplexF64)
    flips = 0
    @inbounds for i in 1:n
        z += log(work[i, i])
        flips += (F.ipiv[i] != i)
    end
    return isodd(flips) ? z + im * pi : z
end

"""
    accumulate_cfd!(acc, Ψ, C_all, V_loc)

Accumulate one correlated sample of the overlap and potential matrices from
the current state of the guide wavefunction `Ψ` (whose `raw_slater_det` must
be up to date for the accepted particle positions). `C_all` is the full
(dim_basis × norb) KS coefficient matrix; `V_loc` the local potential energy
of the current configuration of particle positions.
"""
function accumulate_cfd!(acc::CFDAccumulator, Ψ::ΨprojDFT, C_all::Matrix{ComplexF64}, V_loc::Float64)
    size(C_all) == (Ψ.dim_basis, acc.cfg.norb) ||
        throw(ArgumentError("C_all must be ($(Ψ.dim_basis) × $(acc.cfg.norb)), got $(size(C_all))"))

    mul!(acc.Φ, C_all', Ψ.raw_slater_det)

    configs = acc.cfg.configs
    K = length(configs)
    @inbounds for k in 1:K
        acc.logr[k] = _logdet_rows!(acc.sub, acc.Φ, configs[k])
    end
    zg = acc.logr[acc.cfg.guide]
    @inbounds for k in 1:K
        acc.r[k] = exp(acc.logr[k] - zg)
        rk2 = abs2(acc.r[k])
        acc.sum_r2[k] += rk2
        acc.sum_r4[k] += rk2 * rk2
    end

    @inbounds for j in 1:K
        rj = acc.r[j]
        for i in 1:K
            z = conj(acc.r[i]) * rj
            acc.S[i, j] += z
            acc.W[i, j] += V_loc * z
        end
    end
    acc.n_samples += 1
    return acc
end

# -------------------------------------------------------------------
# Local potential energy
# -------------------------------------------------------------------

"""
    coulomb_local_energy(Ψ, twoQ) -> Float64

Chord-distance Coulomb energy Σ_{i<j} 1/(R·d_ij) in units of e²/εl_B, using
the guide's maintained `dist_matrix` (entries 2 sin(γ_ij/2), i.e. chord on the
unit sphere) and the physical sphere radius R = √Q l_B set by the *electron*
monopole `twoQ` (passed explicitly; do not confuse with 2Q*). The
uniform-background constant −N²/(2√Q) is NOT included here (it is a constant
shift, applied at the reduction stage).
"""
function coulomb_local_energy(Ψ::ΨprojDFT, twoQ::Int)
    N = Ψ.system_size
    R = sqrt(twoQ / 2)
    s = 0.0
    @inbounds for i in 1:(N - 1), j in (i + 1):N
        s += 1.0 / Ψ.dist_matrix[j - 1, i]
    end
    return s / R
end

# -------------------------------------------------------------------
# External (disorder) potential: evaluate a spherical-harmonic field
# V(Ω) = Σ_{λμ} V_λμ Y_λμ(Ω) at particle positions. Coefficients must satisfy
# the reality condition V_{λ,-μ} = (-1)^μ conj(V_{λμ}).
# -------------------------------------------------------------------

struct HarmonicFieldEvaluator
    l::Vector{Int}
    m::Vector{Int}
    coeffs::Vector{ComplexF64}
    lmax::Int
    plm::Matrix{Float64}           # workspace: (lmax+1) × (lmax+1), [l+1, m+1]
end

function HarmonicFieldEvaluator(l::Vector{Int}, m::Vector{Int}, coeffs::Vector{ComplexF64})
    length(l) == length(m) == length(coeffs) ||
        throw(ArgumentError("l, m, coeffs must have equal length"))
    lmax = isempty(l) ? 0 : maximum(l)
    return HarmonicFieldEvaluator(l, m, coeffs, lmax, zeros(lmax + 1, lmax + 1))
end

# Fully normalized associated Legendre functions with the Condon-Shortley
# phase folded in: Y_lm(θ,ϕ) = plm[l+1, m+1] e^{imϕ} for m ≥ 0.
function _normalized_plm!(P::Matrix{Float64}, lmax::Int, x::Float64, sx::Float64)
    P[1, 1] = 1.0 / sqrt(4.0 * pi)
    @inbounds for m in 1:lmax
        P[m + 1, m + 1] = -sqrt((2m + 1) / (2m)) * sx * P[m, m]
    end
    @inbounds for m in 0:(lmax - 1)
        P[m + 2, m + 1] = sqrt(2m + 3.0) * x * P[m + 1, m + 1]
    end
    @inbounds for m in 0:lmax, l in (m + 2):lmax
        a = sqrt((4.0 * l^2 - 1.0) / ((l - m) * (l + m)))
        b = sqrt(((2l + 1.0) * (l - 1 - m) * (l - 1 + m)) / ((2l - 3.0) * (l - m) * (l + m)))
        P[l + 1, m + 1] = a * x * P[l, m + 1] - b * P[l - 1, m + 1]
    end
    return
end

"""
    field_local_energy(fe, θ, ϕ) -> Float64

Σ_i V(θ_i, ϕ_i) for a real spherical-harmonic field.
"""
function field_local_energy(fe::HarmonicFieldEvaluator, θ::Vector{Float64}, ϕ::Vector{Float64})
    isempty(fe.l) && return 0.0
    s = 0.0
    @inbounds for i in eachindex(θ)
        x = cos(θ[i])
        sx = sin(θ[i])
        _normalized_plm!(fe.plm, fe.lmax, x, sx)
        v = 0.0 + 0.0im
        for k in eachindex(fe.l)
            l, m = fe.l[k], fe.m[k]
            if m >= 0
                Y = fe.plm[l + 1, m + 1] * cis(m * ϕ[i])
            else
                Y = (isodd(-m) ? -1.0 : 1.0) * fe.plm[l + 1, -m + 1] * cis(m * ϕ[i])
            end
            v += fe.coeffs[k] * Y
        end
        s += real(v)
    end
    return s
end

end # module CFDEstimators
