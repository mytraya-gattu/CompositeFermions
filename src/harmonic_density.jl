module HarmonicDensity

export HarmonicAccumulator, accumulate_density!, finalize_n_lm

using SpecialFunctions: loggamma

# -----------------------------------------------------------------------
# Accumulates n_lm = (1/nsteps) * sum_{steps,particles} conj(Y_lm(θ,φ))
# using the Condon-Shortley convention for Y_lm.
# Only m >= 0 modes are stored online; m < 0 is filled by the reality
# condition n_{l,-m} = (-1)^m * conj(n_{l,m}) in finalize_n_lm.
# -----------------------------------------------------------------------

struct HarmonicAccumulator
    lmax::Int
    # n_acc[l*(l+1)/2 + m + 1] for l = 0..lmax, m = 0..l   (m >= 0 only)
    n_acc::Vector{ComplexF64}
    # precomputed normalization factors N_lm for m = 0..l
    norms::Vector{Float64}
    nsteps::Base.RefValue{Int64}
    # scratch buffer for _assoc_legendre!, reused every accumulation step
    scratch_P::Vector{Float64}
end

_flat_idx(l, m) = l * (l + 1) ÷ 2 + m + 1

function HarmonicAccumulator(lmax::Int)
    ntot = (lmax + 1) * (lmax + 2) ÷ 2   # sum_{l=0}^{lmax} (l+1)
    n_acc = zeros(ComplexF64, ntot)
    norms = zeros(Float64, ntot)
    for l in 0:lmax, m in 0:l
        # N_lm = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)  (no Condon-Shortley sign)
        # We absorb the (-1)^m phase into the m > 0 output in finalize_n_lm.
        log_norm = 0.5 * (log(2l + 1) - log(4 * π) + loggamma(l - m + 1) - loggamma(l + m + 1))
        norms[_flat_idx(l, m)] = exp(log_norm)
    end
    scratch_P = zeros(Float64, ntot)
    return HarmonicAccumulator(lmax, n_acc, norms, Ref(Int64(0)), scratch_P)
end

# Accumulate the current configuration θ/φ into the density.
# Call this once per MC step (whether accepted or rejected) with the
# current θ and φ vectors.
function accumulate_density!(acc::HarmonicAccumulator, θvec, φvec)
    lmax = acc.lmax
    P = acc.scratch_P  # reuse pre-allocated buffer; P[_flat_idx(l,m)]
    @inbounds for k in eachindex(θvec)
        x  = cos(θvec[k])
        sx = sqrt(max(0.0, 1.0 - x * x))
        ϕ  = φvec[k]

        # ---- Associated Legendre P_l^m(x) for l = 0..lmax, m = 0..l ----
        # Using the standard three-term recurrence without normalization.
        # P_0^0 = 1, P_1^1 = -sx, then sectoral P_l^l and P_{l+1}^l.
        _assoc_legendre!(P, lmax, x, sx)

        # ---- e^{-im phi} for m = 0..lmax ----
        eimφ = cis(-ϕ)
        eim = one(ComplexF64)         # e^{-im phi} at m = 0

        for m in 0:lmax
            # eim = e^{-im phi}
            for l in m:lmax
                # conj(Y_lm) = N_lm * P_l^m(x) * e^{-im phi}  (m >= 0)
                acc.n_acc[_flat_idx(l, m)] += acc.norms[_flat_idx(l, m)] * P[_flat_idx(l, m)] * eim
            end
            eim *= eimφ
        end
    end
    acc.nsteps[] += 1
    return
end

function _assoc_legendre!(P, lmax, x, sx)
    P[1] = 1.0                                   # P_0^0
    if lmax == 0; return; end

    P[_flat_idx(1, 0)] = x                       # P_1^0
    P[_flat_idx(1, 1)] = -sx                     # P_1^1

    for l in 2:lmax
        # Sectoral: P_l^l = -(2l-1) sx P_{l-1}^{l-1}
        P[_flat_idx(l, l)] = -(2l - 1) * sx * P[_flat_idx(l - 1, l - 1)]
        # Subsectoral: P_l^{l-1} = (2l-1) x P_{l-1}^{l-1}
        P[_flat_idx(l, l - 1)] = (2l - 1) * x * P[_flat_idx(l - 1, l - 1)]
        # Tesseral: P_l^m for m < l-1
        for m in 0:(l - 2)
            P[_flat_idx(l, m)] = ((2l - 1) * x * P[_flat_idx(l - 1, m)] - (l + m - 1) * P[_flat_idx(l - 2, m)]) / (l - m)
        end
    end
    return
end

# Return (l_vec, m_vec, n_lm_vec) suitable for CF-DFT's load_density_lm.
# Applies the Condon-Shortley (-1)^m sign for m > 0 and fills m < 0 by
# the reality condition n_{l,-m} = (-1)^m conj(n_{l,m}).
function finalize_n_lm(acc::HarmonicAccumulator)
    acc.nsteps[] > 0 || error("no samples accumulated")
    lmax = acc.lmax
    norm = Float64(acc.nsteps[])

    ls     = Int[]
    ms     = Int[]
    n_lm   = ComplexF64[]

    for l in 0:lmax
        for m in -l:l
            mabs = abs(m)
            # raw = N_lm * (1/nsteps) Σ_{step,i} P_l^mabs(cos θ_i) e^{-i mabs φ_i}
            # (P here is the standard associated Legendre, no CS sign)
            raw  = acc.n_acc[_flat_idx(l, mabs)] / norm
            # CS convention: Y_{lm} = (-1)^m N_lm P_l^m e^{imφ} (m >= 0)
            #                Y_{l,-m} = N_lm P_l^m e^{-imφ}       (m > 0)
            # so conj(Y_{lm}) = (-1)^m * raw  for m >= 0
            #    conj(Y_{l,-m}) = conj(raw)    for m > 0
            val = m >= 0 ? (-1)^m * raw : conj(raw)
            push!(ls, l)
            push!(ms, m)
            push!(n_lm, val)
        end
    end
    return ls, ms, n_lm
end

end # module HarmonicDensity
