# MCMC convergence diagnostics implemented by hand (no external dependencies):
# the Gelman-Rubin split-R̂ across chains and the effective sample size (ESS) via
# autocovariance with Geyer's initial-positive-sequence truncation. Both operate on
# scalar traces (e.g. the recorded `logpdf_current`) produced by the chain driver.

"""
    split_rhat(traces::Vector{Vector{Float64}}) -> Float64

Gelman-Rubin **split-R̂** across MCMC chains. Each of the `M` input chains is split in half
(giving `2M` sub-chains of length `n = ⌊L/2⌋`, `L` the common trimmed length), and R̂ is
computed from the between- and within-sub-chain variances:

    θ̄ₘ = (1/n) Σₜ θₘₜ                 (sub-chain mean)
    θ̄  = (1/2M) Σₘ θ̄ₘ                 (grand mean)
    B  = n/(2M-1) Σₘ (θ̄ₘ - θ̄)²        (between-sub-chain variance)
    sₘ² = 1/(n-1) Σₜ (θₘₜ - θ̄ₘ)²
    W  = (1/2M) Σₘ sₘ²                 (within-sub-chain variance)
    var⁺ = (n-1)/n · W + B/n           (marginal posterior variance estimate)
    R̂ = √(var⁺ / W)

R̂ → 1 as the chains mix. Values ≳ 1.01–1.1 flag non-convergence.

Degenerate inputs return `NaN` rather than throwing:
- fewer than two chains (cross-chain R̂ is undefined with a single chain),
- traces shorter than 4 samples (each half would have < 2 points),
- `W == 0` (all sub-chains constant — e.g. an identical constant trace).
All input chains are trimmed to their common minimum length before splitting.
"""
function split_rhat(traces::Vector{Vector{Float64}})

    M::Int = length(traces)
    M < 2 && return NaN

    L::Int = minimum(length, traces)
    L < 4 && return NaN

    n::Int = L ÷ 2                       # length of each half

    # Build 2M split sub-chains as views into the trimmed traces.
    means = Float64[]
    within_vars = Float64[]
    for trace in traces
        for half in (1:n, (n+1):(2n))
            seg = @view trace[half]
            m = sum(seg) / n
            s2 = 0.0
            @inbounds for x in seg
                s2 += (x - m)^2
            end
            s2 /= (n - 1)
            push!(means, m)
            push!(within_vars, s2)
        end
    end

    M2 = length(means)                   # == 2M
    grand = sum(means) / M2

    B = 0.0
    @inbounds for m in means
        B += (m - grand)^2
    end
    B *= n / (M2 - 1)

    W = sum(within_vars) / M2
    W == 0.0 && return NaN

    var_plus = (n - 1) / n * W + B / n
    return sqrt(var_plus / W)
end

"""
    _autocov(x, μ, t) -> Float64

Biased lag-`t` autocovariance `(1/n) Σᵢ (xᵢ-μ)(xᵢ₊ₜ-μ)` of the length-`n` vector `x` with
mean `μ`. Dividing by `n` (not `n-t`) is the standard choice for ESS estimation: it damps the
high-lag tail and keeps the summed autocorrelation well behaved.
"""
@inline function _autocov(x::AbstractVector{Float64}, μ::Float64, t::Int)
    n = length(x)
    s = 0.0
    @inbounds for i in 1:(n - t)
        s += (x[i] - μ) * (x[i + t] - μ)
    end
    return s / n
end

"""
    effective_sample_size(trace::Vector{Float64}) -> Float64

Effective sample size of a single scalar trace via the integrated autocorrelation time,
truncated with **Geyer's initial positive sequence**. With normalized autocorrelations
`ρₜ = γ(t)/γ(0)`, form the consecutive pairs `Γₘ = ρ₂ₘ + ρ₂ₘ₊₁` and sum them while the pair is
positive (Geyer 1992). The integrated autocorrelation time and ESS are

    τ = -1 + 2 Σₘ Γₘ  (sum truncated at the first m ≥ 1 with Γₘ ≤ 0),   ESS = n / τ.

`τ` is floored at 1 (an antithetic trace cannot yield ESS above the raw sample count here).

Degenerate inputs return without throwing:
- `n < 2` returns `float(n)`,
- a constant trace (`γ(0) == 0`) returns `NaN` (autocorrelation is undefined).
"""
function effective_sample_size(trace::Vector{Float64})

    n::Int = length(trace)
    n < 2 && return float(n)

    μ = sum(trace) / n
    γ0 = _autocov(trace, μ, 0)
    γ0 == 0.0 && return NaN

    ρ(t) = _autocov(trace, μ, t) / γ0

    # Γ₀ = ρ₀ + ρ₁ = 1 + ρ₁ is always included.
    Σ = 1.0 + (n > 1 ? ρ(1) : 0.0)
    m = 1
    while 2m <= n - 1
        Γ = ρ(2m) + (2m + 1 <= n - 1 ? ρ(2m + 1) : 0.0)
        Γ <= 0.0 && break
        Σ += Γ
        m += 1
    end

    τ = 2Σ - 1.0
    τ < 1.0 && (τ = 1.0)
    return n / τ
end

"""
    effective_sample_size(traces::Vector{Vector{Float64}}) -> Float64

Multi-chain (pooled) effective sample size combining `M` chains of common trimmed length `n`
the standard way (Vehtari et al. 2021 / Stan). Let `sₘ²` and `γₘ(t)` be the per-chain variance
and autocovariance, `W = (1/M) Σ sₘ²`, and `B = n/(M-1) Σ (θ̄ₘ - θ̄)²`. With
`var⁺ = (n-1)/n · W + B/n`, the combined autocorrelation is

    ρₜ = 1 - (W - (1/M) Σₘ γₘ(t)) / var⁺,

to which the same Geyer initial-positive-sequence truncation is applied. Then
`ESS = M·n / τ` with `τ = -1 + 2 Σ Γₘ`.

Falls back to the single-chain estimator when one chain is supplied. Returns `NaN` when
`var⁺ == 0` (all chains constant and identical) and `float(total)` for a too-short trace.
"""
function effective_sample_size(traces::Vector{Vector{Float64}})

    M::Int = length(traces)
    M == 0 && return NaN
    M == 1 && return effective_sample_size(traces[1])

    n::Int = minimum(length, traces)
    n < 2 && return float(M * n)

    segs = [@view trace[1:n] for trace in traces]
    means = [sum(seg) / n for seg in segs]
    grand = sum(means) / M

    # Per-chain variance sₘ² (unbiased) and within-chain variance W.
    s2 = Float64[]
    for (seg, mm) in zip(segs, means)
        v = 0.0
        @inbounds for x in seg
            v += (x - mm)^2
        end
        push!(s2, v / (n - 1))
    end
    W = sum(s2) / M

    B = 0.0
    @inbounds for mm in means
        B += (mm - grand)^2
    end
    B *= n / (M - 1)

    var_plus = (n - 1) / n * W + B / n
    var_plus == 0.0 && return NaN

    # Mean over chains of the lag-t autocovariance, then the combined autocorrelation.
    mean_autocov(t) = sum(_autocov(segs[c], means[c], t) for c in 1:M) / M
    ρ(t) = 1.0 - (W - mean_autocov(t)) / var_plus

    Σ = 1.0 + ρ(1)
    m = 1
    while 2m <= n - 1
        Γ = ρ(2m) + (2m + 1 <= n - 1 ? ρ(2m + 1) : 0.0)
        Γ <= 0.0 && break
        Σ += Γ
        m += 1
    end

    τ = 2Σ - 1.0
    τ < 1.0 && (τ = 1.0)
    return (M * n) / τ
end
