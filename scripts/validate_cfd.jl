#!/usr/bin/env julia
# Machinery validation for the CFD estimators (no physics, no MC):
#
#  1. Configuration ratios r_c from CFDAccumulator (shared raw_slater_det)
#     must equal ratios of independently constructed ΨprojDFT amplitudes
#     evaluated at the same particle positions.
#  2. The guide row-selection determinant must reproduce the guide's own
#     slater_det determinant.
#  3. HarmonicFieldEvaluator must match explicit closed-form Y_lm at low l
#     and produce a real field for reality-condition coefficients.
#  4. coulomb_local_energy must match a from-scratch great-circle computation.
#
# Run: julia --project=<CF_PROJECT> scripts/validate_cfd.jl

using LinearAlgebra
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CFsOnSphere
using Random

const RNG = Random.MersenneTwister(20260702)

failures = 0
function check(name, ok)
    global failures
    println(rpad(name, 64), ok ? "PASS" : "FAIL")
    ok || (failures += 1)
    return
end

# ---------------------------------------------------------------- setup

N, twoQ, p, nmax = 5, 12, 1, 2
Qstar = twoQ // 2 - p * (N - 1)
qabs = abs(Qstar)
dim_basis = sum(numerator(2 * (qabs + λ)) + 1 for λ in 0:nmax)
norb = min(dim_basis, N + 5)

# random orthonormal "KS orbitals"
A = randn(RNG, ComplexF64, dim_basis, norb)
C_all = Matrix(qr(A).Q)[:, 1:norb]

configs = Vector{Int}[collect(1:N)]                       # guide: Aufbau
push!(configs, sort([1:(N - 1); N + 1]))                  # single excitation
push!(configs, sort([1:(N - 2); N + 1; N + 2]))           # double excitation
push!(configs, sort([1:(N - 3); N; N + 3; N + 4]))        # another double
push!(configs, sort([2:N; N + 5]))                        # hole at bottom
cfg = CFDConfigs(norb, configs, 1)
K = length(configs)

θ, ϕ = rand_θ_ϕ_gen(RNG, N)

# guide wavefunction at these positions
Ψg = ΨprojDFT(Qstar, 2p, N, nmax, Matrix{ComplexF64}(C_all[:, configs[1]]))
update_wavefunction!(Ψg, θ, ϕ)

acc = CFDAccumulator(cfg, dim_basis, N)
accumulate_cfd!(acc, Ψg, Matrix{ComplexF64}(C_all), 0.0)

# ---------------------------------------------------------------- 1 & 2: ratios

# brute force: independent ΨprojDFT per configuration
log_amp = zeros(ComplexF64, K)
for k in 1:K
    Ψk = ΨprojDFT(Qstar, 2p, N, nmax, Matrix{ComplexF64}(C_all[:, configs[k]]))
    update_wavefunction!(Ψk, θ, ϕ)
    log_amp[k] = logdet(Ψk.slater_det) + Ψk.jastrow_factor_log
end

max_ratio_err = maximum(abs(acc.r[k] - exp(log_amp[k] - log_amp[1])) / abs(acc.r[k]) for k in 2:K)
check("config ratios vs independent ΨprojDFT (rel err < 1e-10)", max_ratio_err < 1e-10)
println("    max relative error: $max_ratio_err")

Φ = C_all' * Ψg.raw_slater_det
guide_err = abs(det(Φ[configs[1], :]) - det(Ψg.slater_det)) / abs(det(Ψg.slater_det))
check("guide row-selection det == guide slater det (rel err < 1e-10)", guide_err < 1e-10)

check("guide ratio r_g == 1", abs(acc.r[1] - 1) < 1e-12)
check("S accumulates r† r outer product", acc.S ≈ conj(acc.r) * transpose(acc.r))

# ---------------------------------------------------------------- 3: field evaluator

Y00(θ, ϕ) = 1 / sqrt(4pi)
Y10(θ, ϕ) = sqrt(3 / (4pi)) * cos(θ)
Y11(θ, ϕ) = -sqrt(3 / (8pi)) * sin(θ) * cis(ϕ)
Y21(θ, ϕ) = -sqrt(15 / (8pi)) * sin(θ) * cos(θ) * cis(ϕ)
Y22(θ, ϕ) = sqrt(15 / (32pi)) * sin(θ)^2 * cis(2ϕ)

θt, ϕt = 1.1234, 2.345
for (name, l, m, Yref) in [("Y00", 0, 0, Y00), ("Y10", 1, 0, Y10), ("Y11", 1, 1, Y11),
                           ("Y21", 2, 1, Y21), ("Y22", 2, 2, Y22)]
    fe1 = HarmonicFieldEvaluator([l], [m], [1.0 + 0.0im])
    # a single complex mode: evaluator returns Re(Y_lm); compare
    got = field_local_energy(fe1, [θt], [ϕt])
    check("evaluator $name (real part)", abs(got - real(Yref(θt, ϕt))) < 1e-12)
end

# reality condition => real field
ls = Int[]; ms = Int[]; cs = ComplexF64[]
for l in 0:4, m in -l:l
    m < 0 && continue
    c = m == 0 ? ComplexF64(randn(RNG)) : ComplexF64(randn(RNG), randn(RNG))
    push!(ls, l); push!(ms, m); push!(cs, c)
    if m > 0
        push!(ls, l); push!(ms, -m); push!(cs, (-1)^m * conj(c))
    end
end
# For a reality-condition coefficient set, the full ±m sum must equal the
# m ≥ 0 modes counted with weight (m == 0 ? 1 : 2) — this exercises the
# evaluator's m < 0 branch (Y_{l,-m} = (-1)^m conj(Y_lm)).
fe_full = HarmonicFieldEvaluator(ls, ms, cs)
reality_ok = true
for _ in 1:50
    θr = acos(2rand(RNG) - 1); ϕr = 2pi * rand(RNG)
    v_full = field_local_energy(fe_full, [θr], [ϕr])
    v_half = 0.0
    for k in eachindex(ls)
        ms[k] < 0 && continue
        fe_k = HarmonicFieldEvaluator([ls[k]], [ms[k]], [cs[k]])
        contrib = field_local_energy(fe_k, [θr], [ϕr])
        v_half += ms[k] == 0 ? contrib : 2 * contrib
    end
    global reality_ok &= abs(v_full - v_half) < 1e-10
end
check("reality-condition field: ±m sum == weighted m ≥ 0 sum", reality_ok)

# ---------------------------------------------------------------- 4: Coulomb

E_direct = 0.0
for i in 1:(N - 1), j in (i + 1):N
    cosγ = cos(θ[i]) * cos(θ[j]) + sin(θ[i]) * sin(θ[j]) * cos(ϕ[i] - ϕ[j])
    chord = 2 * sin(acos(clamp(cosγ, -1, 1)) / 2)
    global E_direct += 1 / (sqrt(twoQ / 2) * chord)
end
E_est = coulomb_local_energy(Ψg, twoQ)
check("coulomb_local_energy vs great-circle recomputation", abs(E_est - E_direct) / E_direct < 1e-10)
println("    E_coulomb = $E_est (e²/εl_B)")

# ----------------------------------------------------------------

println()
if failures == 0
    println("ALL CFD MACHINERY CHECKS PASSED")
else
    println("$failures CHECK(S) FAILED")
    exit(1)
end
