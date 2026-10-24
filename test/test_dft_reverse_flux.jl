# ΨprojDFT with identity coefficients (filled Lambda levels) must reproduce Ψproj exactly,
# for forward and reverse flux, after full updates and after single-particle moves.
#   julia --project=. test/test_dft_reverse_flux.jl
using CFsOnSphere, LinearAlgebra, Random, Printf
fails = 0
function compare(label, Qstar, P, N, nfill)
    qa = abs(Qstar)
    lm = NTuple{2,Rational{Int64}}[(qa + λ, m) for λ in 0:(nfill - 1) for m in -(qa + λ):(qa + λ)]
    length(lm) == N || error("filled levels hold $(length(lm)) states, N = $N")
    ref = Ψproj(Qstar, P, N, lm)
    dft = ΨprojDFT(Qstar, P, N, nfill - 1, Matrix{ComplexF64}(I, N, N))
    rng = MersenneTwister(3)
    θ, ϕ = rand_θ_ϕ_gen(rng, N)
    update_wavefunction!(ref, θ, ϕ); update_wavefunction!(dft, θ, ϕ)
    val(ψ) = logdet(ψ.slater_det) + ψ.jastrow_factor_log
    errs = [abs(real(val(ref) - val(dft)))]
    phase0 = imag(val(ref) - val(dft))
    for k in 1:20
        i = mod1(k, N)
        θi, ϕi = proposal(rng, θ[i], ϕ[i], 0.5)
        θ[i], ϕ[i] = θi, ϕi
        update_wavefunction!(ref, θi, ϕi, i); update_wavefunction!(dft, θi, ϕi, i)
        push!(errs, abs(real(val(ref) - val(dft))))
    end
    # fresh full update at the final positions (checks the incremental path did not drift)
    dft2 = ΨprojDFT(Qstar, P, N, nfill - 1, Matrix{ComplexF64}(I, N, N)); update_wavefunction!(dft2, θ, ϕ)
    d = val(dft2) - val(dft)          # complex logs agree up to 2πi multiples (principal-value phase increments)
    drift = abs(real(d)) + abs(angle(cis(imag(d))))
    # rotational invariance of |Ψ| (filled Lambda levels are L = 0): fresh full updates at X and RX
    Rm = Matrix(qr(randn(rng, 3, 3)).Q)
    xyz = [Rm * [sin(θ[i])cos(ϕ[i]), sin(θ[i])sin(ϕ[i]), cos(θ[i])] for i in 1:N]
    θr = [acos(clamp(x[3], -1, 1)) for x in xyz]; ϕr = [atan(x[2], x[1]) for x in xyz]
    dft3 = ΨprojDFT(Qstar, P, N, nfill - 1, Matrix{ComplexF64}(I, N, N)); update_wavefunction!(dft3, θr, ϕr)
    rot = abs(real(val(dft3) - val(dft2)))
    @printf("    rotation invariance of log|Ψ| (full updates): %.1e\n", rot)
    ok = maximum(errs) < 1e-8 && drift < 1e-8 && rot < 1e-8
    global fails += !ok
    @printf("%-40s %s  max|Δ log|Ψ|| = %.1e over 21 configs, incremental drift %.1e, jk_order %d\n",
            label, ok ? "PASS" : "FAIL", maximum(errs), drift, dft.jk_order)
end
compare("forward  nu=1/3 N=6 Q*=5/2", 5 // 2, 2, 6, 1)
compare("forward  nu=2/5 N=8 Q*=1",   1 // 1, 2, 8, 2)
compare("reverse  nu=2/3 N=8 Q*=-1",  -1 // 1, 2, 8, 2)
compare("reverse  nu=2/3 N=12 Q*=-2", -2 // 1, 2, 12, 2)
compare("reverse  nu=2/3 N=48 Q*=-11", -11 // 1, 2, 48, 2)
println(fails == 0 ? "ALL PASS" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)
