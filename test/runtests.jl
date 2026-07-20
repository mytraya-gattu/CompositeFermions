using CFsOnSphere
using Random
using Test
using Statistics
using LinearAlgebra
using Distributed

# Test-group selection: `julia --project=. test/runtests.jl <group>...` runs only the named
# groups (so each testset can be exercised as a separate command per the test policy), while
# `Pkg.test()` (no args) runs every group. Groups: core, diagnostics, serial, distributed, resume.
const SELECTED = isempty(ARGS) ? ["all"] : ARGS
rungroup(name) = ("all" in SELECTED) || (name in SELECTED)

rungroup("core") && @testset "CFsOnSphere.jl" begin

    # Deterministic seed: the full-vs-incremental consistency check below is exact for any
    # non-(near-)singular configuration, but an unseeded RNG can occasionally land on a
    # near-singular fully-filled determinant and spuriously fail.
    RNG = MersenneTwister(20240625)

    @testset "rand_θ_ϕ_gen + density" begin
        N = 25
        θ, ϕ = rand_θ_ϕ_gen(RNG, N)
        @test length(θ) == N
        @test length(ϕ) == N
        @test all(0.0 .<= θ .<= π)
        @test all(-π .<= ϕ .<= π)

        θmesh = collect(0:π/100:π)
        accumulated_density = zeros(length(θmesh))
        update_density!(θmesh, θ, accumulated_density)
        @test sum(accumulated_density) == N
        @test accumulated_density[end] == 0.0

        ϕmesh = collect(-π:π/100:π)
        accumulated_density2 = zeros(length(θmesh), length(ϕmesh))
        update_density!(θmesh, ϕmesh, θ, ϕ, accumulated_density2)
        @test sum(accumulated_density2) == N
    end

    @testset "Ψproj full-vs-incremental consistency" begin
        N = 25; n = -1; p = 1
        Qstar = (N//n - n)//2
        l_m_list::Vector{NTuple{2,Rational{Int64}}} =
            [(abs(Qstar)+ll, m) for ll in 0:abs(n)-1 for m in -(abs(Qstar)+ll):(abs(Qstar)+ll)]

        θ, ϕ = rand_θ_ϕ_gen(RNG, N)
        ψ = Ψproj(Qstar, p, N, l_m_list)

        res = ComplexF64[]
        update_wavefunction!(ψ, θ, ϕ)
        push!(res, logdet(ψ.slater_det) + ψ.jastrow_factor_log * 0.50)
        for i in 1:N
            update_wavefunction!(ψ, θ[i], ϕ[i], i)   # move to current position = no-op
            push!(res, logdet(ψ.slater_det) + ψ.jastrow_factor_log * 0.50)
        end
        @test isapprox(std(real.(res)), 0.0, atol=1e-10)
    end

    @testset "Ψproj real-move incremental vs full rebuild" begin
        rng = MersenneTwister(12)
        N, n, p = 9, 3, 2
        Qstar, l_m_list = cf_ground_state_lm(N, n, p)
        θ, ϕ = rand_θ_ϕ_gen(rng, N)

        ψi = Ψproj(Qstar, p, N, l_m_list)
        update_wavefunction!(ψi, θ, ϕ)
        for i in 1:N
            θ[i], ϕ[i] = proposal(rng, θ[i], ϕ[i], 0.5)
            update_wavefunction!(ψi, θ[i], ϕ[i], i)
        end

        ψf = Ψproj(Qstar, p, N, l_m_list)
        update_wavefunction!(ψf, θ, ϕ)

        @test maximum(abs.(ψi.slater_det .- ψf.slater_det)) < 1e-10
        @test maximum(abs.(ψi.u_v_ratio_matrix .- ψf.u_v_ratio_matrix)) < 1e-10
        @test maximum(abs.(ψi.dist_matrix .- ψf.dist_matrix)) < 1e-10
        # Imaginary part of the Jastrow log differs by 2πk (branch cut); compare real part.
        @test abs(real(ψi.jastrow_factor_log - ψf.jastrow_factor_log)) < 1e-9
    end

    @testset "Ψparton real-move incremental vs full rebuild" begin
        rng = MersenneTwister(13)
        N, p = 12, 2
        # Two parton sectors that both hold N = 12 fermions: n = 3 (Q* = 1/2) and
        # n = 2 (Q* = 2) filled Λ-levels.
        Qstar3, lm3 = cf_ground_state_lm(N, 3, p)
        Qstar2, lm2 = cf_ground_state_lm(N, 2, p)
        Qstars = [Qstar3, Qstar2]
        l_m_lists = [lm3, lm2]
        θ, ϕ = rand_θ_ϕ_gen(rng, N)

        ψi = Ψparton(Qstars, p, N, l_m_lists)
        update_wavefunction!(ψi, θ, ϕ)
        for i in 1:N
            θ[i], ϕ[i] = proposal(rng, θ[i], ϕ[i], 0.5)
            update_wavefunction!(ψi, θ[i], ϕ[i], i)
        end

        ψf = Ψparton(Qstars, p, N, l_m_lists)
        update_wavefunction!(ψf, θ, ϕ)

        @test size(ψi.slater_det) == (length(lm3) + length(lm2), N)
        @test maximum(abs.(ψi.slater_det .- ψf.slater_det)) < 1e-10
        @test abs(real(ψi.jastrow_factor_log - ψf.jastrow_factor_log)) < 1e-9
    end

    @testset "ΨoneLL full-vs-incremental consistency" begin
        rng = MersenneTwister(14)
        N, p = 12, 3   # ν = 1/3 Laughlin
        θ, ϕ = rand_θ_ϕ_gen(rng, N)

        ψi = ΨoneLL(p, N)
        update_wavefunction!(ψi, θ, ϕ)
        for i in 1:N
            θ[i], ϕ[i] = proposal(rng, θ[i], ϕ[i], 0.5)
            update_wavefunction!(ψi, θ[i], ϕ[i], i)
        end

        ψf = ΨoneLL(p, N)
        update_wavefunction!(ψf, θ, ϕ)

        @test maximum(abs.(ψi.U .- ψf.U)) < 1e-12
        @test maximum(abs.(ψi.dist_matrix .- ψf.dist_matrix)) < 1e-10
        @test abs(real(ψi.jastrow_factor_log - ψf.jastrow_factor_log)) < 1e-9
    end

    @testset "construct_det_ratios" begin
        rng = MersenneTwister(21)
        N = 6
        S = complex.(randn(rng, N + 2, N), randn(rng, N + 2, N))
        denominator_rows = collect(1:N)
        Sinv = inv(S[denominator_rows, :])
        ref(rows) = det(S[rows, :]) / det(S[denominator_rows, :])

        # Single differing row (the k = 1 fast path).
        nums1 = [vcat(collect(1:N-1), [N+1]), vcat(collect(1:N-1), [N+2])]
        helper1! = construct_det_ratios(denominator_rows, nums1)
        res1 = zeros(ComplexF64, length(nums1))
        helper1!(res1, S, Sinv)
        @test all(abs.(res1 .- ref.(nums1)) .< 1e-10)

        # Two differing rows, including permuted row orders (sign bookkeeping).
        nums2 = [vcat(collect(1:N-2), [N+1, N+2]),
                 vcat(collect(1:N-2), [N+2, N+1]),
                 vcat([N+1], collect(1:N-2), [N-1])]
        helper2! = construct_det_ratios(denominator_rows, nums2)
        res2 = zeros(ComplexF64, length(nums2))
        helper2!(res2, S, Sinv)
        @test all(abs.(res2 .- ref.(nums2)) .< 1e-10)
    end

    @testset "Λ-level builders" begin
        Qstar, lm = cf_ground_state_lm(9, 3, 1)
        @test Qstar == 0//1
        @test length(lm) == 9
        _, lmh = cf_quasihole_lm(9, 3, 1)
        @test length(lmh) == 8
        _, lmp = cf_quasiparticle_lm(9, 3, 1)
        @test length(lmp) == 10
    end

    @testset "ESP all orders vs ground truth" begin
        # Ground truth: coefficients of ∏_i (1 + r_i x) by direct convolution.
        # g[k+1] = e_k (degree-k elementary symmetric polynomial of the roots).
        groundtruth(roots) = begin
            poly = ComplexF64[1.0]
            for r in roots
                nxt = zeros(ComplexF64, length(poly)+1)
                for j in 1:length(poly)
                    nxt[j] += poly[j]; nxt[j+1] += r*poly[j]
                end
                poly = nxt
            end
            poly
        end
        relerr(a, c) = abs(a - c) / max(abs(c), 1e-300)
        maxun = 0.0; maxreg = 0.0
        for M in (1, 2, 3, 5, 8)
            rng = MersenneTwister(100M)
            roots = [complex(randn(rng), randn(rng)) for _ in 1:M]
            g = groundtruth(roots)              # degrees 0 .. M
            Np = M + 1
            for bb in 0:M                        # every order, incl. b=0,1 special cases
                dest = zeros(ComplexF64, bb+1)
                get_symmetric_polynomials!(dest, roots, bb)
                for k in 0:bb; maxun = max(maxun, relerr(dest[k+1], g[k+1])); end
                reg = [i/((Np-1) - i + 1) for i in 1:max(bb,1)]
                destr = zeros(ComplexF64, bb+1)
                get_symmetric_polynomials!(destr, roots, bb, reg)
                acc = 1.0
                for k in 0:bb
                    if k >= 1; acc *= reg[k]; end
                    maxreg = max(maxreg, relerr(destr[k+1], g[k+1]*acc))
                end
            end
        end
        @test maxun < 1e-10
        @test maxreg < 1e-10
        # degrees beyond N-1 must be exactly zero
        rng = MersenneTwister(7); M = 8
        roots = [complex(randn(rng), randn(rng)) for _ in 1:M]
        bb = M + 5
        dest = zeros(ComplexF64, bb+1)
        get_symmetric_polynomials!(dest, roots, bb)
        @test all(dest[M+2:end] .== 0)
    end

    @testset "proposal convention + isotropy" begin
        rng = MersenneTwister(2)
        θ0, ϕ0, σ = 0.9, 0.3, 0.3
        for _ in 1:5000
            θ, ϕ = proposal(rng, θ0, ϕ0, σ)
            @test 0.0 <= θ <= π
            @test -π <= ϕ <= π
        end
        r0 = (sin(θ0)cos(ϕ0), sin(θ0)sin(ϕ0), cos(θ0))
        ns = 200000; acc = 0.0
        for _ in 1:ns
            θ, ϕ = proposal(rng, θ0, ϕ0, σ)
            r = (sin(θ)cos(ϕ), sin(θ)sin(ϕ), cos(θ))
            acc += r[1]*r0[1] + r[2]*r0[2] + r[3]*r0[3]
        end
        @test isapprox(acc/ns, exp(-σ^2/2); atol=5e-3)   # E[cos δθ] = e^{-σ²/2}
    end

    @testset "Ψunproj orbitals + incremental" begin
        rng = MersenneTwister(3)
        N, n, p = 9, 3, 1
        Qstar, l_m_list = cf_ground_state_lm(N, n, p)
        ψ = Ψunproj(Qstar, p, N, l_m_list)
        θ, ϕ = rand_θ_ϕ_gen(rng, N)
        update_wavefunction!(ψ, θ, ϕ)

        Sref = zeros(ComplexF64, N, N)
        for i in 1:N, (oi, (L, Lz)) in enumerate(l_m_list)
            Sref[oi, i] = calculate_ll(L, Qstar, θ[i], ϕ[i])[round(Int, Lz + L) + 1]
        end
        @test maximum(abs.(ψ.slater_det .- Sref)) < 1e-10

        ψ2 = Ψunproj(Qstar, p, N, l_m_list); update_wavefunction!(ψ2, θ, ϕ)
        θ3, ϕ3 = copy(θ), copy(ϕ)
        for i in 1:N
            θ3[i], ϕ3[i] = proposal(rng, θ3[i], ϕ3[i], 0.5)
            update_wavefunction!(ψ2, θ3[i], ϕ3[i], i)
        end
        ψ3 = Ψunproj(Qstar, p, N, l_m_list); update_wavefunction!(ψ3, θ3, ϕ3)
        @test maximum(abs.(ψ2.slater_det .- ψ3.slater_det)) < 1e-10
        # imaginary part of the Jastrow log differs by 2πk (branch cut); compare real part.
        @test abs(real(ψ2.jastrow_factor_log - ψ3.jastrow_factor_log)) < 1e-9
    end

    @testset "Sherman-Morrison inverse tracking (Ψunproj)" begin
        rng = MersenneTwister(4)
        N, n, p = 9, 3, 2
        Qstar, l_m_list = cf_ground_state_lm(N, n, p)
        ψc = Ψunproj(Qstar, p, N, l_m_list); ψn = Ψunproj(Qstar, p, N, l_m_list)
        θ, ϕ = rand_θ_ϕ_gen(rng, N)
        update_wavefunction!(ψc, θ, ϕ); copy!(ψn, ψc); initialize_inverse!(ψc)
        @test maximum(abs.(ψc.slater_det_inv .- inv(ψc.slater_det))) < 1e-9

        temp = zeros(ComplexF64, N); σ = 0.4; siter = 1; naccept = 0
        for step in 1:3000
            nθ, nϕ = proposal(rng, θ[siter], ϕ[siter], σ)
            update_wavefunction!(ψn, nθ, nϕ, siter)
            dr = slater_det_ratio(ψc, ψn, siter)
            if step % 500 == 0
                @test abs(dr - det(ψn.slater_det)/det(ψc.slater_det)) < 1e-6 * max(1, abs(dr))
            end
            if 2*real(log(dr) + ψn.jastrow_factor_log - ψc.jastrow_factor_log) >= log(rand(rng))
                θ[siter] = nθ; ϕ[siter] = nϕ
                update_inverse!(ψc, ψn, siter, dr, temp)
                copy!(ψc, ψn, siter); naccept += 1
            else
                copy!(ψn, ψc, siter)
            end
            siter = mod(siter, N) + 1
        end
        @test naccept > 0
        @test maximum(abs.(ψc.slater_det_inv .- inv(ψc.slater_det))) < 1e-7
    end

    @testset "build_extended_slater! (quasihole/quasiparticle)" begin
        rng = MersenneTwister(7)
        N, n, p = 9, 3, 1
        # A quasiparticle l_m_list has N+1 orbitals; with N electrons the electron block is
        # the non-square (N+1)×N `slater_det`, and one fixed orbital column makes it square.
        Qstar, lm = cf_quasiparticle_lm(N, n, p)
        @test length(lm) == N + 1
        ψ = Ψunproj(Qstar, p, N, lm)
        @test size(ψ.slater_det) == (N + 1, N)
        θ, ϕ = rand_θ_ϕ_gen(rng, N)
        update_wavefunction!(ψ, θ, ϕ)

        # One fixed quasihole coordinate → an (N+1)×1 orbital column.
        θqh, ϕqh = 0.7, 1.1
        qh = zeros(ComplexF64, N + 1, 1)
        for (oi, (L, Lz)) in enumerate(lm)
            qh[oi, 1] = calculate_ll(L, Qstar, θqh, ϕqh)[round(Int, Lz + L) + 1]
        end

        Sfull = zeros(ComplexF64, N + 1, N + 1)
        lu_full = build_extended_slater!(Sfull, ψ, qh)
        ref = hcat(ψ.slater_det, qh)
        @test maximum(abs.(Sfull .- ref)) == 0.0
        @test abs(logdet(lu_full) - logdet(ref)) < 1e-8

        # Amplitude extraction: transpose(lu) \ e_end == row `end` of inv(Sfull).
        e_end = zeros(ComplexF64, N + 1); e_end[end] = 1
        a = transpose(lu_full) \ e_end
        @test maximum(abs.(a .- inv(Sfull)[end, :])) < 1e-9
    end

    @testset "Ψproj single-bound-pair normalization" begin
        rng = MersenneTwister(5)
        N, n, p = 10, 2, 4   # ν = 2/9 via outer Jastrow p = 2·p̃ (p̃ = 2)
        Qstar, l_m_list = cf_ground_state_lm(N, n, p)
        ψ = Ψproj(Qstar, p, N, l_m_list)
        @test ψ.reg_coeffs[1] ≈ 1/(N-1)   # 1/C(N-1, 1), single bound vortex pair
        θ, ϕ = rand_θ_ϕ_gen(rng, N)
        update_wavefunction!(ψ, θ, ϕ)
        @test isfinite(real(logdet(ψ.slater_det)))
    end

end

# ---------------------------------------------------------------------------------------------
# Convergence diagnostics (split-R̂, effective sample size).
# ---------------------------------------------------------------------------------------------
rungroup("diagnostics") && @testset "diagnostics: split-R̂ and ESS" begin

    # (a) Four iid N(0,1) chains of 10_000 draws: well-mixed, ESS near the true value N.
    rng = Xoshiro(20240718)
    chains = [randn(rng, 10_000) for _ in 1:4]
    @test split_rhat(chains) < 1.01

    e1 = effective_sample_size(chains[1])
    @test 8_000 < e1 <= 10_000            # iid single chain: ESS ≈ N
    em = effective_sample_size(chains)
    @test 32_000 < em <= 40_000           # iid pooled: ESS ≈ 4N

    # AR(1) with φ = 0.8: ESS ≈ N·(1-φ)/(1+φ) — a known limiting case for the estimator.
    φ = 0.8; n = 20_000; x = zeros(n); r2 = Xoshiro(7)
    for i in 2:n; x[i] = φ * x[i-1] + randn(r2); end
    ar = effective_sample_size(x)
    @test ar < n / 3
    @test isapprox(ar, n * (1 - φ) / (1 + φ); rtol = 0.15)

    # (b) Two chains offset by 5σ: strongly non-mixed, split-R̂ ≫ 1.
    c1 = randn(Xoshiro(1), 5_000)
    c2 = randn(Xoshiro(2), 5_000) .+ 5.0
    @test split_rhat([c1, c2]) > 2.0

    # (c) Degenerate inputs must return NaN / a sane value, never throw.
    @test isnan(split_rhat([randn(100)]))                         # single chain
    @test isnan(split_rhat([fill(3.0, 100), fill(3.0, 100)]))     # constant traces
    @test isnan(effective_sample_size(fill(3.0, 100)))            # constant trace
    @test effective_sample_size([1.0]) == 1.0                     # too-short trace
    @test isnan(split_rhat([[1.0, 2.0], [1.0, 2.0]]))             # traces shorter than 4

    # (d) Online integrated autocorrelation time τ_int (variance-ratio estimator). Convention:
    # τ_int = 1/2 for uncorrelated samples (so ESS_steps = T·M / (2 τ_int)). The chains here use
    # thinning 1, so the true steps-per-chain T equals the trace length.
    let
        r3 = Xoshiro(11)
        iid = [randn(r3, 500) for _ in 1:256]
        @test isapprox(integrated_autocorr_time(iid, 500), 0.5; atol = 0.1)

        # AR(1) with φ = 0.9: analytic τ_int = (1+φ)/(2(1-φ)) = 9.5.
        φ = 0.9; L = 4_000; r4 = Xoshiro(7)
        ar = Vector{Vector{Float64}}()
        for _ in 1:300
            x = zeros(L)
            for i in 2:L; x[i] = φ * x[i-1] + randn(r4); end
            push!(ar, x)
        end
        @test isapprox(integrated_autocorr_time(ar, L), (1 + φ) / (2 * (1 - φ)); rtol = 0.2)

        # Degenerate inputs return NaN, never throw.
        @test isnan(integrated_autocorr_time([randn(100)], 100))              # single chain
        @test isnan(integrated_autocorr_time([fill(2.0, 50), fill(2.0, 50)], 50))  # constant
    end
end

# ---------------------------------------------------------------------------------------------
# Multi-chain driver, serial (in-process) path: nprocs() == 1 runs the chains sequentially.
# ---------------------------------------------------------------------------------------------
rungroup("serial") && @testset "driver serial (in-process)" begin

    ref = load(joinpath(@__DIR__, "data_10_particles_1_3_filling_factor_1_chain_number.jld2"))

    dir = mktempdir()
    res = sample_density_chains(dir, 10, 1, 2; n_chains = 2, seed = 2718,
                                num_thermalization = 10_000, num_steps = 50_000,
                                record_every = 500, checkpoint_every = 25_000)

    @test isfile(joinpath(dir, "chain_1.jld2"))
    @test isfile(joinpath(dir, "chain_2.jld2"))

    rep = monitor_chains(dir)
    @test rep.n_read == 2
    @test all(0.35 .< rep.acceptance .< 0.65)   # ARM targets 0.50
    @test isfinite(rep.rhat)

    d = res["observables"]["density"]
    @test length(d) == length(ref["density"])   # same 499-bin mesh as the reference
    @test res["grids"]["theta grid"] ≈ ref["theta grid"]

    # The reference is a single 10^6-step chain; this is 2×5·10^4 steps, so per-bin MC noise is
    # large. Compare the *interior mean density level* (bulk plateau) and the overall level,
    # which are robust to bin-by-bin noise, rather than pointwise densities.
    bulk = 100:400
    @test isapprox(mean(d[bulk]), mean(ref["density"][bulk]); rtol = 0.03)
    @test isapprox(sum(d) / length(d), sum(ref["density"]) / length(ref["density"]); rtol = 0.02)
    @test isapprox(sum(d) / length(d), 10 / (4π); rtol = 0.02)   # ∫ n dΩ = N over the sphere

    # Improvement 3: run_chains must return promptly once the chains finish and NOT quantize its
    # wall time to `monitor_every`. Set `monitor_every` huge and check the tiny run still returns
    # quickly (the serial in-process path never sleeps on it; this guards against regressions).
    dir2 = mktempdir()
    twall = @elapsed sample_density_chains(dir2, 10, 1, 2; n_chains = 2, seed = 99,
                                           num_thermalization = 1_000, num_steps = 4_000,
                                           record_every = 500, checkpoint_every = 4_000,
                                           monitor_every = 1.0e6)
    @test twall < 60.0
end

# ---------------------------------------------------------------------------------------------
# Multi-chain driver, distributed path: run over worker processes, check determinism, clean up.
# ---------------------------------------------------------------------------------------------
rungroup("distributed") && @testset "driver distributed (addprocs)" begin

    addprocs(2; exeflags = "--project=$(Base.active_project())")
    try
        @everywhere using CFsOnSphere
        @everywhere using LinearAlgebra
        @everywhere LinearAlgebra.BLAS.set_num_threads(1)

        function traces_for(seed)
            dir = mktempdir()
            sample_density_chains(dir, 10, 1, 2; n_chains = 2, seed = seed,
                                  num_thermalization = 3_000, num_steps = 12_000,
                                  record_every = 300, checkpoint_every = 6_000,
                                  monitor_every = 1.0)
            return [load(joinpath(dir, "chain_$(c).jld2"))["logpdf trace"] for c in 1:2]
        end

        a = traces_for(31415)
        @test length(a[1]) == 40 && length(a[2]) == 40
        b = traces_for(31415)
        @test a[1] == b[1] && a[2] == b[2]       # deterministic given the seed
        cc = traces_for(27182)
        @test a[1] != cc[1]                        # a different seed gives a different chain
    finally
        rmprocs(workers())
    end
end

# ---------------------------------------------------------------------------------------------
# Multi-chain driver, resume: continue a chain from its checkpoint; step counts must add up.
# ---------------------------------------------------------------------------------------------
rungroup("resume") && @testset "driver resume from checkpoint" begin

    dir = mktempdir()

    # First leg: run to 15_000 steps and checkpoint (simulates being killed after a checkpoint).
    sample_density_chains(dir, 10, 1, 2; n_chains = 2, seed = 161,
                          num_thermalization = 3_000, num_steps = 15_000,
                          record_every = 500, checkpoint_every = 15_000)
    s1 = [load(joinpath(dir, "chain_$(c).jld2"))["number of steps"] for c in 1:2]
    @test all(s1 .== 15_000)

    # Resume to 40_000 total steps; thermalization is skipped, the remaining 25_000 steps run.
    res = sample_density_chains(dir, 10, 1, 2; n_chains = 2, seed = 161,
                                num_thermalization = 3_000, num_steps = 40_000,
                                record_every = 500, checkpoint_every = 40_000, resume = true)
    s2 = [load(joinpath(dir, "chain_$(c).jld2"))["number of steps"] for c in 1:2]
    @test all(s2 .== 40_000)                     # 15_000 + 25_000 add up
    @test all(isfinite, res["observables"]["density"])
    @test isapprox(sum(res["observables"]["density"]) / 499, 10 / (4π); rtol = 0.02)

    # --- Bit-identical resume (improvement 2). An interrupted+resumed chain must reproduce an
    # uninterrupted run with the same seed BIT-FOR-BIT (logpdf trace + positions). This works
    # because the RNG state is checkpointed (schema >= 2) and the incrementally accumulated
    # `jastrow_factor_log` is restored (a full rebuild alone drifts ~1e-13 from the incremental
    # value; the rebuilt Slater matrix / caches are already bit-identical). ---
    @testset "bit-identical resume" begin
        K = 20_000
        dfull = mktempdir()
        sample_density_chains(dfull, 10, 1, 2; n_chains = 2, seed = 999,
                              num_thermalization = 3_000, num_steps = K,
                              record_every = 500, checkpoint_every = K)
        dres = mktempdir()
        sample_density_chains(dres, 10, 1, 2; n_chains = 2, seed = 999,
                              num_thermalization = 3_000, num_steps = K ÷ 2,
                              record_every = 500, checkpoint_every = K ÷ 2)
        sample_density_chains(dres, 10, 1, 2; n_chains = 2, seed = 999,
                              num_thermalization = 3_000, num_steps = K,
                              record_every = 500, checkpoint_every = K, resume = true)
        for c in 1:2
            a = load(joinpath(dfull, "chain_$(c).jld2"))
            b = load(joinpath(dres,  "chain_$(c).jld2"))
            @test a["logpdf trace"] == b["logpdf trace"]   # bit-identical trace
            @test a["theta vector"] == b["theta vector"]   # bit-identical positions
            @test a["phi vector"]   == b["phi vector"]
        end
    end

    # --- Old-schema fallback. A checkpoint predating RNG-state storage (schema < 2, no
    # "rng state" key) must still resume — reseeded, with a printed note — rather than error. ---
    @testset "old-schema resume fallback" begin
        dold = mktempdir()
        sample_density_chains(dold, 10, 1, 2; n_chains = 1, seed = 77,
                              num_thermalization = 2_000, num_steps = 8_000,
                              record_every = 500, checkpoint_every = 8_000)
        p = joinpath(dold, "chain_1.jld2")
        d = load(p)
        delete!(d, "rng state"); delete!(d, "jastrow log")   # simulate a schema-1 checkpoint
        d["schema version"] = 1
        save(p, d)
        sample_density_chains(dold, 10, 1, 2; n_chains = 1, seed = 77,
                              num_thermalization = 2_000, num_steps = 16_000,
                              record_every = 500, checkpoint_every = 16_000, resume = true)
        @test load(p)["number of steps"] == 16_000        # 8_000 + 8_000 add up, no error
    end
end
