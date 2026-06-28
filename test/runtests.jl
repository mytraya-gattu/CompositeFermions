using CFsOnSphere
using Random
using Test
using Statistics
using LinearAlgebra

@testset "CFsOnSphere.jl" begin

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
