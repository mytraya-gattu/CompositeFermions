using CFsOnSphere
using Random
using Test
using Statistics

@testset "CFsOnSphere.jl" begin

    N = 25
    n = -1
    p = 1

    Qstar = (N // n - n) // 2

    RNG = Random.default_rng()

    θ, ϕ = rand_θ_ϕ_gen(RNG, N)
    @test length(θ) == N
    @test length(ϕ) == N

    l_m_list::Vector{NTuple{2, Rational{Int64}}} = [(abs(Qstar) + ll_index, m) for ll_index in 0:(abs(n) - 1) for m in -(abs(Qstar) + ll_index):(abs(Qstar) + ll_index)]

    ψ = Ψproj(Qstar, p, N, l_m_list)

    res = Vector{ComplexF64}()

    update_wavefunction!(ψ, θ, ϕ)

    push!(res, (logdet(ψ.slater_det) + ψ.jastrow_factor_log * 0.5))

    for i in 1:N

        update_wavefunction!(ψ, θ[i], ϕ[i], i)
        push!(res, (logdet(ψ.slater_det) + ψ.jastrow_factor_log * 0.5))

    end

    θ, ϕ = rand_θ_ϕ_gen(RNG, N)
    update_wavefunction!(ψ, θ, ϕ)
    push!(res, (logdet(ψ.slater_det) + ψ.jastrow_factor_log * 0.5))

    @test isapprox(std(real.(res)), 0.0, atol = 1.0e-10)

    θmesh = collect(0:(π / 100):π)
    accumulated_density = zeros(length(θmesh))
    update_density!(θmesh, θ, accumulated_density)
    @test sum(accumulated_density) == N
    @test accumulated_density[end] == 0.0

    θmesh = collect(0:(π / 100):π)
    ϕmesh = collect(-π:(π / 100):π)
    accumulated_density = zeros(length(θmesh), length(ϕmesh))
    update_density!(θmesh, ϕmesh, θ, ϕ, accumulated_density)
    @test sum(accumulated_density) == N
end
