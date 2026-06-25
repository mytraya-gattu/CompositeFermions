using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using CFsOnSphere
using CFsOnSphere.MonteCarloOnSphere: arm_parameters, arm_scale_factor
using CoordinateTransformations
using Random
using Statistics
using Measurements
# using OnlineStats

mutable struct MonteCarloState
    Ψ::Ψproj
    θ::Vector{Float64}
    ϕ::Vector{Float64}
    θproposed::Vector{Float64}
    ϕproposed::Vector{Float64}
    logpsi::ComplexF64
    RNG::AbstractRNG
end

"""
    slerp!(θ0, ϕ0, θ1, ϕ1, θstored, ϕstored, t)

Performs spherical linear interpolation (SLERP) between two sets of spherical coordinates.
"""
function slerp!(θ0::Vector{Float64}, ϕ0::Vector{Float64}, θ1::Vector{Float64}, ϕ1::Vector{Float64}, θstored::Vector{Float64}, ϕstored::Vector{Float64}, t::Float64)
    @simd for iter in eachindex(θ0)
        @inbounds r0 = CartesianFromSpherical()(Spherical(1.0, ϕ0[iter], pi / 2 - θ0[iter]))
        @inbounds r1 = CartesianFromSpherical()(Spherical(1.0, ϕ1[iter], pi / 2 - θ1[iter]))

        ω = acos(clamp(dot(r0, r1), -1.0, 1.0))
        sω = sin(ω) + 1.0e-10

        r = r1 + t * (r0 - r1)
        # r = sin((1 - t) * ω) / sω * r1 + sin(t * ω) / sω * r0
        sph = SphericalFromCartesian()(r)
        @inbounds θstored[iter] = pi / 2 - sph.ϕ
        @inbounds ϕstored[iter] = sph.θ
    end
    return
end

"""
    emcee_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps, num_chains)

Performs ensemble MCMC sampling for a given wavefunction.
"""
function emcee_sampler(
        filename::String,
        Qstar::Rational{Int64},
        l_m_list::Vector{NTuple{2, Rational{Int64}}},
        p::Int64;
        num_thermalization::Int64 = 5 * 10^5,
        num_steps::Int64 = 10^6,
        num_chains::Int64 = Threads.nthreads(),
        a::Float64 = 2.0
    )

    system_size = length(l_m_list)
    logpsi(ψ::Ψproj) = logdet(ψ.slater_det) + ψ.jastrow_factor_log

    # Helper function to initialize Monte Carlo state
    function _build_monte_carlo_state()
        RNG = Random.TaskLocalRNG()
        θ, ϕ = rand_θ_ϕ_gen(RNG, system_size)
        ψ = Ψproj(Qstar, 2p, system_size, l_m_list)
        return MonteCarloState(ψ, θ, ϕ, copy(θ), copy(ϕ), logpsi(ψ), RNG)
    end

    MC_states = [_build_monte_carlo_state() for _ in 1:num_chains]

    # Thermalization function
    function _thermalize!(MC_state::MonteCarloState, σinit::Float64, num_thermalization::Int64)
        gibbs_thermalization!(
            MC_state.RNG,
            MC_state.Ψ,
            deepcopy(MC_state.Ψ),
            MC_state.θ,
            MC_state.ϕ,
            MC_state.θproposed,
            MC_state.ϕproposed,
            σinit,
            x -> 2.0 * real(logpsi(x)),
            num_thermalization
        )

        MC_state.logpsi = logpsi(MC_state.Ψ) ### Update logpsi after thermalization
        MC_state.θproposed .= MC_state.θ
        MC_state.ϕproposed .= MC_state.ϕ

        return
    end

    Threads.@threads for i in 1:num_chains
        _thermalize!(MC_states[i], pi / √12, num_thermalization)
    end

    accepted = zeros(Bool, num_chains)
    energies = zeros(Float64, num_chains)
    # Helper function for a single MCMC step
    function _step!(
            move_first_half::Bool,
            MC_states::Vector{MonteCarloState},
            inverse_cdf::Function, accepted::Vector{Bool}, energies::Vector{Float64}
        )

        states_to_move = move_first_half ? (1:div(num_chains, 2)) : ((div(num_chains, 2) + 1):num_chains)
        states_to_not_move = move_first_half ? ((div(num_chains, 2) + 1):num_chains) : (1:div(num_chains, 2))

        Threads.@threads for k in states_to_move
            MC_current = MC_states[k]
            paired_walker = rand(states_to_not_move)

            Z = inverse_cdf(rand(MC_current.RNG))

            @inbounds slerp!(
                MC_current.θ,
                MC_current.ϕ,
                MC_states[paired_walker].θ,
                MC_states[paired_walker].ϕ,
                MC_current.θproposed,
                MC_current.ϕproposed,
                Z
            )

            update_wavefunction!(MC_current.Ψ, MC_current.θproposed, MC_current.ϕproposed)
            logpsi_new = logpsi(MC_current.Ψ)

            if 2.0 * real(logpsi_new - MC_current.logpsi) + (2 * system_size - 1) * log(Z) >= log(rand(MC_current.RNG))
                # Accept the move
                MC_current.θ, MC_current.ϕ = MC_current.θproposed, MC_current.ϕproposed
                MC_current.logpsi = logpsi_new
                accepted[k] = true
            else
                accepted[k] = false
            end
            energies[k] += sum(0.5 ./ MC_current.Ψ.dist_matrix)
        end

        return accepted
    end

    # Preallocate matrices for results
    θmat = zeros(Float64, system_size, num_chains, num_steps)
    ϕmat = zeros(Float64, system_size, num_chains, num_steps)

    inverse_cdf(u) = (1 / sqrt(a) + u * (√a - 1 / √a))^2
    # Main MCMC loop
    total_accepted = 0
    for step in 1:num_steps

        _step!(false, MC_states, inverse_cdf, accepted, energies)
        _step!(true, MC_states, inverse_cdf, accepted, energies)

        total_accepted += sum(accepted) ### Count the number of accepted moves

        for i in 1:num_chains
            θmat[:, i, step] .= MC_states[i].θ
            ϕmat[:, i, step] .= MC_states[i].ϕ
        end

        if step % 1000 == 0
            println("Step $(step) of $(num_steps) completed. Acceptance rate: $(total_accepted / (step * num_chains))")
        end

    end

    return energies ./ num_steps
end


# function sample_cf_gs(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

#     Qstar = (N//n-n)//2
#     l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]

#     filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(2*n*p+1)_filling_factor_$(chain_number)_chain_number.jld2")

#     emcee_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

#     return

# end

# sample_cf_gs("./data/", parse(Int64, ARGS[1]), 4, 1, 1)


let

    N = 4
    n = 1
    p = 2

    Qstar = (N // n - n) // 2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar) + abs(n) - 1) for Lz in -L:1:L]


    E = emcee_sampler(
        "", Qstar, l_m_list, p,
        num_thermalization = 1 * 10^5,
        num_steps = 10^4,
        num_chains = 128,
        a = 2.0
    )

    m = measurement(mean(E), std(E))
    @show (m - N^2 / 2.0) / sqrt(N / (2 * n / (2 * n + 1))) / N # should be close to 0

end
