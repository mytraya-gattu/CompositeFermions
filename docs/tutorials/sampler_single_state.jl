# using MKL - If available, use MKL.
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)

using CFsOnSphere

using Random
const global RNG = Random.default_rng()

function gibbs_sampler(filename::String, Qstar::Rational{Int64}, l_m_list::Vector{NTuple{2, Rational{Int64}}}, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    system_size = length(l_m_list)

    Ψcurrent = Ψproj(Qstar, p, system_size, l_m_list)
    Ψnext = Ψproj(Qstar, p, system_size, l_m_list)

    θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, system_size)

    θnext = copy(θcurrent)
    ϕnext = copy(ϕcurrent)

    logpdf(ψ::Ψproj) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

    σ = pi / sqrt(12.0)

    sampling_iter, σ, δt_therm, thermalization_acceptance_rate = gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, num_thermalization)

    data = Dict("theta vector" => θcurrent, "phi vector" => ϕcurrent, "thermalization acceptance rate" => thermalization_acceptance_rate, "number of thermalization steps" => num_thermalization, "thermalization duration" => δt_therm, "step size" => σ)

    save(filename, data)

    num_samples_accepted = zero(Int64)

    logpdf_current::Float64 = 0.0
    logpdf_next::Float64 = 0.0

    logpdf_current = logpdf(Ψcurrent)
    logpdf_next = logpdf_current

    rgrid::Vector{Float64} = LinRange(0.0, 2.0, 5_000) ### 5_000 points between 0 and 2.

    dr = rgrid[2] - rgrid[1]

    accumulated_pair_density = zeros(Float64, length(rgrid) - 1)

    current_distance_distribution = zeros(Float64, length(rgrid) - 1)

    for i in 1:(system_size - 1)
        for j in (i + 1):system_size
            r = Ψcurrent.dist_matrix[j - 1, i]
            current_distance_distribution[ceil(Int64, r / dr)] += 1.0
        end
    end

    θmesh = map(x -> acos(x), LinRange(1.0, -1.0, 500))

    Agrid = 2.0 * pi .* (cos.(θmesh[begin:(end - 1)]) .- cos.(θmesh[(begin + 1):end]))

    accumulated_density = zeros(Float64, length(θmesh) - 1)

    t0 = time()

    for monte_carlo_iter in 1:num_steps

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)

        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        logpdf_next = logpdf(Ψnext)

        if logpdf_next - logpdf_current >= log(rand())

            for i in 1:(system_size - 1)
                @inbounds current_distance_distribution[ceil(Int64, Ψnext.dist_matrix[i, sampling_iter] / dr)] += 1.0
                @inbounds current_distance_distribution[ceil(Int64, Ψcurrent.dist_matrix[i, sampling_iter] / dr)] -= 1.0
            end

            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]

            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next

            num_samples_accepted += 1

        else

            # If the proposed state is rejected, we set the proposed state to the current state.
            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]

            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current

        end

        accumulated_pair_density .+= current_distance_distribution
        update_density!(θmesh, θcurrent, accumulated_density)

        sampling_iter = mod(sampling_iter, system_size) + 1

        if monte_carlo_iter == num_steps || mod(monte_carlo_iter, 5 * 10^5) == 0

            data["number of steps"] = monte_carlo_iter
            data["acceptance rate"] = num_samples_accepted / monte_carlo_iter
            data["monte carlo duration"] = time() - t0
            data["pair densities"] = accumulated_pair_density ./ monte_carlo_iter
            data["r grid"] = 0.5 .* (rgrid[1:(end - 1)] .+ rgrid[2:end])
            data["density"] = accumulated_density ./ monte_carlo_iter ./ Agrid
            data["theta grid"] = 0.5 .* (θmesh[1:(end - 1)] .+ θmesh[2:end])

            save(filename, data)

        end

    end

    return
end


function sample_cf_gs(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N // n - n) // 2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar) + abs(n) - 1) for Lz in -L:1:L]

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(2 * n * p + 1)_filling_factor_$(chain_number)_chain_number.jld2")

    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

function sample_cf_qh(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N // n - n) // 2
    Lqh = abs(Qstar) + abs(n) - 1

    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar) + abs(n) - 1) for Lz in -L:1:L if !(L == Lqh && Lz == Lqh)]

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(2 * n * p + 1)_filling_factor_$(chain_number)_chain_number.jld2")

    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

function sample_cf_qp(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N // n - n) // 2
    Lqp = abs(Qstar) + abs(n)

    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar) + abs(n) - 1) for Lz in -L:1:L]
    push!(l_m_list, (Lqp, Lqp))

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(2 * n * p + 1)_filling_factor_$(chain_number)_chain_number.jld2")
    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
