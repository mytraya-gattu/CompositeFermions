using CFsOnSphere
using Random
const global RNG = Random.default_rng()

function gibbs_sampler(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 1 * 10^6)

    ν::Rational{Int64} = n//(2*n*p+1)
    Qstar::Rational{Int64} = (N//n-n)//2
    Q::Rational{Int64} = Qstar + p*(N-1)

    l_m_list::Vector{NTuple{2, Rational{Int64}}} = [(abs(Qstar)+ll_index, m) for ll_index in 0:abs(n)-1 for m in -(abs(Qstar)+ll_index):(abs(Qstar)+ll_index)]

    Ψcurrent::Ψproj = Ψproj(Qstar, p, N, l_m_list)
    Ψnext::Ψproj = Ψproj(Qstar, p, N, l_m_list)

    # filename = "$(folder_name)/data_$(N)_particles_$(numerator(ν))_$(denominator(ν))_filling_factor_$(chain_number)_chain_number.jld2"
    filename = joinpath(folder_name, "data_$(N)_particles_$(numerator(ν))_$(denominator(ν))_filling_factor_$(chain_number)_chain_number.jld2")
    
    θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, N)
    
    θnext = copy(θcurrent)
    ϕnext = copy(ϕcurrent)

    logpdf(Ψ::Ψproj) = 2.0 * real(logdet(Ψ.slater_det) + Ψ.jastrow_factor_log)
   
    δt_therm::Float64 = 0.0
    thermalization_acceptance_rate = 0.0

    sampling_iter::Int64 = 1

    sampling_iter, σ, δt_therm, thermalization_acceptance_rate = gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, pi/sqrt(12.0), logpdf, num_thermalization)
        
    save(filename, Dict("theta vector"=>θcurrent, "phi vector"=>ϕcurrent, "thermalization acceptance rate"=>thermalization_acceptance_rate, "number of thermalization steps"=>num_thermalization, "thermalization duration"=>δt_therm, "step size"=>σ))

    num_samples_accepted::Int64 = 0
    logpdf_current::Float64 = 0.0
    logpdf_next::Float64 = 0.0

    update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
    copy!(Ψnext, Ψcurrent)

    logpdf_current = logpdf(Ψcurrent)
    logpdf_next = logpdf_current

    rgrid::Vector{Float64} = LinRange(0.0, 2.0, 5_000) ### 5_000 points between 0 and 2.

    dr = rgrid[2] - rgrid[1]

    accumulated_pair_density = zeros(Float64, length(rgrid)-1)
    
    current_distance_distribution = zeros(Float64, length(rgrid)-1)
    
    for i in 1:system_size-1
        for j in i+1:system_size
            r = Ψcurrent.dist_matrix[j-1, i]
            current_distance_distribution[ceil(Int64, r/dr)] += 1.0
        end
    end

    θmesh = map(x->acos(x), LinRange(1.0, -1.0, 500))
    Agrid = 2.0 * pi .* (cos.(θmesh[begin:end-1]) .- cos.(θmesh[begin+1:end]))

    accumulated_density = zeros(Float64, length(θmesh))
    t0 = time()

    for monte_carlo_iter in 1:num_steps

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        logpdf_next = logpdf(Ψnext)
        
        if logpdf_next - logpdf_current >= log(rand())

            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]

            @views @inbounds current_coulomb_energy += sum(1 ./ Ψnext.dist_matrix[:, sampling_iter]) - sum(1 ./ Ψcurrent.dist_matrix[:, sampling_iter])

            copy!(Ψcurrent, Ψnext, sampling_iter)

            logpdf_current = logpdf_next
            num_samples_accepted += 1

        else

            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]

            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current

        end
        
        accumulated_coulomb_energy += current_coulomb_energy
        update_density!(θmesh, θcurrent, accumulated_density)

        sampling_iter = mod(sampling_iter, N) + 1

        if mod(monte_carlo_iter, div(num_steps, 5))==0 || monte_carlo_iter == num_steps

            save(filename, Dict("theta vector"=>θcurrent, "phi vector"=>ϕcurrent, "thermalization acceptance rate"=>thermalization_acceptance_rate, "number of thermalization steps"=>num_thermalization, "thermalization duration"=>δt_therm, "number of steps"=>monte_carlo_iter, "acceptance rate"=>num_samples_accepted/monte_carlo_iter, "monte carlo duration"=>time()-t0, "step size"=>σ, "coulomb energy"=>accumulated_coulomb_energy/monte_carlo_iter/sqrt(N / (2 * ν)), "density"=>accumulated_density[begin:end-1] ./ monte_carlo_iter ./ dA, "theta mesh"=>0.50 .* (θmesh[begin:end-1] .+ θmesh[begin+1:end])))

        end
        
    end

    return
end

gibbs_sampler(pwd(), 1, 10, 1, 1)
