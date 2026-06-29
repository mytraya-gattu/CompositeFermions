#  # Basic usage of CompositeFermion.jl

# It is recommended that if the user has access to MKL, they should use it.
## using MKL - If available, use MKL.
using LinearAlgebra
# A common pitfall, when it comes to computing determinants, is a sudden drop in performance when the size of the matrix is more than ~ 100 x 100. This is because BLAS switches to a multi-threaded implementation by default, which gives worse performance than single-threading. Thi line sets the number of threads used by BLAS to 1.
LinearAlgebra.BLAS.set_num_threads(1)

# ## Example 1. Calculating the density and inter-particle distance distribution of a single CF Slater determinant state using the Metropolis-Hastings-Gibbs algorithm.

# First, we import the CFsOnSphere module.
using CFsOnSphere

# Next, we import the Random module, and define the random number generator that our Monte Carlo simulation must use. We will use the default random number generator provided by the Random module.
using Random
const global RNG = Random.default_rng()

# We will now write a function, to which we will pass:
# 1. the effective monopole strength Qstar felt by composite fermions,
# 2. their Lambda level occupation (l_m_list) represented as pairs (L, Lz) - L is the angular momentum and Lz the azimuthal quantum number, 
# 3. p: the number of flux quanta (vortices) bound to each electron (even; p=2 is one pair)
# 4. num_thermalization: the number of thermalization steps which by default is set to 500_000
# 4. num_steps: the number of (actual) Monte Carlo steps which by default is set to 1_000_000
function gibbs_sampler(filename::String, N::Int64, Qstars::Vector{Rational{Int64}}, l_m_lists::Vector{Vector{NTuple{2, Rational{Int64}}}}, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    if isempty(l_m_lists)

        gibbs_sampler(filename, N, p, num_thermalization, num_steps)
        
        return
    end
    
    @assert length(Qstars) == length(l_m_lists) "Qstars and l_m_lists must have the same length."
    @assert N >= 2 "A minimum of two particles necessary."
    @assert (unique(map(length, l_m_lists)) == 1) "All l_m_lists must have the same length."
    @assert (length(l_m_lists[1]) == N) "Each parton like state must contain N particles."
    
    system_size = N
    
    Qstars_unique = unique(Qstars)
    l_m_lists_combined = [union(l_m_lists[findall(isequal(Qstar), Qstars)]...) for Qstar in Qstars_unique]

    Ψcurrent_vec = []

    for iter in eachindex(Qstars_unique)

        Qstar = Qstars_unique[iter]
        l_m_list = l_m_lists_combined[iter]

        if iter == 1
            push!(Ψcurrent_vec, Ψproj(Qstar, p, system_size, l_m_list))
        else
            push!(Ψcurrent_vec, Ψproj(Qstar, 0, system_size, l_m_list))
        end
    
    end

    Ψnext_vec = deepcopy.(Ψcurrent_vec)
    # Next, we create two instances of the Ψproj type, which will be used to store the wavefunctions of the current state and the proposed state during our Monte Carlo simulation.
    
    # We will now initialize our Monte Carlo simulation by generating random values for the initial positions of our particles.
    # Again, we create two instances of the θ and ϕ vectors, which will be used to store the current and proposed positions of the particles.

    θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, system_size)

    θnext = copy(θcurrent)
    ϕnext = copy(ϕcurrent)

    # Next, we define the log of the probability density according to which we will sample the positions of the particles. In this case, it is simply the probability density of the CF Slater determinant state. Recall, that the CF wavefunction is equal to the product of a Slater determinant containing electrons at Qstar and the Jastrow factor.
    function logpdf(Ψ_vec)

        logpdf_val = 0.0
        
        for iter in eachindex(Qstars)

            

        end
        
    end

    # We are now almost done setting up. The only thing we need to specify is provide an initial guess for the step size σ for our Gaussian proposal distribution. By default, we will set it to π/√12 (which is the standard deviation of a uniform distribution on the interval [0, π]).
    σ = pi/sqrt(12.0)

    # We can now start our thermalization. We will use the function gibbs_thermalization! to perform the thermalization. This function will return the final sampling_iter, the final value of σ (adjusted to achieve an acceptance rate of 0.50), the duration of the thermalization, and the acceptance rate during the thermalization.

    sampling_iter, σ, δt_therm, thermalization_acceptance_rate = gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, num_thermalization)

    # We will now save the data from the thermalization to a file. This will allow us to restart the simulation from this point, if needed.
    data = Dict("theta vector"=>θcurrent, "phi vector"=>ϕcurrent, "thermalization acceptance rate"=>thermalization_acceptance_rate, "number of thermalization steps"=>num_thermalization, "thermalization duration"=>δt_therm, "step size"=>σ)

    # CFsOnSphere imports the save function from the JLD2 package, which allows us to save the data to a file in the JLD2 format.
    save(filename, data)

    # We will now perform the actual Monte Carlo simulation i.e. start estimating the density and inter-particle distance distribution of the CF Slater determinant state.

    # We start by creating a variable num_samples_accepted, to keep track of the number of samples accepted during the Monte Carlo simulation. Normally, you needn't keep track of this, but it is useful for debugging purposes.
    num_samples_accepted = zero(Int64)

    # We also create two variables logpdf_current and logpdf_next to store the log of the probability density of the current and proposed states.
    logpdf_current::Float64 = 0.0
    logpdf_next::Float64 = 0.0

    # We now compute the logpdf of the current state i.e. corresponding to θcurrent, ϕcurrent.
    logpdf_current = logpdf(Ψcurrent)
    logpdf_next = logpdf_current

    # We will now create a grid of points between 0 and 2.0 (the maximum inter-particle distance on the unit sphere), which will be used to construct a discrete approximation to the inter-particle distance distribution.
    rgrid::Vector{Float64} = LinRange(0.0, 2.0, 5_000) ### 5_000 points between 0 and 2.

    # This is the bin-width, which will allow us to quickly compute the bin in which a given inter-particle distance falls.
    dr = rgrid[2] - rgrid[1]

    # We will now create a variable accumulated_pair_density, which will be used to accumulate the inter-particle distance distribution during the Monte Carlo simulation.
    accumulated_pair_density = zeros(Float64, length(rgrid)-1)
    
    # We will also create a variable current_distance_distribution, which will be used to store the inter-particle distance distribution for the current accepted state.
    current_distance_distribution = zeros(Float64, length(rgrid)-1)
    
    # We now update the current_distance_distribution according to the current state.
    for i in 1:system_size-1
        for j in i+1:system_size
            r = Ψcurrent.dist_matrix[j-1, i]
            current_distance_distribution[ceil(Int64, r/dr)] += 1.0
        end
    end

    # We will now create a grid of points between 0 and π, which will be used to construct a discrete approximation to the density of the CF Slater determinant state. Here, we create a grid uniform in cos(θ) instead of θ, as this creates equal-area bins.
    θmesh = map(x->acos(x), LinRange(1.0, -1.0, 500))

    # This is the area of each bin used in the density calculation.
    Agrid = 2.0 * pi .* (cos.(θmesh[begin:end-1]) .- cos.(θmesh[begin+1:end]))

    # We will now create a variable accumulated_density, which will be used to accumulate the density during the Monte Carlo simulation.
    accumulated_density = zeros(Float64, length(θmesh)-1)

    # We will now start our Monte Carlo simulation.
    t0 = time()    

    for monte_carlo_iter in 1:num_steps

        # The first step is to propose a new state using the Gaussian proposal distribution.
        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)

        # We then update the wavefunction.
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)
        
        # And calculate the log of the probability density of the proposed state.
        logpdf_next = logpdf(Ψnext)

        # We now accept or reject the proposed state according to the Metropolis-Hastings criterion.
        if logpdf_next - logpdf_current >= log(rand()) 

            # If the proposed state is accepted, we update the inter-particle distance distribution.
            for i in 1:system_size-1
                @inbounds current_distance_distribution[ceil(Int64, Ψnext.dist_matrix[i, sampling_iter] / dr)] += 1.0
                @inbounds current_distance_distribution[ceil(Int64, Ψcurrent.dist_matrix[i, sampling_iter] / dr)] -= 1.0
            end

            # We also set the current state to the proposed state.
            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]

            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next

            # And update the number of accepted samples.
            num_samples_accepted += 1

        else

            ## If the proposed state is rejected, we set the proposed state to the current state.
            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]

            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current

        end

        # We now update the accumulated_density and accumulated_pair_density.
        accumulated_pair_density .+= current_distance_distribution
        update_density!(θmesh, θcurrent, accumulated_density)

        # Finally, we advance the sampling_iter by 1.
        sampling_iter = mod(sampling_iter, system_size) + 1

        # Since, Monte Carlo simulations can take a long time, we will save the data to a file every 500,000 steps for backup or of course when the simulation is complete.
        if monte_carlo_iter == num_steps || mod(monte_carlo_iter, 5 * 10^5) == 0
            
            data["number of steps"] = monte_carlo_iter
            data["acceptance rate"] = num_samples_accepted/monte_carlo_iter
            data["monte carlo duration"] = time() - t0
            data["pair densities"] = accumulated_pair_density ./ monte_carlo_iter
            data["r grid"] = 0.50 .* (rgrid[1:end-1] .+ rgrid[2:end])
            data["density"] = accumulated_density ./ monte_carlo_iter ./ Agrid
            data["theta grid"] = 0.50 .* (θmesh[1:end-1] .+ θmesh[2:end])

            save(filename, data)
        
        end

    end

    return
end


function sample_cf_gs(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N//n-n)//2
    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(n*p+1)_filling_factor_$(chain_number)_chain_number.jld2")
    
    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

function sample_cf_qh(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N//n-n)//2
    Lqh = abs(Qstar) + abs(n) - 1

    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L if !(L == Lqh && Lz == Lqh)]

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(n*p+1)_filling_factor_$(chain_number)_chain_number.jld2")
    
    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

function sample_cf_qp(folder_name::String, chain_number::Int64, N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    Qstar = (N//n-n)//2
    Lqp = abs(Qstar) + abs(n)

    l_m_list = [(L, Lz) for L in abs(Qstar):1:(abs(Qstar)+abs(n)-1) for Lz in -L:1:L]
    push!(l_m_list, (Lqp, Lqp))

    filename = joinpath(folder_name, "data_$(N)_particles_$(n)_$(n*p+1)_filling_factor_$(chain_number)_chain_number.jld2")
    gibbs_sampler(filename, Qstar, l_m_list, p, num_thermalization, num_steps)

    return

end

