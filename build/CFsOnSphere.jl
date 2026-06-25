module CFsOnSphere

include("projected_wavefunction.jl")
using .SpinPolarizedProjectedWavefunction

include("monte_carlo_utilities.jl")
using .MonteCarloOnSphere

include("legendre_polynomials.jl")
using .LegendrePolynomials

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using JLD2

export Ψproj, update_wavefunction!, gibbs_thermalization!, rand_θ_ϕ_gen, proposal, legendre_polynomials!, save, load, logdet, lu, inv, update_density!, construct_det_ratios

"""
    update_density!(θmesh::Vector{Float64}, θcurrent::Vector{Float64}, accumulated_density::Vector{Float64})

Update the accumulated polar density histogram based on current particle positions.

# Arguments
- `θmesh::Vector{Float64}`: Vector containing the bin edges for the polar density histogram
- `θcurrent::Vector{Float64}`: Vector containing current polar positions of particles
- `accumulated_density::Vector{Float64}`: Vector storing the accumulated density counts

# Description
For each position θ in θcurrent, increments the count in the corresponding bin of 
accumulated_density by 1.0. The bin is determined by finding the first index in θmesh 
that is greater than θ and subtracting 1.

The function modifies accumulated_density in-place and returns nothing.

# Note
Assumes θmesh is sorted in ascending order for correct bin assignment using searchsortedfirst.
"""
function update_density!(θmesh::Vector{Float64}, θcurrent::Vector{Float64}, accumulated_density::Vector{Float64})

    for θ in θcurrent
        accumulated_density[searchsortedfirst(θmesh, θ) - 1] += 1.0
    end

    return

end

"""
    update_density!(θmesh::Vector{Float64}, ϕmesh::Vector{Float64}, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, accumulated_density::Matrix{Float64})

Updates a 2D density matrix based on the current positions of particles in spherical coordinates.

# Arguments
- `θmesh::Vector{Float64}`: Vector of theta (polar angle) mesh points
- `ϕmesh::Vector{Float64}`: Vector of phi (azimuthal angle) mesh points
- `θcurrent::Vector{Float64}`: Current theta positions of particles
- `ϕcurrent::Vector{Float64}`: Current phi positions of particles
- `accumulated_density::Matrix{Float64}`: 2D matrix storing the accumulated density values

# Description
For each particle position (θ, ϕ), increments the corresponding bin in the accumulated_density matrix by 1.0.
The bin indices are determined using searchsortedfirst on the mesh vectors.

# Notes
- Assumes the mesh vectors are sorted
- The function modifies the accumulated_density matrix in-place
"""
function update_density!(θmesh::Vector{Float64}, ϕmesh::Vector{Float64}, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, accumulated_density::Matrix{Float64})

    for iter in eachindex(θcurrent)
        accumulated_density[searchsortedfirst(θmesh, θcurrent[iter]) - 1, searchsortedfirst(ϕmesh, ϕcurrent[iter]) - 1] += 1.0
    end

    return

end

end
