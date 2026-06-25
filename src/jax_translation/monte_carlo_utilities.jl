module MonteCarloOnSphere
using CoordinateTransformations
using Quaternionic
using StaticArrays
using LinearAlgebra
using Combinatorics
using Random

export rand_θ_ϕ_gen, proposal, arm_parameters, arm_scale_factor, construct_det_ratios, update_density!

"""
    rand_θ_ϕ_gen(RNG, n_samples::Int) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate random spherical coordinates (θ,ϕ) uniformly distributed on a unit sphere.

# Arguments
- `RNG`: Random number generator
- `n_samples::Int`: Number of random samples to generate

# Returns
- `θlist::Vector{Float64}`: Array of θ values in [0,π]
- `ϕlist::Vector{Float64}`: Array of ϕ values in (-π,π]

"""
function rand_θ_ϕ_gen(RNG, n_samples::Int)
    Xmat = randn(RNG, Float64, 3, n_samples)
    θlist = zeros(Float64, n_samples)
    ϕlist = zeros(Float64, n_samples)
    @simd for i in axes(Xmat, 2)
        # x = randn(RNG, Float64, 3)
        @inbounds @views sph = SphericalFromCartesian()(Xmat[:, i])
        θlist[i], ϕlist[i] = pi / 2 - sph.ϕ, sph.θ
    end
    return θlist, ϕlist
end


"""
    proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64) -> (θnew::Float64, ϕnew::Float64)

Generate a proposed new position on a sphere for a Monte Carlo step, given the current position (θcurrent, ϕcurrent).

The function generates a new position on the sphere using the following steps:
1. Creates a random displacement using a Gaussian step size (σ) and random direction
2. Represents this displacement as a quaternion from the north pole
3. Uses quaternion rotation to map the current position to the proposal position
4. Converts the result back to spherical coordinates

This method ensures uniform sampling across the sphere.

# Arguments
- `RNG`: Random number generator
- `θcurrent`: Current polar angle θ ∈ [0, π]
- `ϕcurrent`: Current azimuthal angle ϕ ∈ [-π, π]
- `σ`: Standard deviation of the Gaussian distribution for step size

# Returns
A tuple containing the new proposed position (θnew, ϕnew) on the sphere.

Note: The angles follow the mathematical physics convention where θ is the polar angle 
from the z-axis and ϕ is the azimuthal angle in the x-y plane.
"""
function proposal(RNG, θcurrent::Float64, ϕcurrent::Float64, σ::Float64)

    δθ = randn(RNG) * σ
    δϕ = rand(RNG) * (2.0 * pi) - pi

    sδθ, cδθ = sincos(δθ)
    sδϕ, cδϕ = sincos(δϕ)

    v = Quaternion(sδθ * cδϕ, sδθ * sδϕ, cδθ)

    sϕ, cϕ = sincos(ϕcurrent)

    q = exp(Quaternion(-sϕ, cϕ, 0.0) * θcurrent / 2)
    v = q * v * inv(q)
    x = SA[v[2], v[3], v[4]]

    sph = SphericalFromCartesian()(x)
    return pi / 2 - sph.ϕ, sph.θ
end

"""
Returns parameters for ARM scheme for step size adapation to maintain acceptance ratio.
"""
function arm_parameters(ideal_acceptance_ratio::Float64, r::Float64)
    a = 1.0
    b = 0.0
    for i in 1:1000
        c = (a * ideal_acceptance_ratio + b)^r
        a = (a * ideal_acceptance_ratio + b)^(1 / r) - c
        b = c
    end
    return a, b
end
"""
Returns ARM scale factor for a given acceptance ratio, the ideal acceptance ratio and ARM parameters.
"""
function arm_scale_factor(p, p_i, a, b)
    return log(a * p_i + b) / log(a * p + b)
end

"""
    construct_det_ratios(denominator_rows::Vector{Int64}, numerator_rows::Vector{Vector{Int64}})

Construct a function that efficiently computes multiple determinant ratios sharing a common denominator with each determinant constructed from the subset of rows in a matrix S.

# Arguments
- `denominator_rows::Vector{Int64}`: Vector containing the row indices for the denominator determinant
- `numerator_rows::Vector{Vector{Int64}}`: Vector of vectors, where each inner vector contains row indices for a numerator determinant

# Returns
- `helper!`: A function that takes three arguments:
    - `res`: Vector to store the results
    - `S`: The full matrix
    - `Sinv`: The inverse of the matrix S[denominator_rows, :]

# Details
The function creates an optimized routine for computing multiple determinant ratios of the form:
det(S[numerator_rows[i], :]) / det(S[denominator_rows, :])

It is assumed that the elements of the matrix S are of the type ComplexF64.

The implementation uses the fact that when rows in numerator and denominator overlap,
the ratio can be reduced to a smaller determinant calculation.

# Requirements
- Length of `denominator_rows` must equal the length of each vector in `numerator_rows`
- All vectors in `numerator_rows` must have the same length

"""
function construct_det_ratios(denominator_rows::Vector{Int64}, numerator_rows::Vector{Vector{Int64}})

    @assert length(denominator_rows) == length(numerator_rows[1]) && length(unique(length, numerator_rows)) == 1 "Lengths of denominator and numerator rows do not match."

    ### First, we need to identify all the common elements between the denominator and numerator rows.
    common_elements = copy(denominator_rows) ### Okay.

    for numerator_row in numerator_rows
        common_elements = intersect(common_elements, numerator_row)
    end

    denominator_diff = setdiff(denominator_rows, common_elements)

    numerator_diffs = [setdiff(row, common_elements) for row in numerator_rows]
    numerator_diffs_union = union(numerator_diffs...)

    iters = Matrix{Int64}(undef, length(denominator_diff), length(numerator_rows))
    for i in eachindex(numerator_rows)

        numerator_diff = numerator_diffs[i]

        for j in eachindex(denominator_diff)

            iters[j, i] = findfirst(isequal(numerator_diff[j]), numerator_diffs_union)

        end

    end

    function get_sign(rows1, rows2)

        p = [findfirst(isequal(elem), rows1) for elem in rows2]

        return levicivita(p)

    end

    denom_sign = get_sign(denominator_rows, vcat(denominator_diff, common_elements))
    numerators_signs = [get_sign(numerator_rows[iter], vcat(numerator_diffs[iter], common_elements)) for iter in eachindex(numerator_rows)]

    temp = zeros(ComplexF64, length(numerator_diffs_union), length(denominator_diff))

    intra_denom_diff = [findfirst(isequal(elem), denominator_rows) for elem in denominator_diff]

    function helper!(res, S, Sinv)

        @views @inbounds temp .= S[numerator_diffs_union, :] * Sinv[:, intra_denom_diff]

        @simd for i in eachindex(numerator_rows)

            @views @inbounds res[i] = det(temp[iters[:, i], :]) * numerators_signs[i] / denom_sign

        end

        return
    end

    return helper!

end

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
function update_density!(θmesh, θcurrent, accumulated_density)

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
function update_density!(θmesh, ϕmesh, θcurrent, ϕcurrent, accumulated_density)

    for iter in eachindex(θcurrent)
        accumulated_density[searchsortedfirst(θmesh, θcurrent[iter]) - 1, searchsortedfirst(ϕmesh, ϕcurrent[iter]) - 1] += 1.0
    end

    return

end
end
