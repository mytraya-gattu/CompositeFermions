module SpinPolarizedProjectedWavefunction
export Ψproj, ΨoneLL, Ψparton, update_wavefunction!, gibbs_thermalization!
include("symmetric_polynomials.jl")
using .SymmetricPolynomials
include("jk_projection_utilities.jl")
using .JKProjection

using CoordinateTransformations
using Quaternionic
using Random
using StaticArrays
using LinearAlgebra

function u_v_generator(θ, ϕ)

    return cos.(θ ./ 2) .* exp.(0.5im .* ϕ), sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)

end

"""
    Ψproj

    A mutable struct representing a Jain-Kamilla projected wavefunction (electrons + 2p vortices).

# Fields
- `Qstar::Rational{Int64}`: Effective monopole strength
- `p::Int64`: The number of flux quanta attached to each electron.
- `system_size::Int64`: Number of particles in the system

- `l_m_list::Vector{NTuple{2,Rational{Int64}}}`: List of (occupied) angular momentum quantum numbers (L, Lz)
- `Lmax::Rational{Int64}`: Maximum total angular momentum
- `μ_list::Vector{Rational{Int64}}`: List of angular momentum values for Wigner-d computation
- `Lz_list::Vector{Rational{Int64}}`: List of z-component angular momentum values

- `fourier_tot_matrix::Matrix{ComplexF64}`: Reshaped Fourier matrix for efficient multiplication
- `U::Vector{ComplexF64}`: U coordinates on sphere (cos(θ/2) * exp(iϕ/2))
- `V::Vector{ComplexF64}`: V coordinates on sphere (sin(θ/2) * exp(-iϕ/2))

- `exp_θ::Matrix{ComplexF64}`: Matrix of exponentials of theta angles
- `exp_ϕ::Matrix{ComplexF64}`: Matrix of exponentials of phi angles

- `dist_matrix::Matrix{Float64}`: Matrix of distances between particles (on the unit sphere)
- `u_v_ratio_matrix::Matrix{ComplexF64}`: Matrix of U/V ratios (for Jain-Kamilla projection)

- `elementary_symmetric_polynomials::Matrix{ComplexF64}`: Matrix of elementary symmetric polynomials (for Jain-Kamilla projection)
- `reg_coeffs::Vector{Float64}`: Regularization coefficients (for Jain-Kamilla projection)

- `wigner_d_matrices::Matrix{ComplexF64}`: Small Wigner d-matrices (for Jain-Kamilla projection)
- `wigner_D_matrices::Array{ComplexF64,3}`: Large Wigner D-matrices (for Jain-Kamilla projection)

- `jastrow_factor_log::ComplexF64}`: Logarithm of the Jastrow factor
- `slater_det::Matrix{ComplexF64}`: Slater determinant matrix

"""
mutable struct Ψproj

    Qstar::Rational{Int64}
    p::Int64
    system_size::Int64

    l_m_list::Vector{NTuple{2, Rational{Int64}}} ### This is a list of tuples of the form (L, Lz)
    Lmax::Rational{Int64}
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64} ### Reshaped fourier matrix for efficient matrix multiplication.

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}
    wigner_D_matrices::Array{ComplexF64, 3}

    jastrow_factor_log::ComplexF64
    slater_det::Matrix{ComplexF64}
end

"""
    Ψparton

    A mutable struct representing a Jain-Kamilla projected parton wavefunction.

# Fields
- `Qstars::Vector{Rational{Int64}}`: Effective monopole strengths of the partons.
- `p::Int64`: Number of Φ₁ partons.
- `system_size::Int64`: Number of particles in the system.

- `l_m_lists::Vector{Vector{NTuple{2,Rational{Int64}}}}`: Lists of (occupied) angular momentum quantum numbers (L, Lz) for each parton.
- `max_JK::Int64`: Maximum order of JK projection required.
- `μ_list::Vector{Rational{Int64}}`: List of angular momentum values for Wigner-d computation
- `Lz_list::Vector{Rational{Int64}}`: List of z-component angular momentum values

- `fourier_tot_matrix::Matrix{ComplexF64}`: Reshaped Fourier matrix for efficient multiplication
- `U::Vector{ComplexF64}`: U coordinates on sphere (cos(θ/2) * exp(iϕ/2))
- `V::Vector{ComplexF64}`: V coordinates on sphere (sin(θ/2) * exp(-iϕ/2))

- `exp_θ::Matrix{ComplexF64}`: Matrix of exponentials of theta angles
- `exp_ϕ::Matrix{ComplexF64}`: Matrix of exponentials of phi angles

- `dist_matrix::Matrix{Float64}`: Matrix of distances between particles (on the unit sphere)
- `u_v_ratio_matrix::Matrix{ComplexF64}`: Matrix of U/V ratios (for Jain-Kamilla projection)

- `elementary_symmetric_polynomials::Matrix{ComplexF64}`: Matrix of elementary symmetric polynomials (for Jain-Kamilla projection)
- `reg_coeffs::Vector{Float64}`: Regularization coefficients (for Jain-Kamilla projection)

- `wigner_d_matrices::Matrix{ComplexF64}`: Small Wigner d-matrices (for Jain-Kamilla projection)
- `wigner_D_matrices::Array{ComplexF64,3}`: Large Wigner D-matrices (for Jain-Kamilla projection)

- `jastrow_factor_log::ComplexF64}`: Logarithm of the Jastrow factor
- `slater_det::Matrix{ComplexF64}`: Slater determinant matrix
- `trackers::Vector{NTuple{2, Int64}}` : The ith parton is represented by the rows trackers[i][1]:trackers[i][2] in slater_det.

"""
mutable struct Ψparton

    Qstars::Vector{Rational{Int64}}
    p::Int64 ### number of ϕ1s..
    system_size::Int64

    l_m_lists::Vector{Vector{NTuple{2, Rational{Int64}}}} ### This is a list of tuples of the form (L, Lz)

    max_JK::Int64
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64} ### Reshaped fourier matrix for efficient matrix multiplication.

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}
    wigner_D_matrices::Array{ComplexF64, 3}

    jastrow_factor_log::ComplexF64
    slater_det::Matrix{ComplexF64} ### For ease, I will keep this as a 3D array. Basically append 0s at the end if there aren't enough rows.
    trackers::Vector{UnitRange{Int64}}

end

"""
    Ψproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2,Rational{Int64}}})

Constructor function for Jain-Kamilla projected wavefunction on a sphere.

# Arguments
- `Qstar::Rational{Int64}`: Effective monopole strength = Q* (= Q - p(N-1)/2, in case of composite fermions)
- `p::Int64`: The number of vortices bound to each electron
- `system_size::Int64`: Number of electrons in the system
- `l_m_list::Vector{NTuple{2,Rational{Int64}}}`: List of angular momentum quantum numbers (l,m) representing 
    the occupied quasi-Landau levels

    Returns a `Ψproj` type object containing all necessary arrays and matrices for wavefunction calculations.
"""
function Ψproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2, Rational{Int64}}})

    Lmax = maximum(first, l_m_list)

    fourier_matrix = zeros(ComplexF64, length(l_m_list), numerator(1 + Lmax - Qstar), numerator(1 + 2 * Lmax))

    Lgrid = unique(first.(l_m_list))
    liters = [findall(x -> x[1] == L, l_m_list) for L in Lgrid]

    for (Liter, L) in enumerate(Lgrid)

        fourier_matrix[liters[Liter], begin:(begin + numerator(L - Qstar)), numerator(1 - L + Lmax):1:numerator(1 + L + Lmax)] .= generate_fourier_matrices(Qstar, system_size, L, last.(l_m_list[liters[Liter]]))

    end

    fourier_tot_matrix = reshape(fourier_matrix, :, numerator(1 + 2 * Lmax))

    Lz_list = last.(l_m_list)

    μ_list = collect(-Lmax:1:Lmax)

    U = zeros(ComplexF64, system_size)
    V = zeros(ComplexF64, system_size)

    exp_θ = zeros(ComplexF64, length(μ_list), system_size)
    exp_ϕ = zeros(ComplexF64, length(l_m_list), system_size)

    jastrow_factor_log = 0.0 + 0.0im

    slater_det = zeros(ComplexF64, length(l_m_list), system_size)
    dist_matrix = zeros(Float64, system_size - 1, system_size)

    u_v_ratio_matrix = zeros(ComplexF64, system_size - 1, system_size)
    elementary_symmetric_polynomials = zeros(ComplexF64, 1 + round(Int64, Lmax - Qstar), system_size)

    wigner_d_matrices = zeros(ComplexF64, length(l_m_list) * (1 + numerator(Lmax - Qstar)), system_size)
    wigner_D_matrices = zeros(ComplexF64, length(l_m_list), 1 + numerator(Lmax - Qstar), system_size)

    reg_coeffs = zeros(Float64, round(Int64, Lmax - Qstar))

    for i in eachindex(reg_coeffs)
        reg_coeffs[i] = (i / (system_size - i))
    end

    return Ψproj(Qstar, p, system_size, l_m_list, Lmax, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, u_v_ratio_matrix, elementary_symmetric_polynomials, reg_coeffs, wigner_d_matrices, wigner_D_matrices, jastrow_factor_log, slater_det)
end


"""
    Ψproj(Qstars::Vector{Rational{Int64}}, p::Int64, system_size::Int64, l_m_lists::Vector{Vector{NTuple{2,Rational{Int64}}}})

Constructor function for Jain-Kamilla projected parton wavefunction on a sphere.

# Arguments
- `Qstars::Vector{Rational{Int64}}`: Effective monopole strengths of the partons.
- `p::Int64`: The number of Φ1 partons.
- `system_size::Int64`: Number of electrons in the system
- `l_m_lists::Vector{Vector{NTuple{2,Rational{Int64}}}}`: List of angular momentum quantum numbers (l,m) representing 
    the occupied quasi-Landau levels in each parton.

    Returns a `Ψparton` type object containing all necessary arrays and matrices for wavefunction calculations.
"""
function Ψparton(Qstars::Vector{Rational{Int64}}, p::Int64, system_size::Int64, l_m_lists::Vector{Vector{NTuple{2, Rational{Int64}}}})

    @assert length(Qstars) == length(l_m_lists) "Number of Qstars and l_m_lists should be the same."
    max_JK = typemin(Int64)

    Lmaxs = Vector{Rational{Int64}}()

    for iter in eachindex(Qstars)

        Lmax = maximum(first, l_m_lists[iter])
        push!(Lmaxs, Lmax)
        max_JK = max(max_JK, numerator(Lmax - Qstars[iter]))

    end

    μ_list = sort(unique(vcat([collect(-Lmax:1:Lmax) for Lmax in Lmaxs]...)))

    fourier_matrix = zeros(ComplexF64, sum(length, l_m_lists), max_JK + 1, length(μ_list))

    tracker = 0
    trackers = Vector{UnitRange{Int64}}()
    for iter in eachindex(l_m_lists)

        l_m_list = l_m_lists[iter]
        Qstar = Qstars[iter]

        Lgrid = unique(first.(l_m_list))
        liters = [findall(x -> x[1] == L, l_m_list) for L in Lgrid]

        for (Liter, L) in enumerate(Lgrid)

            θ_iters = [findfirst(isequal(μ), μ_list) for μ in -L:1:L]

            fourier_matrix[tracker .+ liters[Liter], begin:(begin + numerator(L - Qstar)), θ_iters] .= generate_fourier_matrices(Qstar, system_size, L, last.(l_m_list[liters[Liter]]))

        end

        push!(trackers, (1 + tracker):(tracker + length(l_m_list)))

        tracker += length(l_m_list)
    end

    fourier_tot_matrix = reshape(fourier_matrix, :, length(μ_list))
    Lz_list = vcat([last.(l_m_list) for l_m_list in l_m_lists]...)

    U = zeros(ComplexF64, system_size)
    V = zeros(ComplexF64, system_size)

    exp_θ = zeros(ComplexF64, length(μ_list), system_size)
    exp_ϕ = zeros(ComplexF64, length(Lz_list), system_size)

    jastrow_factor_log = 0.0 + 0.0im

    slater_det = zeros(ComplexF64, sum(length, l_m_lists), system_size)

    dist_matrix = zeros(Float64, system_size - 1, system_size)

    u_v_ratio_matrix = zeros(ComplexF64, system_size - 1, system_size)
    elementary_symmetric_polynomials = zeros(ComplexF64, 1 + max_JK, system_size)

    wigner_d_matrices = zeros(ComplexF64, sum(length, l_m_lists) * (1 + max_JK), system_size)
    wigner_D_matrices = zeros(ComplexF64, sum(length, l_m_lists), (1 + max_JK), system_size)

    reg_coeffs = zeros(Float64, max_JK)

    for i in eachindex(reg_coeffs)
        reg_coeffs[i] = (i / (system_size - i))
    end

    return Ψparton(Qstars, p, system_size, l_m_lists, max_JK, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, u_v_ratio_matrix, elementary_symmetric_polynomials, reg_coeffs, wigner_d_matrices, wigner_D_matrices, jastrow_factor_log, slater_det, trackers)
end


"""
    update_wavefunction!(Ψ::Ψproj, θ::Vector{Float64}, ϕ::Vector{Float64})

Updates the many-body wavefunction Ψ given new particle positions specified by spherical coordinates (θ,ϕ).

# Arguments
- `Ψ::Ψproj`: The projected wavefunction object to be updated
- `θ::Vector{Float64}`: Vector of polar angles θ for each particle
- `ϕ::Vector{Float64}`: Vector of azimuthal angles ϕ for each particle

# Returns
- `nothing`:, modifies the input wavefunction object in-place.
"""
function update_wavefunction!(Ψ::Ψproj, θ::Vector{Float64}, ϕ::Vector{Float64})

    Ψ.exp_θ .= exp.(-1.0im .* Ψ.μ_list * transpose(θ))
    Ψ.exp_ϕ .= exp.(1.0im .* Ψ.Lz_list * transpose(ϕ))

    Ψ.jastrow_factor_log = zero(ComplexF64)

    Ψ.U, Ψ.V = u_v_generator(θ, ϕ)

    δu = zero(ComplexF64)
    δv = zero(ComplexF64)

    for i in 1:(Ψ.system_size - 1)
        for j in (i + 1):Ψ.system_size

            δu = conj(Ψ.U[i]) * Ψ.U[j] + conj(Ψ.V[i]) * Ψ.V[j]
            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.u_v_ratio_matrix[j - 1, i] = δu / δv
            Ψ.u_v_ratio_matrix[i, j] = -conj(δu) / δv

            Ψ.jastrow_factor_log += Ψ.p * log(δv)

            Ψ.dist_matrix[j - 1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j - 1, i]

        end
    end

    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)

        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], numerator(Ψ.Lmax - Ψ.Qstar), Ψ.reg_coeffs)

    end

    Ψ.wigner_d_matrices .= Ψ.fourier_tot_matrix * Ψ.exp_θ
    Ψ.wigner_D_matrices .= reshape(Ψ.wigner_d_matrices, size(Ψ.wigner_D_matrices))

    @simd for iter in axes(Ψ.wigner_D_matrices, 3)

        @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    end

    @simd for iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, iter] .= Ψ.wigner_D_matrices[:, :, iter] * Ψ.elementary_symmetric_polynomials[:, iter]

    end

    return
end

"""
    update_wavefunction!(Ψ::Ψproj, θ::Vector{Float64}, ϕ::Vector{Float64})

Updates the many-body wavefunction Ψ given new particle positions specified by spherical coordinates (θ,ϕ).

# Arguments
- `Ψ::Ψparton`: The parton wavefunction object to be updated.
- `θ::Vector{Float64}`: Vector of polar angles θ for each particle.
- `ϕ::Vector{Float64}`: Vector of azimuthal angles ϕ for each particle.

# Returns
- `nothing`:, modifies the input wavefunction object in-place.
"""
function update_wavefunction!(Ψ::Ψparton, θ::Vector{Float64}, ϕ::Vector{Float64})

    Ψ.exp_θ .= exp.(-1.0im .* Ψ.μ_list * transpose(θ))
    Ψ.exp_ϕ .= exp.(1.0im .* Ψ.Lz_list * transpose(ϕ))

    Ψ.jastrow_factor_log = zero(ComplexF64)

    Ψ.U, Ψ.V = u_v_generator(θ, ϕ)

    δu = zero(ComplexF64)
    δv = zero(ComplexF64)

    for i in 1:(Ψ.system_size - 1)
        for j in (i + 1):Ψ.system_size

            δu = conj(Ψ.U[i]) * Ψ.U[j] + conj(Ψ.V[i]) * Ψ.V[j]
            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.u_v_ratio_matrix[j - 1, i] = δu / δv
            Ψ.u_v_ratio_matrix[i, j] = -conj(δu) / δv

            Ψ.jastrow_factor_log += Ψ.p * log(δv)

            Ψ.dist_matrix[j - 1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j - 1, i]

        end
    end

    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)

        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], Ψ.max_JK, Ψ.reg_coeffs) ### Max order of JK projection required.

    end

    Ψ.wigner_d_matrices .= Ψ.fourier_tot_matrix * Ψ.exp_θ
    Ψ.wigner_D_matrices .= reshape(Ψ.wigner_d_matrices, size(Ψ.wigner_D_matrices))

    @simd for iter in axes(Ψ.wigner_D_matrices, 3)

        @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    end ### This only makes sense because I can move a particle one by by one. There, this will be more efficient.

    @simd for iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, iter] .= Ψ.wigner_D_matrices[:, :, iter] * Ψ.elementary_symmetric_polynomials[:, iter]

    end

    return
end


"""
    update_wavefunction!(Ψ::Ψproj, θ::Float64, ϕ::Float64, iter::Int64)

Updates the quantum many-body wavefunction by moving the `iter`-th particle to a new position 
specified by spherical coordinates (θ,ϕ).

# Arguments
- `Ψ::Ψproj`: The projected wavefunction state object to be updated
- `θ::Float64`: Polar angle (theta) of the new position in radians
- `ϕ::Float64`: Azimuthal angle (phi) of the new position in radians  
- `iter::Int64`: Index of the particle being moved

# Returns
- `nothing`:, modifies the input wavefunction object in-place.
"""
function update_wavefunction!(Ψ::Ψproj, θ::Float64, ϕ::Float64, iter::Int64)

    Ψ.exp_θ[:, iter] .= exp.(-1.0im .* Ψ.μ_list * θ)
    Ψ.exp_ϕ[:, iter] .= exp.(1.0im .* Ψ.Lz_list * ϕ)

    unew, vnew = u_v_generator(θ, ϕ)

    δv_old = zero(ComplexF64)
    δv_new = zero(ComplexF64)

    δu_new = zero(ComplexF64)

    jk_order = numerator(Ψ.Lmax - Ψ.Qstar)
    use_incremental = 2 * jk_order < Ψ.system_size  # stable when reg_coeffs[k] = k/(N-k) < 1

    for i in 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew - Ψ.V[i] * unew

            δu_new = conj(Ψ.U[i]) * unew + conj(Ψ.V[i]) * vnew

            if use_incremental
                r_old = Ψ.u_v_ratio_matrix[iter - 1, i]
            end

            Ψ.u_v_ratio_matrix[iter - 1, i] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[i, iter] = -conj(δu_new) / δv_new

            if use_incremental
                @inbounds @views update_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, i], r_old, Ψ.u_v_ratio_matrix[iter - 1, i], jk_order, Ψ.reg_coeffs)
            end

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter - 1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter - 1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew + Ψ.V[i] * unew

            δu_new = (Ψ.U[i]) * conj(unew) + (Ψ.V[i]) * conj(vnew)

            if use_incremental
                r_old = Ψ.u_v_ratio_matrix[iter, i]
            end

            Ψ.u_v_ratio_matrix[i - 1, iter] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[iter, i] = -conj(δu_new) / δv_new

            if use_incremental
                @inbounds @views update_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, i], r_old, Ψ.u_v_ratio_matrix[iter, i], jk_order, Ψ.reg_coeffs)
            end

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i - 1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i - 1, iter]

        end

    end

    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    if use_incremental
        # Only column iter needs full recompute (all its roots changed)
        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, iter], Ψ.u_v_ratio_matrix[:, iter], jk_order, Ψ.reg_coeffs)
    else
        # Full recompute for all columns (numerically safe for large jk_order)
        @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)
            @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], jk_order, Ψ.reg_coeffs)
        end
    end

    Ψ.wigner_d_matrices[:, iter] .= Ψ.fourier_tot_matrix * Ψ.exp_θ[:, iter]

    @simd for j in axes(Ψ.wigner_D_matrices, 2)
        @inbounds @views Ψ.wigner_D_matrices[:, j, iter] .= Ψ.wigner_d_matrices[(1 + size(Ψ.wigner_D_matrices, 1) * (j - 1)):(size(Ψ.wigner_D_matrices, 1) * (j)), iter]
    end

    Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    @simd for electron_iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, electron_iter] .= Ψ.wigner_D_matrices[:, :, electron_iter] * Ψ.elementary_symmetric_polynomials[:, electron_iter]

    end

    return
end

"""
    update_wavefunction!(Ψ::Ψparton, θ::Float64, ϕ::Float64, iter::Int64)

Updates the quantum many-body wavefunction by moving the `iter`-th particle to a new position 
specified by spherical coordinates (θ,ϕ).

# Arguments
- `Ψ::Ψparton`: The parton wavefunction state object to be updated
- `θ::Float64`: Polar angle (theta) of the new position in radians
- `ϕ::Float64`: Azimuthal angle (phi) of the new position in radians  
- `iter::Int64`: Index of the particle being moved

# Returns
- `nothing`:, modifies the input wavefunction object in-place.
"""
function update_wavefunction!(Ψ::Ψparton, θ::Float64, ϕ::Float64, iter::Int64)

    Ψ.exp_θ[:, iter] .= exp.(-1.0im .* Ψ.μ_list * θ)
    Ψ.exp_ϕ[:, iter] .= exp.(1.0im .* Ψ.Lz_list * ϕ)

    unew, vnew = u_v_generator(θ, ϕ)

    δv_old = zero(ComplexF64)
    δv_new = zero(ComplexF64)

    δu_new = zero(ComplexF64)

    use_incremental = 2 * Ψ.max_JK < Ψ.system_size  # stable when reg_coeffs[k] = k/(N-k) < 1

    for i in 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew - Ψ.V[i] * unew

            δu_new = conj(Ψ.U[i]) * unew + conj(Ψ.V[i]) * vnew

            if use_incremental
                r_old = Ψ.u_v_ratio_matrix[iter - 1, i]
            end

            Ψ.u_v_ratio_matrix[iter - 1, i] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[i, iter] = -conj(δu_new) / δv_new

            if use_incremental
                @inbounds @views update_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, i], r_old, Ψ.u_v_ratio_matrix[iter - 1, i], Ψ.max_JK, Ψ.reg_coeffs)
            end

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter - 1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter - 1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew + Ψ.V[i] * unew

            δu_new = (Ψ.U[i]) * conj(unew) + (Ψ.V[i]) * conj(vnew)

            if use_incremental
                r_old = Ψ.u_v_ratio_matrix[iter, i]
            end

            Ψ.u_v_ratio_matrix[i - 1, iter] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[iter, i] = -conj(δu_new) / δv_new

            if use_incremental
                @inbounds @views update_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, i], r_old, Ψ.u_v_ratio_matrix[iter, i], Ψ.max_JK, Ψ.reg_coeffs)
            end

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i - 1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i - 1, iter]

        end

    end

    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    if use_incremental
        # Only column iter needs full recompute (all its roots changed)
        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, iter], Ψ.u_v_ratio_matrix[:, iter], Ψ.max_JK, Ψ.reg_coeffs)
    else
        # Full recompute for all columns (numerically safe for large jk_order)
        @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)
            @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], Ψ.max_JK, Ψ.reg_coeffs)
        end
    end

    Ψ.wigner_d_matrices[:, iter] .= Ψ.fourier_tot_matrix * Ψ.exp_θ[:, iter]

    @simd for j in axes(Ψ.wigner_D_matrices, 2)
        @inbounds @views Ψ.wigner_D_matrices[:, j, iter] .= Ψ.wigner_d_matrices[(1 + size(Ψ.wigner_D_matrices, 1) * (j - 1)):(size(Ψ.wigner_D_matrices, 1) * (j)), iter]
    end ### This is because I cannot allocate and reshape at the same time.

    Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    @simd for electron_iter in axes(Ψ.slater_det, 2)

        @inbounds @views Ψ.slater_det[:, electron_iter] .= Ψ.wigner_D_matrices[:, :, electron_iter] * Ψ.elementary_symmetric_polynomials[:, electron_iter]

    end

    return
end


"""
    copy!(Ψ1::Ψproj, Ψ2::Ψproj)

In-place copy of a projected wavefunction `Ψ2` into `Ψ1`.

Copies all components of the projected wavefunction including:
- Distance matrix
- Angular components (exp_θ, exp_ϕ)
- U and V matrices
- Jastrow factor (logarithmic form)
- Slater determinant
- Elementary symmetric polynomials
- U/V ratio matrix
- Wigner d and D matrices

# Arguments
- `Ψ1::Ψproj`: Destination projected wavefunction
- `Ψ2::Ψproj`: Source projected wavefunction

# Returns
Nothing, modifies `Ψ1` in-place.
"""
function Base.copy!(Ψ1::Ψproj, Ψ2::Ψproj)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ .= Ψ2.exp_θ
    Ψ1.exp_ϕ .= Ψ2.exp_ϕ
    Ψ1.U .= Ψ2.U
    Ψ1.V .= Ψ2.V

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials
    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices .= Ψ2.wigner_d_matrices
    Ψ1.wigner_D_matrices .= Ψ2.wigner_D_matrices

    return
end

function Base.copy!(Ψ1::Ψparton, Ψ2::Ψparton)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ .= Ψ2.exp_θ
    Ψ1.exp_ϕ .= Ψ2.exp_ϕ
    Ψ1.U .= Ψ2.U
    Ψ1.V .= Ψ2.V

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials
    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices .= Ψ2.wigner_d_matrices
    Ψ1.wigner_D_matrices .= Ψ2.wigner_D_matrices

    return
end

"""
    copy!(Ψ1::Ψproj, Ψ2::Ψproj, iter::Int64)

Copy the state of projected wavefunction `Ψ2` into `Ψ1` assuming only the `iter`-th particle has been moved.

# Arguments
- `Ψ1::Ψproj`: Destination projected wavefunction
- `Ψ2::Ψproj`: Source projected wavefunction
- `iter::Int64`: Index of the particle that has been moved

# Returns
- `nothing`, modifies `Ψ1` in-place.

# Note
This method extends Base.copy! for Ψproj type objects.
"""
function Base.copy!(Ψ1::Ψproj, Ψ2::Ψproj, iter::Int64)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ[:, iter] .= Ψ2.exp_θ[:, iter]
    Ψ1.exp_ϕ[:, iter] .= Ψ2.exp_ϕ[:, iter]
    Ψ1.U[iter] = Ψ2.U[iter]
    Ψ1.V[iter] = Ψ2.V[iter]

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials

    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices[:, iter] .= Ψ2.wigner_d_matrices[:, iter]
    Ψ1.wigner_D_matrices[:, :, iter] .= Ψ2.wigner_D_matrices[:, :, iter]

    return
end

function Base.copy!(Ψ1::Ψparton, Ψ2::Ψparton, iter::Int64)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.exp_θ[:, iter] .= Ψ2.exp_θ[:, iter]
    Ψ1.exp_ϕ[:, iter] .= Ψ2.exp_ϕ[:, iter]
    Ψ1.U[iter] = Ψ2.U[iter]
    Ψ1.V[iter] = Ψ2.V[iter]

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.slater_det .= Ψ2.slater_det

    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials

    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix

    Ψ1.wigner_d_matrices[:, iter] .= Ψ2.wigner_d_matrices[:, iter]
    Ψ1.wigner_D_matrices[:, :, iter] .= Ψ2.wigner_D_matrices[:, :, iter]

    return
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

mutable struct ΨoneLL

    p::Int64
    system_size::Int64

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    dist_matrix::Matrix{Float64}

    jastrow_factor_log::ComplexF64
end


function ΨoneLL(p::Int64, system_size::Int64)

    U = zeros(ComplexF64, system_size)
    V = zeros(ComplexF64, system_size)

    jastrow_factor_log = 0.0 + 0.0im

    dist_matrix = zeros(Float64, system_size - 1, system_size)

    return ΨoneLL(p, system_size, U, V, dist_matrix, jastrow_factor_log)
end


function update_wavefunction!(Ψ::ΨoneLL, θ::Vector{Float64}, ϕ::Vector{Float64})

    Ψ.jastrow_factor_log = zero(ComplexF64)

    Ψ.U, Ψ.V = u_v_generator(θ, ϕ)

    δu = zero(ComplexF64)
    δv = zero(ComplexF64)

    for i in 1:(Ψ.system_size - 1)
        for j in (i + 1):Ψ.system_size

            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.jastrow_factor_log += Ψ.p * log(δv)

            Ψ.dist_matrix[j - 1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j - 1, i]

        end
    end

    return
end

function update_wavefunction!(Ψ::ΨoneLL, θ::Float64, ϕ::Float64, iter::Int64)

    unew, vnew = u_v_generator(θ, ϕ)

    δv_old = zero(ComplexF64)
    δv_new = zero(ComplexF64)

    for i in 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew - Ψ.V[i] * unew

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter - 1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter - 1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew + Ψ.V[i] * unew

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i - 1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i - 1, iter]

        end

    end

    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    return
end

function Base.copy!(Ψ1::ΨoneLL, Ψ2::ΨoneLL)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.U .= Ψ2.U
    Ψ1.V .= Ψ2.V

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log

    return
end

function Base.copy!(Ψ1::ΨoneLL, Ψ2::ΨoneLL, iter::Int64)

    Ψ1.dist_matrix .= Ψ2.dist_matrix

    Ψ1.U[iter] = Ψ2.U[iter]
    Ψ1.V[iter] = Ψ2.V[iter]

    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log

    return
end

"""
    gibbs_thermalization!(RNG, Ψcurrent::Ψproj, Ψnext::Ψproj, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, θnext::Vector{Float64}, ϕnext::Vector{Float64}, σinit::Float64, logpdf::Function, num_thermalization::Int64)
    
    Performs thermalization for gibbs sampling with respect to composite fermion (CF) wavefunction on the sphere.

# Arguments
- `RNG`: Random number generator
- `Ψcurrent::Ψproj`: Current state of the CF wavefunction
- `Ψnext::Ψproj`: Next state of the CF wavefunction
- `θcurrent::Vector{Float64}`: Current theta angles
- `ϕcurrent::Vector{Float64}`: Current phi angles
- `θnext::Vector{Float64}`: Next theta angles
- `ϕnext::Vector{Float64}`: Next phi angles
- `σinit::Float64`: Initial step size for the proposal distribution
- `logpdf::Function`: Log probability density function to sample from. 
    The function should take a `Ψproj` object as input and return a real scalar value.
- `num_thermalization::Int64`: Number of thermalization steps

# Returns
- `sampling_iter::Int64`: Final iteration index
- `σ::Float64`: Final step size of the proposal distribution
- `δt_therm::Float64`: Total thermalization time
- `acceptance_rate::Float64`: Acceptance rate

# Description
Implements a Gibbs sampling algorithm with adaptive step size for thermalization
of a composite fermion wavefunction. Uses Metropolis-Hastings acceptance criterion and 
updates one particle position at a time. The step size is tuned during 
thermalization to achieve a target acceptance rate of 50%.
"""
function gibbs_thermalization!(RNG, Ψcurrent::T, Ψnext::T, θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64}, θnext::Vector{Float64}, ϕnext::Vector{Float64}, σinit::Float64, logpdf::Function, num_thermalization::Int64) where {T <: Union{Ψproj, Ψparton, ΨoneLL}}

    acceptance_target::Float64 = 0.5 ### Gibbs sampling.
    a::Float64, b::Float64 = arm_parameters(acceptance_target, 3.0)

    num_samples_accepted_thermalization::Int64 = 0
    δ::Float64 = 1.0
    σ::Float64 = σinit

    logpdf_current::Float64 = 0.0
    logpdf_next::Float64 = 0.0

    update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
    copy!(Ψnext, Ψcurrent)

    logpdf_current = logpdf(Ψcurrent)

    tuning_schedule::Vector{Int64} = round.(Int64, exp.(LinRange(log(10.0), log(num_thermalization), 25)))

    sampling_iter::Int64 = 1
    t0::Float64 = time()
    for monte_carlo_iter in 1:num_thermalization

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        logpdf_next = logpdf(Ψnext)

        if logpdf_next - logpdf_current >= log(rand())

            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]

            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next

            num_samples_accepted_thermalization += 1

        else

            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]

            copy!(Ψnext, Ψcurrent, sampling_iter)
            logpdf_next = logpdf_current

        end

        if monte_carlo_iter ∈ tuning_schedule

            δ = arm_scale_factor(num_samples_accepted_thermalization / monte_carlo_iter, acceptance_target, a, b)
            σ *= δ
        end

        sampling_iter = mod(sampling_iter, Ψcurrent.system_size) + 1

    end

    δt_therm::Float64 = time() - t0
    return sampling_iter, σ, δt_therm, num_samples_accepted_thermalization / num_thermalization
end

end
