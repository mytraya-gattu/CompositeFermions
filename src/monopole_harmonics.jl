# Single-particle monopole harmonics (the unprojected orbital base functions).

"""
    calculate_ll(l::Rational{Int}, Q::Rational{Int}, θ::Float64, φ::Float64) -> Vector{ComplexF64}

Evaluate the monopole harmonics ``Y_{Q,l,m}(θ, φ)`` of total angular momentum `l` at
monopole strength `Q`, for all `m = -l, -l+1, …, l`. The returned vector is ordered by
increasing `m`.

These are the single-particle orbitals making up the *unprojected* composite-fermion
Slater determinant (`Ψunproj`), and are also used to build the fixed quasihole /
quasiparticle orbital columns consumed by [`build_extended_slater!`](@ref).

The orbital is assembled from the `J_y` eigenstates (see [`calculate_j_y_eigenstates`](@ref))
as a basis rotation,

    Y_{Q,l,m}(θ,φ) = √((2l+1)/4π) · Σ_μ ⟨l,μ|_y |l,Q⟩ ⟨l,m|l,μ⟩_y e^{-iμθ} e^{imφ},

which avoids the (discontinuous) phase of the explicit monopole-harmonic formula.
"""
function calculate_ll(l::Rational{Int}, Q::Rational{Int}, θ::Float64, φ::Float64)
    jy_eigs = calculate_j_y_eigenstates(l)   # jy_eigs[(μ, m1, m2)]

    m_values = -l:1:l
    μ_values = -l:1:l
    num_states = length(m_values)

    fourier_mat = zeros(ComplexF64, num_states, num_states)
    for (i, m) in enumerate(m_values)
        for (j, μ) in enumerate(μ_values)
            fourier_mat[i, j] = jy_eigs[(μ, Q, m)]
        end
    end

    exp_θ = exp.(-im .* μ_values .* θ)
    exp_φ = exp.(im .* m_values .* φ)

    return (fourier_mat * exp_θ) .* exp_φ .* sqrt((2 * l + 1) / (4.0 * π))
end
