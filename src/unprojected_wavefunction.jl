# Unprojected composite-fermion wavefunctions on the sphere.
#
# `Ψunproj` is the unprojected CF state  det[Y_{Q*,l,m}(Ωᵢ)] · ∏_{j<k}(uⱼvₖ-uₖvⱼ)^p,
# built from *single-particle* monopole-harmonic orbitals (see `calculate_ll`). Because each
# orbital depends on one particle only, moving particle `iter` changes exactly one column of
# `slater_det`, so the Slater inverse can be tracked by a rank-1 Sherman-Morrison update
# (see `slater_inverse.jl`). This is the key structural difference from `Ψproj`.
#
# `ΨoneLL` is the bare Jastrow (Laughlin ν = 1/(2p+1)) wavefunction with no Slater
# determinant.

"""
    Ψunproj

Mutable state of an unprojected composite-fermion wavefunction with single-particle
monopole-harmonic orbitals.

# Key fields
- `Qstar::Rational{Int64}`: monopole strength of the orbitals.
- `p::Int64`: power of the global Jastrow factor `∏(uᵢvⱼ-uⱼvᵢ)^p`.
- `system_size::Int64`: number of electrons `N`.
- `l_m_list`: occupied `(L, Lz)` orbitals.
- `U`, `V`: spinor coordinates.
- `dist_matrix::Matrix{Float64}`: pairwise chord distances.
- `slater_det::Matrix{ComplexF64}`: `slater_det[orbital, particle] = Y_{Q*,L,Lz}(Ωᵢ)`.
- `slater_det_inv::Matrix{ComplexF64}`: maintained inverse of `slater_det` (valid only for a
  square / closed-shell determinant; updated by the Sherman-Morrison helpers).
- `jastrow_factor_log::ComplexF64`: log of the global Jastrow factor.

`fourier_tot_matrix`, `μ_list`, `Lz_list`, `exp_θ`, `exp_ϕ` are caches for the orbitals.
"""
mutable struct Ψunproj

    Qstar::Rational{Int64}
    p::Int64
    system_size::Int64

    l_m_list::Vector{NTuple{2,Rational{Int64}}}
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64}

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}

    slater_det::Matrix{ComplexF64}
    slater_det_inv::Matrix{ComplexF64}

    jastrow_factor_log::ComplexF64
end

"""
    Ψunproj(Qstar, p, system_size, l_m_list)

Construct an unprojected CF wavefunction: a Slater determinant of monopole harmonics
`Y_{Qstar,L,Lz}` (the orbitals in `l_m_list`) times the global Jastrow `∏(uᵢvⱼ-uⱼvᵢ)^p`.
The orbital transform is precomputed from the `J_y` eigenstates.
"""
function Ψunproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2,Rational{Int64}}})

    num_orbitals = length(l_m_list)
    Lz_list = last.(l_m_list)

    Lgrid = unique(first.(l_m_list))
    Lmax = maximum(Lgrid)
    μ_list = collect(-Lmax:1:Lmax)
    num_μ = length(μ_list)

    fourier_tot_matrix = zeros(ComplexF64, num_orbitals, num_μ)
    for L in Lgrid
        orbital_iters = findall(x -> x[1] == L, l_m_list)
        jy = calculate_j_y_eigenstates(L)
        for oi in orbital_iters
            m = Lz_list[oi]
            for (μ_idx, μ) in enumerate(μ_list)
                if abs(μ) <= L
                    fourier_tot_matrix[oi, μ_idx] = jy[(μ, Qstar, m)] * sqrt((2 * L + 1) / (4.0 * π))
                end
            end
        end
    end

    U = zeros(ComplexF64, system_size)
    V = zeros(ComplexF64, system_size)

    exp_θ = zeros(ComplexF64, num_μ, system_size)
    exp_ϕ = zeros(ComplexF64, num_orbitals, system_size)

    dist_matrix = zeros(Float64, system_size - 1, system_size)

    slater_det = zeros(ComplexF64, num_orbitals, system_size)
    slater_det_inv = zeros(ComplexF64, system_size, system_size)

    jastrow_factor_log = 0.0 + 0.0im

    return Ψunproj(Qstar, p, system_size, l_m_list, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, slater_det, slater_det_inv, jastrow_factor_log)
end

"""
    update_wavefunction!(ψ::Ψunproj, θ::Vector{Float64}, ϕ::Vector{Float64})

Full rebuild of `ψ` from all particle positions. Modifies `ψ` in place. (Does not refresh
`slater_det_inv`; call [`initialize_inverse!`](@ref) afterwards if you maintain the inverse.)
"""
function update_wavefunction!(ψ::Ψunproj, θ::Vector{Float64}, ϕ::Vector{Float64})

    ψ.exp_θ .= exp.(-1.0im .* ψ.μ_list .* transpose(θ))
    ψ.exp_ϕ .= exp.(1.0im .* ψ.Lz_list .* transpose(ϕ))

    ψ.U .= cos.(θ ./ 2) .* exp.(0.5im .* ϕ)
    ψ.V .= sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)

    mul!(ψ.slater_det, ψ.fourier_tot_matrix, ψ.exp_θ)
    ψ.slater_det .*= ψ.exp_ϕ

    ψ.jastrow_factor_log = zero(ComplexF64)
    for i = 1:ψ.system_size-1
        for j = i+1:ψ.system_size
            δv = ψ.U[i] * ψ.V[j] - ψ.V[i] * ψ.U[j]
            ψ.jastrow_factor_log += ψ.p * log(δv)
            ψ.dist_matrix[j-1, i] = 2.0 * abs(δv)
            ψ.dist_matrix[i, j] = ψ.dist_matrix[j-1, i]
        end
    end

    return
end

"""
    update_wavefunction!(ψ::Ψunproj, θ::Float64, ϕ::Float64, iter::Int64)

Incremental update after moving only particle `iter` to `(θ, ϕ)`: updates the Jastrow log,
the `iter` row/column of `dist_matrix`, and only column `iter` of `slater_det` (single
particle orbital). Modifies `ψ` in place.
"""
function update_wavefunction!(ψ::Ψunproj, θ::Float64, ϕ::Float64, iter::Int64)

    unew, vnew = u_v_generator(θ, ϕ)
    uold, vold = ψ.U[iter], ψ.V[iter]

    for i = 1:ψ.system_size

        if i < iter

            δv_old = ψ.U[i] * vold - ψ.V[i] * uold
            δv_new = ψ.U[i] * vnew - ψ.V[i] * unew

            ψ.jastrow_factor_log += ψ.p * log(δv_new / δv_old)

            ψ.dist_matrix[iter-1, i] = 2.0 * abs(δv_new)
            ψ.dist_matrix[i, iter] = ψ.dist_matrix[iter-1, i]

        elseif i > iter

            δv_old = -ψ.U[i] * vold + ψ.V[i] * uold
            δv_new = -ψ.U[i] * vnew + ψ.V[i] * unew

            ψ.jastrow_factor_log += ψ.p * log(δv_new / δv_old)

            ψ.dist_matrix[i-1, iter] = 2.0 * abs(δv_new)
            ψ.dist_matrix[iter, i] = ψ.dist_matrix[i-1, iter]

        end

    end

    ψ.U[iter] = unew
    ψ.V[iter] = vnew

    @views ψ.exp_θ[:, iter] .= exp.(-1.0im .* ψ.μ_list .* θ)
    @views ψ.exp_ϕ[:, iter] .= exp.(1.0im .* ψ.Lz_list .* ϕ)

    @inbounds @views mul!(ψ.slater_det[:, iter], ψ.fourier_tot_matrix, ψ.exp_θ[:, iter])
    @inbounds @views ψ.slater_det[:, iter] .*= ψ.exp_ϕ[:, iter]

    return
end

"""
    copy!(ψ1::Ψunproj, ψ2::Ψunproj)

Full in-place copy. Includes `slater_det_inv`.
"""
function Base.copy!(ψ1::Ψunproj, ψ2::Ψunproj)
    ψ1.U .= ψ2.U
    ψ1.V .= ψ2.V
    ψ1.exp_θ .= ψ2.exp_θ
    ψ1.exp_ϕ .= ψ2.exp_ϕ
    ψ1.dist_matrix .= ψ2.dist_matrix
    ψ1.slater_det .= ψ2.slater_det
    ψ1.slater_det_inv .= ψ2.slater_det_inv
    ψ1.jastrow_factor_log = ψ2.jastrow_factor_log
    return
end

"""
    copy!(ψ1::Ψunproj, ψ2::Ψunproj, iter::Int64)

Partial in-place copy assuming only particle `iter` changed. Does **not** copy
`slater_det_inv` (the Sherman-Morrison helpers maintain the inverse on the accepted state).
"""
function Base.copy!(ψ1::Ψunproj, ψ2::Ψunproj, iter::Int64)
    ψ1.U[iter] = ψ2.U[iter]
    ψ1.V[iter] = ψ2.V[iter]
    ψ1.exp_θ[:, iter] .= ψ2.exp_θ[:, iter]
    ψ1.exp_ϕ[:, iter] .= ψ2.exp_ϕ[:, iter]
    ψ1.dist_matrix .= ψ2.dist_matrix
    ψ1.slater_det[:, iter] .= ψ2.slater_det[:, iter]
    ψ1.jastrow_factor_log = ψ2.jastrow_factor_log
    return
end


# ----------------------------------------------------------------------------------------
# ΨoneLL: bare Jastrow (Laughlin ν = 1/(2p+1)) wavefunction, no Slater determinant.
# ----------------------------------------------------------------------------------------

"""
    ΨoneLL

Bare single-Landau-level Jastrow wavefunction `∏_{j<k}(uⱼvₖ-uₖvⱼ)^p` (no Slater
determinant). Fields: `p`, `system_size`, `U`, `V`, `dist_matrix`, `jastrow_factor_log`.
"""
mutable struct ΨoneLL

    p::Int64
    system_size::Int64

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    dist_matrix::Matrix{Float64}

    jastrow_factor_log::ComplexF64
end

"""
    ΨoneLL(p, system_size)

Construct a bare Jastrow (Laughlin) wavefunction.
"""
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

    for i = 1:Ψ.system_size-1
        for j = i+1:Ψ.system_size

            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.jastrow_factor_log += Ψ.p * log(δv)

            Ψ.dist_matrix[j-1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j-1, i]

        end
    end

    return
end

function update_wavefunction!(Ψ::ΨoneLL, θ::Float64, ϕ::Float64, iter::Int64)

    unew, vnew = u_v_generator(θ, ϕ)

    for i = 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew -  Ψ.V[i] * unew

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter-1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter-1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew +  Ψ.V[i] * unew

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i-1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i-1, iter]

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
