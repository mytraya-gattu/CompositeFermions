# Jain-Kamilla projected composite-fermion wavefunctions: the single-component `Ψproj`
# and the multi-sector parton `Ψparton`. Both carry a Slater determinant of *projected*
# (multi-particle) orbitals; moving one particle changes the elementary symmetric
# polynomials of every particle and hence every column of `slater_det`, so there is no
# rank-1 / Sherman-Morrison update here (see `Ψunproj` for that).

"""
    Ψproj

Mutable state of a Jain-Kamilla projected composite-fermion wavefunction (electrons + `p`
attached flux), at filling controlled by `Qstar`, `l_m_list`.

The JK projection binds a single vortex pair into each orbital (`Q₁ = (N-1)/2`, ESP roots
with multiplicity one). Higher Jastrow powers `ν = n/(2pn+1)` come from the global Jastrow
factor `∏(uᵢvⱼ-uⱼvᵢ)^p`: e.g. `Ψ_{n/(2pn+1)} = Φ₁^{2(p̃-1)} P_LLL[Φₙ Φ₁²]` is obtained with
`p = 2 p̃`.

# Key fields
- `Qstar::Rational{Int64}`: effective monopole strength `Q*`.
- `p::Int64`: power of the global Jastrow factor `∏(uᵢvⱼ-uⱼvᵢ)^p`.
- `system_size::Int64`: number of electrons `N`.
- `l_m_list`: occupied `(L, Lz)` orbitals.
- `slater_det::Matrix{ComplexF64}`: the projected Slater matrix.
- `jastrow_factor_log::ComplexF64`: log of the global Jastrow factor.
- `reg_coeffs::Vector{Float64}`: ESP normalization, `reg_coeffs[i] = i/((N-1)-i+1)` so
  `∏ = 1/C(N-1, k)`.

The remaining fields (`U`, `V`, `exp_θ`, `exp_ϕ`, `dist_matrix`, `u_v_ratio_matrix`,
`elementary_symmetric_polynomials`, `wigner_d_matrices`, `wigner_D_matrices`,
`fourier_tot_matrix`, `μ_list`, `Lz_list`, `Lmax`) are work arrays / caches.
"""
mutable struct Ψproj

    Qstar::Rational{Int64}
    p::Int64
    system_size::Int64

    l_m_list::Vector{NTuple{2,Rational{Int64}}}
    Lmax::Rational{Int64}
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64}

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}
    wigner_D_matrices::Array{ComplexF64,3}

    jastrow_factor_log::ComplexF64
    slater_det::Matrix{ComplexF64}
end

"""
    Ψparton

Mutable state of a Jain-Kamilla projected parton wavefunction (multiple sectors with
per-sector `Qstars` and `l_m_lists`). As in `Ψproj`, the JK projection binds a single vortex
pair into each orbital (`Q₁ = (N-1)/2`).

Fields mirror `Ψproj`, with `Qstars`/`l_m_lists` replacing the single-sector versions,
`max_JK` the largest JK order over sectors, and `trackers[i]` the `slater_det` row range of
parton `i`.
"""
mutable struct Ψparton

    Qstars::Vector{Rational{Int64}}
    p::Int64
    system_size::Int64

    l_m_lists::Vector{Vector{NTuple{2,Rational{Int64}}}}

    max_JK::Int64
    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    fourier_tot_matrix::Matrix{ComplexF64}

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}
    wigner_D_matrices::Array{ComplexF64,3}

    jastrow_factor_log::ComplexF64
    slater_det::Matrix{ComplexF64}
    trackers::Vector{UnitRange{Int64}}

end

"""
    Ψproj(Qstar, p, system_size, l_m_list)

Construct a Jain-Kamilla projected CF wavefunction. `Qstar = Q - p(N-1)/2` is the effective
monopole strength, `p` the global Jastrow power, `l_m_list` the occupied `(L, Lz)` orbitals.
The JK projection binds a single vortex pair into each orbital (`Q₁ = (N-1)/2`).
"""
function Ψproj(Qstar::Rational{Int64}, p::Int64, system_size::Int64, l_m_list::Vector{NTuple{2,Rational{Int64}}})

    Lmax = maximum(first, l_m_list)

    fourier_matrix = zeros(ComplexF64, length(l_m_list), numerator(1 + Lmax-Qstar), numerator(1+2*Lmax))

    Lgrid = unique(first.(l_m_list))
    liters = [findall(x->x[1]==L, l_m_list) for L in Lgrid]

    for (Liter, L) in enumerate(Lgrid)

        fourier_matrix[liters[Liter], begin:begin+numerator(L-Qstar), numerator(1-L+Lmax):1:numerator(1+L+Lmax)] .= generate_fourier_matrices(Qstar, system_size, L, last.(l_m_list[liters[Liter]]))

    end

    fourier_tot_matrix = reshape(fourier_matrix, :, numerator(1+2*Lmax))

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
        reg_coeffs[i] = i / ((system_size - 1) - i + 1)
    end

    return Ψproj(Qstar, p, system_size, l_m_list, Lmax, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, u_v_ratio_matrix, elementary_symmetric_polynomials, reg_coeffs, wigner_d_matrices, wigner_D_matrices, jastrow_factor_log, slater_det)
end

"""
    Ψparton(Qstars, p, system_size, l_m_lists)

Construct a Jain-Kamilla projected parton wavefunction. `Qstars[i]` and `l_m_lists[i]` give
the effective monopole strength and occupied orbitals of parton `i`.
"""
function Ψparton(Qstars::Vector{Rational{Int64}}, p::Int64, system_size::Int64, l_m_lists::Vector{Vector{NTuple{2, Rational{Int64}}}})

    @assert length(Qstars) == length(l_m_lists) "Number of Qstars and l_m_lists should be the same."
    max_JK = typemin(Int64)

    Lmaxs = Vector{Rational{Int64}}()

    for iter in eachindex(Qstars)

        Lmax = maximum(first, l_m_lists[iter])
        push!(Lmaxs, Lmax)
        max_JK = max(max_JK, numerator(Lmax-Qstars[iter]))

    end

    μ_list = sort(unique(vcat([collect(-Lmax:1:Lmax) for Lmax in Lmaxs]...)))

    fourier_matrix = zeros(ComplexF64, sum(length, l_m_lists), max_JK + 1, length(μ_list))

    tracker = 0
    trackers = Vector{UnitRange{Int64}}()
    for iter in eachindex(l_m_lists)

        l_m_list = l_m_lists[iter]
        Qstar = Qstars[iter]

        Lgrid = unique(first.(l_m_list))
        liters = [findall(x->x[1]==L, l_m_list) for L in Lgrid]

        for (Liter, L) in enumerate(Lgrid)

            θ_iters = [findfirst(isequal(μ),μ_list) for μ in -L:1:L]

            fourier_matrix[tracker .+ liters[Liter], begin:begin+numerator(L-Qstar), θ_iters] .= generate_fourier_matrices(Qstar, system_size, L, last.(l_m_list[liters[Liter]]))

        end

        push!(trackers, 1 + tracker:tracker + length(l_m_list))

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
        reg_coeffs[i] = (i/(system_size-i))
    end

    return Ψparton(Qstars, p, system_size, l_m_lists, max_JK, μ_list, Lz_list, fourier_tot_matrix, U, V, exp_θ, exp_ϕ, dist_matrix, u_v_ratio_matrix, elementary_symmetric_polynomials, reg_coeffs, wigner_d_matrices, wigner_D_matrices, jastrow_factor_log, slater_det, trackers)
end

# --- ESP order per wavefunction type (shared update code below) ---

_esp_order(Ψ::Ψproj) = numerator(Ψ.Lmax - Ψ.Qstar)
_esp_order(Ψ::Ψparton) = Ψ.max_JK

# --- Pairwise spinor data (U, V, jastrow, ratios, distances), shared by Ψproj/Ψparton ---

# Full recompute from all positions.
function _update_pairs!(Ψ::Union{Ψproj,Ψparton}, θ::Vector{Float64}, ϕ::Vector{Float64})

    # Broadcast outer products (μ_list ⊗ θ) fuse into exp_θ/exp_ϕ without a matmul temporary.
    Ψ.exp_θ .= exp.(-1.0im .* Ψ.μ_list .* transpose(θ))
    Ψ.exp_ϕ .= exp.(1.0im .* Ψ.Lz_list .* transpose(ϕ))

    Ψ.jastrow_factor_log = zero(ComplexF64)
    # In-place spinor coordinates (avoid allocating fresh U/V each call).
    Ψ.U .= cos.(θ ./ 2) .* exp.(0.5im .* ϕ)
    Ψ.V .= sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)

    for i = 1:Ψ.system_size-1
        for j = i+1:Ψ.system_size

            δu = conj(Ψ.U[i]) * Ψ.U[j] + conj(Ψ.V[i]) * Ψ.V[j]
            δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]

            Ψ.u_v_ratio_matrix[j-1, i] = δu / δv
            Ψ.u_v_ratio_matrix[i, j] = -conj(δu) / δv

            Ψ.jastrow_factor_log += Ψ.p * log(δv)

            Ψ.dist_matrix[j-1, i] = 2.0 * abs(δv)
            Ψ.dist_matrix[i, j] = Ψ.dist_matrix[j-1, i]

        end
    end

    return
end

# Incremental update for a single moved particle `iter`.
function _update_pairs!(Ψ::Union{Ψproj,Ψparton}, θ::Float64, ϕ::Float64, iter::Int64)

    @views Ψ.exp_θ[:, iter] .= exp.(-1.0im .* Ψ.μ_list .* θ)
    @views Ψ.exp_ϕ[:, iter] .= exp.(1.0im .* Ψ.Lz_list .* ϕ)

    unew, vnew = u_v_generator(θ, ϕ)

    for i = 1:Ψ.system_size

        if i < iter

            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew -  Ψ.V[i] * unew

            δu_new = conj(Ψ.U[i]) * unew + conj(Ψ.V[i]) * vnew

            Ψ.u_v_ratio_matrix[iter-1, i] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[i, iter] = -conj(δu_new) / δv_new

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[iter-1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter] = Ψ.dist_matrix[iter-1, i]

        elseif i > iter

            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew +  Ψ.V[i] * unew

            δu_new = (Ψ.U[i]) * conj(unew) + (Ψ.V[i]) * conj(vnew)

            Ψ.u_v_ratio_matrix[i-1, iter] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[iter, i] = -conj(δu_new) / δv_new

            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)

            Ψ.dist_matrix[i-1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i] = Ψ.dist_matrix[i-1, iter]

        end

    end

    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    return
end

"""
    update_wavefunction!(Ψ::Union{Ψproj,Ψparton}, θ::Vector{Float64}, ϕ::Vector{Float64})

Full rebuild of the projected wavefunction `Ψ` from all particle positions `(θ, ϕ)`.
Modifies `Ψ` in place.
"""
function update_wavefunction!(Ψ::Union{Ψproj,Ψparton}, θ::Vector{Float64}, ϕ::Vector{Float64})

    _update_pairs!(Ψ, θ, ϕ)

    b = _esp_order(Ψ)
    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)
        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], b, Ψ.reg_coeffs)
    end

    mul!(Ψ.wigner_d_matrices, Ψ.fourier_tot_matrix, Ψ.exp_θ)
    Ψ.wigner_D_matrices .= reshape(Ψ.wigner_d_matrices, size(Ψ.wigner_D_matrices))

    @simd for iter in axes(Ψ.wigner_D_matrices, 3)
        @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]
    end

    @simd for iter in axes(Ψ.slater_det, 2)
        @inbounds @views mul!(Ψ.slater_det[:, iter], Ψ.wigner_D_matrices[:, :, iter], Ψ.elementary_symmetric_polynomials[:, iter])
    end

    return
end

"""
    update_wavefunction!(Ψ::Union{Ψproj,Ψparton}, θ::Float64, ϕ::Float64, iter::Int64)

Update the projected wavefunction after moving only particle `iter` to `(θ, ϕ)`. Note that
because the projection couples all particles, the elementary symmetric polynomials and the
whole `slater_det` are recomputed (no rank-1 shortcut). Modifies `Ψ` in place.
"""
function update_wavefunction!(Ψ::Union{Ψproj,Ψparton}, θ::Float64, ϕ::Float64, iter::Int64)

    _update_pairs!(Ψ, θ, ϕ, iter)

    b = _esp_order(Ψ)
    @simd for electron_iter in axes(Ψ.elementary_symmetric_polynomials, 2)
        @inbounds @views get_symmetric_polynomials!(Ψ.elementary_symmetric_polynomials[:, electron_iter], Ψ.u_v_ratio_matrix[:, electron_iter], b, Ψ.reg_coeffs)
    end

    @views mul!(Ψ.wigner_d_matrices[:, iter], Ψ.fourier_tot_matrix, Ψ.exp_θ[:, iter])

    nr = size(Ψ.wigner_D_matrices, 1)
    nc = size(Ψ.wigner_D_matrices, 2)
    @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .= reshape(Ψ.wigner_d_matrices[:, iter], nr, nc)
    @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter]

    @simd for electron_iter in axes(Ψ.slater_det, 2)
        @inbounds @views mul!(Ψ.slater_det[:, electron_iter], Ψ.wigner_D_matrices[:, :, electron_iter], Ψ.elementary_symmetric_polynomials[:, electron_iter])
    end

    return
end

"""
    copy!(Ψ1, Ψ2)

Full in-place copy of one projected wavefunction into another of the same type.
"""
function Base.copy!(Ψ1::T, Ψ2::T) where {T<:Union{Ψproj,Ψparton}}

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
    copy!(Ψ1, Ψ2, iter::Int64)

Partial in-place copy assuming only particle `iter` changed (the per-particle caches plus
the full coupled arrays that a single move touches).
"""
function Base.copy!(Ψ1::T, Ψ2::T, iter::Int64) where {T<:Union{Ψproj,Ψparton}}

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
