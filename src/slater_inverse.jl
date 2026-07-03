# Sherman-Morrison maintenance of the Slater inverse for `Ψunproj`, and an extended-Slater
# helper for fixed quasihole / quasiparticle orbital columns.
#
# These apply ONLY to `Ψunproj` (single-particle orbitals): moving one particle changes a
# single column of `slater_det`, so the inverse updates by a rank-1 (Sherman-Morrison)
# formula. They are deliberately NOT defined for `Ψproj`/`Ψparton`, where a single move
# changes every column (the projection couples all particles) and the inverse must be
# recomputed (use `logdet` per step there).
#
# Typical usage (custom MCMC loop with `ψcurrent`, `ψnext`):
#
#   update_wavefunction!(ψcurrent, θ, ϕ);  initialize_inverse!(ψcurrent)
#   temp = zeros(ComplexF64, N)
#   for step ...
#       update_wavefunction!(ψnext, θ_i, ϕ_i, i)
#       dr = slater_det_ratio(ψcurrent, ψnext, i)          # O(N)
#       if 2*real(log(dr) + ψnext.jastrow_factor_log - ψcurrent.jastrow_factor_log) ≥ log(rand(RNG))
#           update_inverse!(ψcurrent, ψnext, i, dr, temp)  # O(N²); update BEFORE copy!
#           copy!(ψcurrent, ψnext, i)
#       else
#           copy!(ψnext, ψcurrent, i)
#       end
#   end

"""
    initialize_inverse!(ψ::Ψunproj)

Compute `ψ.slater_det_inv = inv(ψ.slater_det)` (O(N³)). Call once after the first full
`update_wavefunction!`; requires a square (closed-shell) `slater_det`.
"""
function initialize_inverse!(ψ::Ψunproj)
    ψ.slater_det_inv .= inv(ψ.slater_det)
    return
end

"""
    slater_det_ratio(ψcurrent::Ψunproj, ψnext::Ψunproj, iter::Int) -> ComplexF64

The determinant ratio `det(Sₙₑₓₜ)/det(S_current)` for a move of particle `iter`, computed in
O(N) from the maintained inverse of `ψcurrent` and the new column of `ψnext`:
`Sinv[iter, :] · Sₙₑₓₜ[:, iter]`.
"""
function slater_det_ratio(ψcurrent::Ψunproj, ψnext::Ψunproj, iter::Int)
    @inbounds @views return transpose(ψcurrent.slater_det_inv[iter, :]) * ψnext.slater_det[:, iter]
end

"""
    update_inverse!(ψcurrent::Ψunproj, ψnext::Ψunproj, iter::Int, det_ratio::ComplexF64, temp::Vector{ComplexF64})

Rank-1 Sherman-Morrison update of `ψcurrent.slater_det_inv` for an accepted move of particle
`iter` to `ψnext` (O(N²)). `det_ratio` is the value returned by [`slater_det_ratio`](@ref);
`temp` is an `N`-length scratch vector. Call **before** `copy!(ψcurrent, ψnext, iter)`.
"""
function update_inverse!(ψcurrent::Ψunproj, ψnext::Ψunproj, iter::Int, det_ratio::ComplexF64, temp::Vector{ComplexF64})
    Sinv = ψcurrent.slater_det_inv
    @inbounds @views mul!(temp, Sinv, ψnext.slater_det[:, iter])
    @inbounds temp[iter] -= one(ComplexF64)

    # Unconjugated rank-1 update Sinv .-= (temp ⊗ Sinv[iter, :]) / det_ratio. Julia's BLAS
    # exposes only the conjugating `ger!` for complex types, so do it manually. Row `iter`
    # is updated last so the other rows read its original (pre-update) entries.
    invdr = one(ComplexF64) / det_ratio
    n = size(Sinv, 1)
    @inbounds for a in 1:n
        a == iter && continue
        fa = temp[a] * invdr
        @simd for b in 1:n
            Sinv[a, b] -= fa * Sinv[iter, b]
        end
    end
    @inbounds begin
        fiter = temp[iter] * invdr
        @simd for b in 1:n
            Sinv[iter, b] -= fiter * Sinv[iter, b]
        end
    end
    return
end

"""
    build_extended_slater!(Sfull, ψ, qh_columns) -> LU

Fill the preallocated `(N+k)×(N+k)` matrix `Sfull` with the electron Slater block of `ψ`
(first `N` columns) and `k` fixed quasihole / quasiparticle orbital columns `qh_columns`
(built from [`calculate_ll`](@ref)), and return its LU factorization.

Works for `Ψproj` (recompute the LU each step — the projection has no rank-1 shortcut) and
for `Ψunproj`. Quasihole amplitudes follow from `transpose(LU) \\ e_end`.
"""
function build_extended_slater!(Sfull::Matrix{ComplexF64}, ψ::Union{Ψproj,Ψunproj}, qh_columns::AbstractMatrix{ComplexF64})
    N = ψ.system_size
    @inbounds @views Sfull[:, 1:N] .= ψ.slater_det
    @inbounds @views Sfull[:, N+1:end] .= qh_columns
    return lu(Sfull)
end
