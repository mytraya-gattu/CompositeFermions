module DFTWavefunction

export ΨprojDFT, load_dft_orbitals

using LinearAlgebra

include("symmetric_polynomials.jl")
using .SymmetricPolynomials
include("jk_projection_utilities.jl")
using .JKProjection

using JLD2

# Bring proposal and rand_θ_ϕ_gen from the parent module (CFsOnSphere), which loads
# MonteCarloOnSphere (exports both) before this module.
using ..: proposal, rand_θ_ϕ_gen
# Extend the generics from SpinPolarizedProjectedWavefunction so all dispatch lives on
# the same function objects — avoids "both ... export X" ambiguity in CFsOnSphere.
import ..: gibbs_thermalization!, update_wavefunction!

# -----------------------------------------------------------------------
# ΨprojDFT: Jain-Kamilla projected CF wavefunction whose N occupied
# single-particle orbitals are arbitrary linear combinations of all
# Lambda-level basis states λ = 0..nmax, as produced by CF-DFT.
#
# The CF-DFT orbital file stores a (dim_basis × N) coefficient matrix
# `coeffs` where column j gives the amplitudes of the N-th eigenstate
# in the monopole-harmonic basis {(λ, L, M)}.
#
# The wavefunction is:
#   Ψ(z_1,...,z_N) = P_LLL [ ∏_{i<j}(z_i-z_j)^{2p}  det[ ψ_j(Ω_i) ] ]
# where
#   ψ_j(Ω) = Σ_a  coeffs[a,j]  Y_{Q*,L_a,M_a}(Ω)
# and Y_{Q*,L,M} are monopole harmonics evaluated via the JK projection.
# -----------------------------------------------------------------------

function u_v_generator(θ, ϕ)
    return cos.(θ ./ 2) .* exp.(0.5im .* ϕ), sin.(θ ./ 2) .* exp.(-0.5im .* ϕ)
end

mutable struct ΨprojDFT
    Qstar::Rational{Int64}
    p::Int64
    system_size::Int64          # N (electrons)

    l_m_list_full::Vector{NTuple{2, Rational{Int64}}}  # all dim basis states
    dim_basis::Int64            # total number of basis states
    nmax::Int64                 # highest Lambda level (nmax = Lmax - |Qstar|)
    Lmax::Rational{Int64}

    μ_list::Vector{Rational{Int64}}
    Lz_list::Vector{Rational{Int64}}

    # DFT coefficient matrix: (dim_basis × system_size)
    dft_coeffs::Matrix{ComplexF64}

    fourier_tot_matrix::Matrix{ComplexF64}

    U::Vector{ComplexF64}
    V::Vector{ComplexF64}

    exp_θ::Matrix{ComplexF64}
    exp_ϕ::Matrix{ComplexF64}       # (dim_basis × system_size)

    dist_matrix::Matrix{Float64}
    u_v_ratio_matrix::Matrix{ComplexF64}

    elementary_symmetric_polynomials::Matrix{ComplexF64}
    reg_coeffs::Vector{Float64}

    wigner_d_matrices::Matrix{ComplexF64}   # (dim_basis*(nmax+1)) × system_size
    wigner_D_matrices::Array{ComplexF64, 3} # dim_basis × (nmax+1) × system_size

    jastrow_factor_log::ComplexF64
    raw_slater_det::Matrix{ComplexF64}  # (dim_basis × system_size): un-rotated
    slater_det::Matrix{ComplexF64}      # (system_size × system_size): DFT-rotated
end

# -------------------------------------------------------------------
# Constructor: builds the full basis for Lambda levels 0..nmax
# and initialises all arrays.
# -------------------------------------------------------------------
function ΨprojDFT(Qstar::Rational{Int64}, p::Int64, system_size::Int64, nmax::Int64, dft_coeffs::Matrix{ComplexF64})
    qabs = abs(Qstar)
    # Build full l_m_list: all (L, M) for λ = 0..nmax
    l_m_list_full = NTuple{2, Rational{Int64}}[]
    for lambda in 0:nmax
        L = qabs + lambda
        for M in -L:1:L
            push!(l_m_list_full, (L, M))
        end
    end
    dim_basis = length(l_m_list_full)
    Lmax = qabs + nmax

    size(dft_coeffs) == (dim_basis, system_size) ||
        throw(ArgumentError("dft_coeffs must be ($dim_basis × $system_size), got $(size(dft_coeffs))"))

    # Fourier matrix: same construction as Ψproj but for the full l_m_list
    fourier_matrix = zeros(ComplexF64, dim_basis, nmax + 1, numerator(1 + 2 * Lmax))

    Lgrid = unique(first.(l_m_list_full))
    liters = [findall(x -> x[1] == L, l_m_list_full) for L in Lgrid]

    for (Liter, L) in enumerate(Lgrid)
        lambda = Int(L - qabs)
        Lz_sub = last.(l_m_list_full[liters[Liter]])
        fm = generate_fourier_matrices(Qstar, system_size, L, Lz_sub)
        # fm has shape (n_states_at_L, lambda+1, 2L+1)
        # place into fourier_matrix at columns covering [-L..L] within [-Lmax..Lmax]
        μ_start = numerator(1 - L + Lmax)
        μ_end   = numerator(1 + L + Lmax)
        fourier_matrix[liters[Liter], 1:(lambda + 1), μ_start:μ_end] .= fm
    end

    fourier_tot_matrix = reshape(fourier_matrix, dim_basis * (nmax + 1), numerator(1 + 2 * Lmax))

    μ_list = collect(-Lmax:1:Lmax)
    Lz_list = last.(l_m_list_full)

    U    = zeros(ComplexF64, system_size)
    V    = zeros(ComplexF64, system_size)
    exp_θ = zeros(ComplexF64, length(μ_list), system_size)
    exp_ϕ = zeros(ComplexF64, dim_basis, system_size)

    dist_matrix      = zeros(Float64, system_size - 1, system_size)
    u_v_ratio_matrix = zeros(ComplexF64, system_size - 1, system_size)
    elementary_symmetric_polynomials = zeros(ComplexF64, nmax + 1, system_size)
    reg_coeffs = Float64[i / (system_size - i) for i in 1:nmax]

    wigner_d_matrices = zeros(ComplexF64, dim_basis * (nmax + 1), system_size)
    wigner_D_matrices = zeros(ComplexF64, dim_basis, nmax + 1, system_size)

    raw_slater_det = zeros(ComplexF64, dim_basis, system_size)
    slater_det     = zeros(ComplexF64, system_size, system_size)

    return ΨprojDFT(Qstar, p, system_size, l_m_list_full, dim_basis, nmax, Lmax,
        μ_list, Lz_list, dft_coeffs,
        fourier_tot_matrix, U, V, exp_θ, exp_ϕ,
        dist_matrix, u_v_ratio_matrix,
        elementary_symmetric_polynomials, reg_coeffs,
        wigner_d_matrices, wigner_D_matrices,
        0.0 + 0.0im, raw_slater_det, slater_det)
end

# -------------------------------------------------------------------
# Full update (all particles)
# -------------------------------------------------------------------
function update_wavefunction!(Ψ::ΨprojDFT, θ::Vector{Float64}, ϕ::Vector{Float64})
    Ψ.exp_θ .= exp.(-1.0im .* Ψ.μ_list * transpose(θ))
    Ψ.exp_ϕ .= exp.(1.0im .* Ψ.Lz_list * transpose(ϕ))

    Ψ.jastrow_factor_log = zero(ComplexF64)
    Ψ.U, Ψ.V = u_v_generator(θ, ϕ)

    δu = zero(ComplexF64); δv = zero(ComplexF64)
    for i in 1:(Ψ.system_size - 1), j in (i + 1):Ψ.system_size
        δu = conj(Ψ.U[i]) * Ψ.U[j] + conj(Ψ.V[i]) * Ψ.V[j]
        δv = Ψ.U[i] * Ψ.V[j] - Ψ.V[i] * Ψ.U[j]
        Ψ.u_v_ratio_matrix[j - 1, i] = δu / δv
        Ψ.u_v_ratio_matrix[i, j]     = -conj(δu) / δv
        Ψ.jastrow_factor_log += Ψ.p * log(δv)
        Ψ.dist_matrix[j - 1, i] = 2.0 * abs(δv)
        Ψ.dist_matrix[i, j]     = Ψ.dist_matrix[j - 1, i]
    end

    @simd for ei in axes(Ψ.elementary_symmetric_polynomials, 2)
        @inbounds @views get_symmetric_polynomials!(
            Ψ.elementary_symmetric_polynomials[:, ei],
            Ψ.u_v_ratio_matrix[:, ei], Ψ.nmax, Ψ.reg_coeffs)
    end

    Ψ.wigner_d_matrices .= Ψ.fourier_tot_matrix * Ψ.exp_θ
    Ψ.wigner_D_matrices .= reshape(Ψ.wigner_d_matrices, size(Ψ.wigner_D_matrices))
    @simd for it in axes(Ψ.wigner_D_matrices, 3)
        @inbounds @views Ψ.wigner_D_matrices[:, :, it] .*= Ψ.exp_ϕ[:, it:it]
    end

    @simd for it in axes(Ψ.raw_slater_det, 2)
        @inbounds @views Ψ.raw_slater_det[:, it] .= Ψ.wigner_D_matrices[:, :, it] * Ψ.elementary_symmetric_polynomials[:, it]
    end
    mul!(Ψ.slater_det, Ψ.dft_coeffs', Ψ.raw_slater_det)
    return
end

# -------------------------------------------------------------------
# Single-particle update (move particle `iter`)
# -------------------------------------------------------------------
function update_wavefunction!(Ψ::ΨprojDFT, θ::Float64, ϕ::Float64, iter::Int64)
    Ψ.exp_θ[:, iter] .= exp.(-1.0im .* Ψ.μ_list * θ)
    Ψ.exp_ϕ[:, iter] .= exp.(1.0im .* Ψ.Lz_list * ϕ)

    unew, vnew = u_v_generator(θ, ϕ)

    δv_old = zero(ComplexF64); δv_new = zero(ComplexF64); δu_new = zero(ComplexF64)
    use_incremental = 2 * Ψ.nmax < Ψ.system_size

    for i in 1:Ψ.system_size
        if i < iter
            δv_old = Ψ.U[i] * Ψ.V[iter] - Ψ.V[i] * Ψ.U[iter]
            δv_new = Ψ.U[i] * vnew - Ψ.V[i] * unew
            δu_new = conj(Ψ.U[i]) * unew + conj(Ψ.V[i]) * vnew
            r_old = use_incremental ? Ψ.u_v_ratio_matrix[iter - 1, i] : 0.0+0.0im
            Ψ.u_v_ratio_matrix[iter - 1, i] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[i, iter]     = -conj(δu_new) / δv_new
            if use_incremental
                @inbounds @views update_symmetric_polynomials!(
                    Ψ.elementary_symmetric_polynomials[:, i],
                    r_old, Ψ.u_v_ratio_matrix[iter - 1, i], Ψ.nmax, Ψ.reg_coeffs)
            end
            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)
            Ψ.dist_matrix[iter - 1, i] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[i, iter]     = Ψ.dist_matrix[iter - 1, i]
        elseif i > iter
            δv_old = -Ψ.U[i] * Ψ.V[iter] + Ψ.V[i] * Ψ.U[iter]
            δv_new = -Ψ.U[i] * vnew + Ψ.V[i] * unew
            δu_new = Ψ.U[i] * conj(unew) + Ψ.V[i] * conj(vnew)
            r_old = use_incremental ? Ψ.u_v_ratio_matrix[iter, i] : 0.0+0.0im
            Ψ.u_v_ratio_matrix[i - 1, iter] = δu_new / δv_new
            Ψ.u_v_ratio_matrix[iter, i]     = -conj(δu_new) / δv_new
            if use_incremental
                @inbounds @views update_symmetric_polynomials!(
                    Ψ.elementary_symmetric_polynomials[:, i],
                    r_old, Ψ.u_v_ratio_matrix[iter, i], Ψ.nmax, Ψ.reg_coeffs)
            end
            Ψ.jastrow_factor_log += Ψ.p * log(δv_new / δv_old)
            Ψ.dist_matrix[i - 1, iter] = 2.0 * abs(δv_new)
            Ψ.dist_matrix[iter, i]     = Ψ.dist_matrix[i - 1, iter]
        end
    end
    Ψ.U[iter], Ψ.V[iter] = unew, vnew

    if use_incremental
        @inbounds @views get_symmetric_polynomials!(
            Ψ.elementary_symmetric_polynomials[:, iter],
            Ψ.u_v_ratio_matrix[:, iter], Ψ.nmax, Ψ.reg_coeffs)
    else
        @simd for ei in axes(Ψ.elementary_symmetric_polynomials, 2)
            @inbounds @views get_symmetric_polynomials!(
                Ψ.elementary_symmetric_polynomials[:, ei],
                Ψ.u_v_ratio_matrix[:, ei], Ψ.nmax, Ψ.reg_coeffs)
        end
    end

    Ψ.wigner_d_matrices[:, iter] .= Ψ.fourier_tot_matrix * Ψ.exp_θ[:, iter]
    @simd for j in axes(Ψ.wigner_D_matrices, 2)
        @inbounds @views Ψ.wigner_D_matrices[:, j, iter] .= Ψ.wigner_d_matrices[(1 + Ψ.dim_basis * (j - 1)):(Ψ.dim_basis * j), iter]
    end
    @inbounds @views Ψ.wigner_D_matrices[:, :, iter] .*= Ψ.exp_ϕ[:, iter:iter]

    @simd for ei in axes(Ψ.raw_slater_det, 2)
        @inbounds @views Ψ.raw_slater_det[:, ei] .= Ψ.wigner_D_matrices[:, :, ei] * Ψ.elementary_symmetric_polynomials[:, ei]
    end
    mul!(Ψ.slater_det, Ψ.dft_coeffs', Ψ.raw_slater_det)
    return
end

# -------------------------------------------------------------------
# copy! methods (full and single-particle)
# -------------------------------------------------------------------
function Base.copy!(Ψ1::ΨprojDFT, Ψ2::ΨprojDFT)
    Ψ1.dist_matrix .= Ψ2.dist_matrix
    Ψ1.exp_θ .= Ψ2.exp_θ
    Ψ1.exp_ϕ .= Ψ2.exp_ϕ
    Ψ1.U .= Ψ2.U
    Ψ1.V .= Ψ2.V
    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.raw_slater_det .= Ψ2.raw_slater_det
    Ψ1.slater_det .= Ψ2.slater_det
    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials
    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix
    Ψ1.wigner_d_matrices .= Ψ2.wigner_d_matrices
    Ψ1.wigner_D_matrices .= Ψ2.wigner_D_matrices
    return
end

function Base.copy!(Ψ1::ΨprojDFT, Ψ2::ΨprojDFT, iter::Int64)
    Ψ1.dist_matrix .= Ψ2.dist_matrix
    Ψ1.exp_θ[:, iter] .= Ψ2.exp_θ[:, iter]
    Ψ1.exp_ϕ[:, iter] .= Ψ2.exp_ϕ[:, iter]
    Ψ1.U[iter] = Ψ2.U[iter]
    Ψ1.V[iter] = Ψ2.V[iter]
    Ψ1.jastrow_factor_log = Ψ2.jastrow_factor_log
    Ψ1.raw_slater_det .= Ψ2.raw_slater_det
    Ψ1.slater_det .= Ψ2.slater_det
    Ψ1.elementary_symmetric_polynomials .= Ψ2.elementary_symmetric_polynomials
    Ψ1.u_v_ratio_matrix .= Ψ2.u_v_ratio_matrix
    Ψ1.wigner_d_matrices[:, iter] .= Ψ2.wigner_d_matrices[:, iter]
    Ψ1.wigner_D_matrices[:, :, iter] .= Ψ2.wigner_D_matrices[:, :, iter]
    return
end

# -------------------------------------------------------------------
# Thermalization for ΨprojDFT (mirrors gibbs_thermalization! in projected_wavefunction.jl)
# ARM logic inlined since arm_parameters/arm_scale_factor are not exported.
# -------------------------------------------------------------------
function _arm_params(p_ideal::Float64, r::Float64 = 3.0)
    a, b = 1.0, 0.0
    for _ in 1:1000
        c = (a * p_ideal + b)^r
        a = (a * p_ideal + b)^(1/r) - c
        b = c
    end
    return a, b
end
_arm_scale(p_obs, p_ideal, a, b) = log(a * p_ideal + b) / log(a * p_obs + b)

function gibbs_thermalization!(RNG, Ψcurrent::ΨprojDFT, Ψnext::ΨprojDFT,
        θcurrent::Vector{Float64}, ϕcurrent::Vector{Float64},
        θnext::Vector{Float64}, ϕnext::Vector{Float64},
        σinit::Float64, logpdf::Function, num_thermalization::Int64)

    a, b = _arm_params(0.5)
    σ = σinit
    n_accepted = 0

    update_wavefunction!(Ψcurrent, θcurrent, ϕcurrent)
    copy!(Ψnext, Ψcurrent)
    logpdf_current = logpdf(Ψcurrent)

    tuning = round.(Int64, exp.(LinRange(log(10.0), log(num_thermalization), 25)))
    sampling_iter = 1
    t0 = time()

    for mc_iter in 1:num_thermalization
        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)
        logpdf_next = logpdf(Ψnext)

        if logpdf_next - logpdf_current >= log(rand())
            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]
            copy!(Ψcurrent, Ψnext, sampling_iter)
            logpdf_current = logpdf_next
            n_accepted += 1
        else
            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]
            copy!(Ψnext, Ψcurrent, sampling_iter)
        end

        if mc_iter ∈ tuning
            σ *= _arm_scale(n_accepted / mc_iter, 0.5, a, b)
        end
        sampling_iter = mod(sampling_iter, Ψcurrent.system_size) + 1
    end

    return sampling_iter, σ, time() - t0, n_accepted / num_thermalization
end

# -------------------------------------------------------------------
# Load DFT orbital file (written by cf_full_dft/src/CFDFT.jl)
# -------------------------------------------------------------------
function load_dft_orbitals(path::String)
    d = JLD2.load(path)
    N     = d["N"]::Int
    twoQ  = d["twoQ"]::Int
    p     = d["p"]::Int
    nmax  = d["nmax"]::Int
    coeffs = d["coeffs"]::Matrix{ComplexF64}  # (dim_basis × N)
    Qstar  = twoQ // 2 - p * (N - 1)
    return Qstar, p, N, nmax, coeffs
end

end # module DFTWavefunction
