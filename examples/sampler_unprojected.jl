# # Sampling an unprojected composite-fermion state with Sherman-Morrison updates
#
# The *unprojected* CF wavefunction det[Y_{Q*,l,m}(Ωᵢ)] · ∏(uⱼvₖ-uₖvⱼ)^p is built from
# single-particle monopole-harmonic orbitals (`Ψunproj`). Because moving one particle
# changes only one column of the Slater determinant, the Slater inverse can be maintained
# by a rank-1 Sherman-Morrison update — turning the per-step cost from O(N³) (`logdet`) into
# O(N) for the acceptance ratio plus O(N²) for the inverse update. (This shortcut does NOT
# apply to the *projected* `Ψproj`/`Ψparton`, where a single move changes every column.)

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using CFsOnSphere
using Random
const global RNG = Random.default_rng()

# Sample the unprojected CF ground state at ν = n/(2np+1) for `N` electrons.
function gibbs_sampler_unprojected(N::Int64, n::Int64, p::Int64, num_thermalization::Int64 = 5 * 10^5, num_steps::Int64 = 10^6)

    # Build the effective monopole strength and Λ-level occupation.
    Qstar, l_m_list = cf_ground_state_lm(N, n, p)
    system_size = length(l_m_list)

    # Two copies of the wavefunction: current (accepted) and next (proposed).
    Ψcurrent = Ψunproj(Qstar, p, system_size, l_m_list)
    Ψnext = Ψunproj(Qstar, p, system_size, l_m_list)

    # Random initial positions.
    θcurrent, ϕcurrent = rand_θ_ϕ_gen(RNG, system_size)
    θnext = copy(θcurrent)
    ϕnext = copy(ϕcurrent)

    # |Ψ|² = |det|² |Jastrow|²; gibbs_thermalization! maximises a real log-pdf.
    logpdf(ψ::Ψunproj) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

    # Phase 1 — thermalize and tune the step size σ with the generic driver (uses logdet).
    σ = pi / sqrt(12.0)
    sampling_iter, σ, _, _ = gibbs_thermalization!(RNG, Ψcurrent, Ψnext, θcurrent, ϕcurrent, θnext, ϕnext, σ, logpdf, num_thermalization)

    # Phase 2 — Sherman-Morrison-accelerated sampling. Maintain the inverse on Ψcurrent.
    initialize_inverse!(Ψcurrent)
    temp = zeros(ComplexF64, system_size)

    # Equal-area polar-angle density histogram.
    θmesh = map(x -> acos(x), LinRange(1.0, -1.0, 200))
    accumulated_density = zeros(Float64, length(θmesh) - 1)

    num_accepted = 0
    for monte_carlo_iter in 1:num_steps

        θnext[sampling_iter], ϕnext[sampling_iter] = proposal(RNG, θcurrent[sampling_iter], ϕcurrent[sampling_iter], σ)
        update_wavefunction!(Ψnext, θnext[sampling_iter], ϕnext[sampling_iter], sampling_iter)

        # O(N) determinant ratio from the maintained inverse and the new column.
        det_ratio = slater_det_ratio(Ψcurrent, Ψnext, sampling_iter)
        δlogpdf = 2.0 * real(log(det_ratio) + Ψnext.jastrow_factor_log - Ψcurrent.jastrow_factor_log)

        if δlogpdf >= log(rand(RNG))
            θcurrent[sampling_iter] = θnext[sampling_iter]
            ϕcurrent[sampling_iter] = ϕnext[sampling_iter]
            # O(N²) rank-1 inverse update, BEFORE copying the state across.
            update_inverse!(Ψcurrent, Ψnext, sampling_iter, det_ratio, temp)
            copy!(Ψcurrent, Ψnext, sampling_iter)
            num_accepted += 1
        else
            θnext[sampling_iter] = θcurrent[sampling_iter]
            ϕnext[sampling_iter] = ϕcurrent[sampling_iter]
            copy!(Ψnext, Ψcurrent, sampling_iter)
        end

        update_density!(θmesh, θcurrent, accumulated_density)

        sampling_iter = mod(sampling_iter, system_size) + 1
    end

    return θcurrent, ϕcurrent, accumulated_density ./ num_steps, num_accepted / num_steps
end

# A small smoke run (unprojected ν = 1/3, N = 9 in n = 3 filled Λ-levels).
# gibbs_sampler_unprojected(9, 3, 1, 10^4, 10^4)
