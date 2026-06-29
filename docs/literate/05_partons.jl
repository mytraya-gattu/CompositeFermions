# # 5. Parton states
#
# **Parton** wavefunctions generalize composite fermions: the electron is split into several
# fictitious "partons", each in its own integer-quantum-Hall-like state, and the physical
# wavefunction is the **product** of the parton determinants (projected to the LLL). The package
# represents such a state with [`Ψparton`](@ref), which stores one Slater matrix whose row blocks
# are the different parton species (tracked by `ψ.trackers`).
#
# This tutorial shows how to build a parton state and evaluate its amplitude. For a complete
# Monte Carlo parton sampler, see `examples/sampler_single_state_parton.jl` in the repository.

using CFsOnSphere
using Random
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# ## Constructing a parton state
#
# [`Ψparton`](@ref) takes a vector of effective monopole strengths `Qstars` (one per parton
# species) and a matching vector of orbital lists `l_m_lists`, each containing `N` orbitals (so
# every species forms an ``N\times N`` block). Here we build a simple two-species state; the
# species share the same ``\Lambda``-level structure for illustration.

N = 6
Qstar, l_m_list = cf_ground_state_lm(N, 1, 2)     # one filled Λ-level, N orbitals
Qstars     = [Qstar, Qstar]
l_m_lists  = [l_m_list, l_m_list]
p = 2

ψ = Ψparton(Qstars, p, N, l_m_lists)
@show length(ψ.trackers) size(ψ.slater_det);

# `ψ.trackers[s]` gives the rows of `ψ.slater_det` belonging to species `s` — each an
# ``N\times N`` block.

# ## Evaluating the amplitude
#
# Updating is identical to the projected wavefunction. The parton amplitude is the **product of
# the per-species block determinants** times the Jastrow factor, so the log-amplitude is the sum
# of the block `logdet`s:

rng = MersenneTwister(8)
θ, ϕ = rand_θ_ϕ_gen(rng, N)
update_wavefunction!(ψ, θ, ϕ)

logamp = sum(logdet(ψ.slater_det[tr, :]) for tr in ψ.trackers) + ψ.jastrow_factor_log
println("number of species = ", length(ψ.trackers))
println("Re(log Ψ) = ", round(real(logamp), digits=6))
println("finite amplitude: ", isfinite(real(logamp)))

# To sample a parton state you would use this product-of-determinants amplitude as the log-pdf
# inside the same Metropolis–Hastings–Gibbs loop as the earlier tutorials (driving it with
# [`gibbs_thermalization!`](@ref), which already accepts [`Ψparton`](@ref)).
