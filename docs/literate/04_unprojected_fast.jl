# # 4. Unprojected states and fast Sherman–Morrison updates
#
# The **unprojected** CF wavefunction ``\det[Y_{Q^\star,l,m}(\Omega_i)]\,\prod(u_jv_k-u_kv_j)^p``
# is built from *single-particle* monopole-harmonic orbitals ([`Ψunproj`](@ref)). Because moving
# one particle changes only **one column** of the Slater determinant, the determinant ratio and
# inverse can be tracked by a rank-1 **Sherman–Morrison** update — turning the per-step cost from
# ``O(N^3)`` (a fresh `logdet`) into ``O(N)`` for the acceptance ratio plus ``O(N^2)`` for the
# inverse refresh.
#
# (This shortcut is *not* available for the projected [`Ψproj`](@ref): there the elementary
# symmetric polynomials couple all particles, so one move changes every column. See
# [Architecture](../architecture.md).)

using CFsOnSphere
using Random
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
using Plots
gr()

# ## The Sherman–Morrison sampling loop
#
# We thermalize with the generic driver (which uses `logdet`), then switch to the accelerated
# inner loop: maintain the inverse on the accepted state with [`initialize_inverse!`](@ref), get
# each acceptance ratio from [`slater_det_ratio`](@ref), and on acceptance apply
# [`update_inverse!`](@ref) **before** copying the state across.

function sample_unprojected(N, n, p; seed = 4, n_therm = 100_000, n_steps = 1_000_000, n_bins = 30)
    Qstar, l_m_list = cf_ground_state_lm(N, n, p)
    system_size = length(l_m_list)
    rng = MersenneTwister(seed)

    ψ      = Ψunproj(Qstar, p, system_size, l_m_list)
    ψ_next = Ψunproj(Qstar, p, system_size, l_m_list)
    logpdf(ψ) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

    θ, ϕ = rand_θ_ϕ_gen(rng, system_size)
    θ_next, ϕ_next = copy(θ), copy(ϕ)
    σ = π / sqrt(12)
    sampling_iter, σ, _, _ = gibbs_thermalization!(rng, ψ, ψ_next, θ, ϕ, θ_next, ϕ_next, σ, logpdf, n_therm)

    initialize_inverse!(ψ)                  # maintain the inverse on the accepted state
    temp = zeros(ComplexF64, system_size)

    θmesh = acos.(LinRange(1.0, -1.0, n_bins + 1))
    Agrid = 2π .* (cos.(θmesh[1:end-1]) .- cos.(θmesh[2:end]))
    density = zeros(Float64, n_bins)

    accepted = 0
    for _ in 1:n_steps
        i = sampling_iter
        θ_next[i], ϕ_next[i] = proposal(rng, θ[i], ϕ[i], σ)
        update_wavefunction!(ψ_next, θ_next[i], ϕ_next[i], i)

        det_ratio = slater_det_ratio(ψ, ψ_next, i)            # O(N)
        δ = 2.0 * real(log(det_ratio) + ψ_next.jastrow_factor_log - ψ.jastrow_factor_log)

        if δ >= log(rand(rng))
            θ[i], ϕ[i] = θ_next[i], ϕ_next[i]
            update_inverse!(ψ, ψ_next, i, det_ratio, temp)    # O(N²), before copy!
            copy!(ψ, ψ_next, i)
            accepted += 1
        else
            θ_next[i], ϕ_next[i] = θ[i], ϕ[i]
            copy!(ψ_next, ψ, i)
        end

        update_density!(θmesh, θ, density)
        sampling_iter = mod(sampling_iter, system_size) + 1
    end

    θc = 0.5 .* (θmesh[1:end-1] .+ θmesh[2:end])
    return θc, density ./ n_steps ./ Agrid, accepted / n_steps
end

# We sample the unprojected ``\nu = 2/5`` state (`n = 2`, `p = 2`) for `N = 8` electrons (a
# closed shell, so the determinant is square and the inverse can be maintained).

θc, dens, accept = sample_unprojected(8, 2, 2)
@show accept;

# Local filling ``\nu(\theta) = 2\pi\,n(\theta)/Q_{\text{shift}}`` with ``Q_{\text{shift}} = N/2\nu``;
# the bulk plateau sits at ``\nu = 2/5``, shown against the ``y=0`` baseline.
N      = 8
ν      = 2 / 5
Qshift = N / (2ν)
νθ     = dens .* (2π / Qshift)

plot(θc, νθ; lw=2, marker=:circle, ms=3, label="measured", ylims=(0.0, 0.6),
     xlabel="θ", ylabel="local filling  ν(θ)", title="Unprojected ν = 2/5 (N = 8, Sherman–Morrison)")
hline!([ν]; ls=:dash, label="ν = 2/5")
