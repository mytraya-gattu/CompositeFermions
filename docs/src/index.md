# CFsOnSphere.jl

*Composite-fermion wavefunctions on the sphere, for the fractional quantum Hall effect.*

CFsOnSphere builds **Jain–Kamilla projected** and **unprojected** composite-fermion (CF) and
parton wavefunctions on the Haldane sphere, and samples them with a Metropolis–Hastings–Gibbs
Monte Carlo walk to compute densities, pair correlations, energies, and overlaps. The
projection uses the quaternion/rotation reformulation of Jain–Kamilla projection
([arXiv:2412.09670](https://arxiv.org/abs/2412.09670)), which is far cheaper and more numerically
stable than the traditional mixed-derivative approach.

If you are new here, read the [Physics background](physics.md) for the concepts and notation,
then work through the [Tutorials](tutorials/01_ground_state.md).

## Installation

The package is not yet in the General registry, so install it from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/mytraya-gattu/CompositeFermions.git")
```

or, for development, clone it and `Pkg.develop` the checkout:

```bash
git clone https://github.com/mytraya-gattu/CompositeFermions.git
```
```julia
using Pkg; Pkg.develop(path="CompositeFermions")
```

## Quickstart

Estimate the density of the ``\nu = 1/3`` composite-fermion ground state for a handful of
particles:

```julia
using CFsOnSphere, Random, LinearAlgebra
rng = MersenneTwister(1)

# Filling ν = n/(pn+1); p is the Jastrow power (even). n=1, p=2  →  ν = 1/3.
N, n, p = 6, 1, 2
Qstar, l_m_list = cf_ground_state_lm(N, n, p)

# Two wavefunction buffers (current + proposed) for the Metropolis walk.
ψ, ψ_next = Ψproj(Qstar, p, N, l_m_list), Ψproj(Qstar, p, N, l_m_list)

# Sampling weight |Ψ|²  ⇒  log-pdf = 2 Re(log det + log Jastrow).
logpdf(ψ) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

θ, ϕ = rand_θ_ϕ_gen(rng, N)
θn, ϕn = copy(θ), copy(ϕ)
σ = π / sqrt(12)
iter, σ, _, accept = gibbs_thermalization!(rng, ψ, ψ_next, θ, ϕ, θn, ϕn, σ, logpdf, 10_000)
@show accept
```

The [first tutorial](tutorials/01_ground_state.md) turns this into a full density and
pair-correlation measurement, with figures.

## What's inside

| Wavefunction | Description | Fast rank-1 updates? |
|---|---|---|
| [`Ψproj`](@ref)   | Jain–Kamilla projected CF state | No (a move changes every column) |
| [`Ψparton`](@ref) | Jain–Kamilla projected parton state | No |
| [`Ψunproj`](@ref) | unprojected `det·Jastrow` (single-particle orbitals) | **Yes (Sherman–Morrison)** |
| [`ΨoneLL`](@ref)  | bare Jastrow (Laughlin) | n/a |

## Where to go next

- [Physics background](physics.md) — LLL, composite fermions, the Jain–Kamilla projection, and the
  meaning of `Qstar`, `p`, and `l_m_list`.
- [Tutorials](tutorials/01_ground_state.md) — hands-on, example-driven walkthroughs.
- [API reference](api.md) — every exported function and type.
- [Architecture](architecture.md) — how the code is organized (for contributors).
- [Theory & citation](theory.md) — the method, the derivation, and how to cite.
