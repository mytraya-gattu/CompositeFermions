# CFsOnSphere.jl

[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://mytraya-gattu.github.io/CompositeFermions/stable)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://mytraya-gattu.github.io/CompositeFermions/dev)
[![CI](https://github.com/mytraya-gattu/CompositeFermions/actions/workflows/CI.yml/badge.svg)](https://github.com/mytraya-gattu/CompositeFermions/actions/workflows/CI.yml)

**Composite-fermion wavefunctions on the sphere, for the fractional quantum Hall effect.**

CFsOnSphere builds Jain–Kamilla **projected** and **unprojected** composite-fermion and parton
wavefunctions on the Haldane sphere, and samples them with a Metropolis–Hastings–Gibbs Monte
Carlo walk to compute densities, pair correlations, energies, and overlaps. The projection uses
the quaternion/rotation reformulation of Jain–Kamilla projection
([arXiv:2412.09670](https://arxiv.org/abs/2412.09670)) — far cheaper and more numerically stable
than the traditional mixed-derivative approach.

📖 **[Read the documentation →](https://mytraya-gattu.github.io/CompositeFermions/)** — physics
primer, hands-on tutorials with figures, and the full API reference.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mytraya-gattu/CompositeFermions.git")
```

## Quickstart

```julia
using CFsOnSphere, Random, LinearAlgebra

# Filling ν = n/(pn+1); p is the Jastrow power (even). n=1, p=2  →  ν = 1/3.
N, n, p = 6, 1, 2
Qstar, l_m_list = cf_ground_state_lm(N, n, p)

ψ, ψ_next = Ψproj(Qstar, p, N, l_m_list), Ψproj(Qstar, p, N, l_m_list)
logpdf(ψ) = 2.0 * real(logdet(ψ.slater_det) + ψ.jastrow_factor_log)

rng = MersenneTwister(1)
θ, ϕ = rand_θ_ϕ_gen(rng, N)
θn, ϕn = copy(θ), copy(ϕ)
iter, σ, _, accept = gibbs_thermalization!(rng, ψ, ψ_next, θ, ϕ, θn, ϕn, π/sqrt(12), logpdf, 10_000)
@show accept
```

The [tutorials](https://mytraya-gattu.github.io/CompositeFermions/) turn this into full density,
pair-correlation, and energy measurements with figures, and cover quasiholes/quasiparticles,
higher fillings, unprojected Sherman–Morrison sampling, and parton states.

> **Note on `p`.** Throughout the package, `p` is the power of the Jastrow factor
> ∏(uⱼvₖ−uₖvⱼ)^p — the number of vortices attached per electron (even; `p = 2` is one vortex
> pair). The filling is **ν = n/(pn+1)**, so the ν = 1/3 Laughlin state is `p = 2`.

## Wavefunctions

| Type | Description | Fast rank-1 updates? |
|---|---|---|
| `Ψproj`   | Jain–Kamilla projected CF state | No (a move changes every column) |
| `Ψparton` | Jain–Kamilla projected parton state | No |
| `Ψunproj` | unprojected `det·Jastrow` (single-particle orbitals) | **Yes (Sherman–Morrison)** |
| `ΨoneLL`  | bare Jastrow (Laughlin) | n/a |

A native, header-only **C++ port** (Eigen + CMake) lives under [`cpp/`](cpp/).

## Citing

If you use this package in published work, please cite
[arXiv:2412.09670](https://arxiv.org/abs/2412.09670). A BibTeX entry is on the
[Theory & citation](https://mytraya-gattu.github.io/CompositeFermions/) page.
