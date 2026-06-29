# API reference

Everything exported by `CFsOnSphere`, grouped by topic. Internal helpers that carry docstrings
are included for completeness.

```@docs
CFsOnSphere
```

## Wavefunction types and updates

The four wavefunction families and their in-place update routines.

```@autodocs
Modules = [CFsOnSphere]
Pages   = ["projected_wavefunction.jl", "unprojected_wavefunction.jl"]
```

## Orbitals, projection, and polynomials

The single-particle monopole harmonics, the Jain–Kamilla projection coefficients, and the
elementary symmetric polynomials at the heart of the projection.

```@autodocs
Modules = [CFsOnSphere]
Pages   = ["monopole_harmonics.jl", "jk_projection_utilities.jl",
           "symmetric_polynomials.jl", "legendre_polynomials.jl",
           "calculate_j_y_eigenstates.jl", "spinor_coordinates.jl"]
```

## Slater inverse and excitations

Sherman–Morrison rank-1 inverse tracking for `Ψunproj`, and the extended Slater determinant for
quasihole/quasiparticle amplitudes.

```@autodocs
Modules = [CFsOnSphere]
Pages   = ["slater_inverse.jl"]
```

## Λ-level builders

Convenience constructors for the `(Qstar, l_m_list)` of the standard composite-fermion states.

```@autodocs
Modules = [CFsOnSphere]
Pages   = ["lambda_levels.jl"]
```

## Monte Carlo

The Metropolis–Hastings–Gibbs driver, proposal, step-size adaptation, and observable
accumulators.

```@autodocs
Modules = [CFsOnSphere]
Pages   = ["monte_carlo.jl"]
```
