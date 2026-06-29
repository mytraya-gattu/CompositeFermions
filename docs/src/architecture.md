# Architecture

How the package is organized, for contributors and anyone who wants to understand or extend the
internals.

## Source layout

`CFsOnSphere` is a single flat module (`src/CFsOnSphere.jl`) that `include`s small,
single-responsibility files in dependency order:

| File | Contents |
|---|---|
| `calculate_j_y_eigenstates.jl` | ``J_y`` eigenstate table (disk-memoised in `tempdir()`) |
| `spinor_coordinates.jl` | `u_v_generator(θ, ϕ)` |
| `monopole_harmonics.jl` | [`calculate_ll`](@ref) — single-particle harmonics |
| `symmetric_polynomials.jl` | [`get_symmetric_polynomials!`](@ref) (elementary symmetric polynomials) |
| `jk_projection_utilities.jl` | `projection_coeff`, `generate_fourier_matrices` |
| `legendre_polynomials.jl` | [`legendre_polynomials!`](@ref) |
| `projected_wavefunction.jl` | [`Ψproj`](@ref), [`Ψparton`](@ref): types, constructors, updates, `copy!` |
| `unprojected_wavefunction.jl` | [`Ψunproj`](@ref), [`ΨoneLL`](@ref): types, constructors, updates, `copy!` |
| `slater_inverse.jl` | Sherman–Morrison + [`build_extended_slater!`](@ref) (`Ψunproj`) |
| `lambda_levels.jl` | [`cf_ground_state_lm`](@ref) / [`cf_quasihole_lm`](@ref) / [`cf_quasiparticle_lm`](@ref) |
| `monte_carlo.jl` | proposal, thermalization, density accumulators |

External dependencies are deliberately light: `LinearAlgebra`, `Random`, `JLD2`,
`Combinatorics`, `Serialization`, `SpecialFunctions`. BLAS is pinned to one thread (multi-thread
BLAS hurts performance for the small dense determinants used here).

## Projected vs. unprojected: the key structural split

The two families differ in how a single particle move propagates:

- **[`Ψunproj`](@ref)** uses *single-particle* orbitals ``Y_{Q^\star,l,m}(\Omega_i)``. Moving
  particle `i` changes only **one column** of the Slater matrix, so the determinant ratio and
  inverse can be tracked by a **rank-1 Sherman–Morrison update** (`slater_inverse.jl`): O(N) for
  the acceptance ratio, O(N²) for the inverse refresh. This is the fast path.
- **[`Ψproj`](@ref) / [`Ψparton`](@ref)** use *multi-particle* projected orbitals: the elementary
  symmetric polynomials couple all particles, so moving one particle changes **every column**.
  There is no rank-1 shortcut; the determinant is recomputed via `logdet`. This is why the
  projected updates rebuild the symmetric polynomials and the whole Slater matrix.

## The shared update path

`Ψproj` and `Ψparton` share their update machinery (`_update_pairs!` and the
`update_wavefunction!` methods in `projected_wavefunction.jl`):

1. Update spinors, pairwise Jastrow log, vortex ratios, and chord distances.
2. Rebuild the elementary symmetric polynomials per particle ([`get_symmetric_polynomials!`](@ref)).
3. Contract the Fourier/Wigner-D matrices with the polynomials to form each Slater column.

The update routines are written to be **allocation-free** on the hot path (in-place `mul!`,
broadcast into preallocated buffers, in-place spinor arrays), so the projected sampler runs at
the same allocation profile as a hand-written loop.

## Adding a new wavefunction type

Mirror the existing pattern: define the mutable struct and constructor, implement
`update_wavefunction!` (full and single-particle-move variants) and `copy!` (full and partial),
and — if the orbitals are single-particle — the Sherman–Morrison interface. Add the type to the
`Union` in [`gibbs_thermalization!`](@ref) so the generic driver accepts it.

## C++ port

A native, header-only C++ core (Eigen + CMake) mirrors the projected and unprojected samplers
under `cpp/`. It is validated against the Julia reference to ``\le 10^{-12}``; see
[Validation](validation.md).
