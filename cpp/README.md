# CFsOnSphere (C++ core sampler)

A native C++ port of the core of the Julia `CFsOnSphere` package: Jain–Kamilla **projected**
(`PsiProj`) and **unprojected** (`PsiUnproj`) composite-fermion wavefunctions on the sphere,
with a Metropolis–Hastings–Gibbs Monte Carlo driver. Intended for FQH practitioners who work
in C++.

Header-only (depends only on [Eigen](https://eigen.tuxfamily.org)); built as a CMake
`INTERFACE` library with example samplers and a test suite.

## Build

```sh
cd cpp
cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew   # point at your Eigen if needed
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the examples:

```sh
./build/sampler_single_state 10 2 1   # projected ν = 2/5, N = 10
./build/sampler_unprojected  9 3 1    # unprojected ν = 1/3, N = 9 (Sherman–Morrison)
```

## What's included

- `PsiProj` — projected CF Slater determinant with a general `jk_type` (1, 2, 3, …); the
  Jastrow power carried through the LLL projection, so `Q₁ = jk_type·(N−1)/2` and the ESP roots
  have multiplicity `jk_type`. `jk_type = 1` is the standard projection.
- `PsiUnproj` — unprojected CF state `det[Y_{Q*,l,m}] · ∏(uᵢvⱼ−uⱼvᵢ)^p` from single-particle
  monopole-harmonic orbitals, with **Sherman–Morrison** inverse tracking
  (`initialize_inverse` / `slater_det_ratio` / `update_inverse`) — valid here because moving one
  particle changes only one Slater column.
- `build_extended_slater` for fixed quasihole/quasiparticle orbital columns;
  `cf_ground_state_lm` / `cf_quasihole_lm` / `cf_quasiparticle_lm` λ-level builders;
  `proposal`, `gibbs_thermalization`, `update_density`, `calculate_ll`,
  `get_symmetric_polynomials`, `legendre_polynomials`.

**Not ported (deferred):** `Ψparton`, `ΨoneLL`, `construct_det_ratios`, and JLD2 I/O (output
here is plain CSV via `io.hpp`).

## Conventions / caveats

- **Half-integer angular momenta are stored doubled.** Every `Q*, L, Lz, μ` is an `int` equal
  to *twice* its value (`two_L = 2L`). Orbital lists are `LMList = vector<pair<int,int>>` of
  `(two_L, two_Lz)`. The λ-level builders return the signed `two_Qstar`.
- **No bit-exact MCMC reproducibility vs Julia.** The RNG (`std::mt19937_64`) differs from
  Julia's, so sampled chains agree only *statistically*. The deterministic wavefunction values
  do match: the test suite cross-checks `slater_det`, ESP, `projection_coeff`, and the Jastrow
  factor against Julia reference CSVs to ≤ 1e-12 (most at machine precision).
- **`log_det` imaginary part** carries a 2πi branch ambiguity (as in Julia); only its real part
  is physically meaningful and used by the sampler.

## Validation

- `test_cfsonsphere` — native structural tests (ESP branch agreement, full-vs-incremental
  consistency for `PsiProj`/`PsiUnproj`, Sherman–Morrison correctness, non-square
  `build_extended_slater`, proposal isotropy, jk_type = 2).
- `test_reference` — cross-checks against Julia. Regenerate the reference CSVs with
  `julia +lts --project=. cpp/reference/dump_reference.jl` from the repo root.
