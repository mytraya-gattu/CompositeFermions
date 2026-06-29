# Physics background

This page is a gentle, self-contained primer on the physics the package implements. It assumes
familiarity with quantum mechanics but not with the fractional quantum Hall effect (FQHE). If
you just want to run code, skim the [notation](@ref Notation-in-code) table and jump to the
[Tutorials](tutorials/01_ground_state.md).

## The lowest Landau level on a sphere

In a strong magnetic field, two-dimensional electrons occupy quantized **Landau levels**. At
low filling only the **lowest Landau level (LLL)** matters, and the kinetic energy is quenched —
all the physics is in the electron–electron interaction projected into the LLL.

Haldane's **sphere** geometry places ``N`` electrons on a sphere with a magnetic monopole of
strength ``Q`` at its centre, so the total flux through the surface is ``2Q`` flux quanta
(``2Q`` is a non-negative integer by Dirac quantization). Single-particle states are **monopole
harmonics** ``Y_{Q,l,m}``, labelled by angular momentum ``l = Q, Q+1, \dots`` (the LLL is
``l = Q``) and ``m = -l,\dots,l``. The sphere has no edge, which makes it ideal for extracting
bulk properties.

Each electron's position is a point on the sphere ``(\theta,\phi)``, encoded by a **spinor**

```math
u = \cos(\theta/2)\,e^{i\phi/2}, \qquad v = \sin(\theta/2)\,e^{-i\phi/2}.
```

## Composite fermions

The **composite fermion (CF)** is the emergent particle of the FQHE: an electron bound to an
even number ``p`` of vortices (quantized flux). The bound vortices screen part of the magnetic
field, so composite fermions experience a **reduced effective flux**

```math
2Q^{\star} = 2Q - p(N-1).
```

Composite fermions then fill ``n`` of their own effective Landau levels (called
**``\Lambda``-levels**), which produces the famous Jain sequence of filling fractions

```math
\nu = \frac{n}{p\,n + 1}.
```

The CF wavefunction is a Slater determinant of CF orbitals times a **Jastrow factor** that
attaches the ``p`` vortices:

```math
\Psi_{\nu} = \mathcal{P}_{\mathrm{LLL}}\;\Phi_n\,\prod_{j<k}(u_j v_k - u_k v_j)^{p},
```

where ``\Phi_n`` is the Slater determinant of ``n`` filled levels at effective flux ``2Q^\star``,
and ``\mathcal{P}_{\mathrm{LLL}}`` projects the product back into the lowest Landau level.

!!! note "The meaning of `p` in this package"
    Throughout the code, `p` is the **power of the Jastrow factor** — the total number of vortices
    (flux quanta) attached to each electron. It is even: `p = 2` attaches one vortex *pair*
    (the textbook ``2\tilde p`` flux with ``\tilde p = 1``), `p = 4` attaches two pairs, and so
    on. In terms of `p` the filling is ``\nu = n/(pn+1)``:

    | `p` | `n` | ``\nu`` |
    |---|---|---|
    | 2 | 1 | 1/3 |
    | 2 | 2 | 2/5 |
    | 2 | 3 | 3/7 |
    | 4 | 1 | 1/5 |
    | 4 | 2 | 2/9 |

    So the ``\nu = 1/3`` Laughlin state is `p = 2`, **not** `p = 1`.

## The Jain–Kamilla projection

Carrying out ``\mathcal{P}_{\mathrm{LLL}}`` exactly is exponentially expensive. The
**Jain–Kamilla (JK) projection** makes it tractable by projecting one electron's orbital at a
time. This package uses the quaternion/rotation reformulation of JK projection
([arXiv:2412.09670](https://arxiv.org/abs/2412.09670)), in which each projected orbital becomes

```math
\frac{\mathcal{P}_{\mathrm{LLL}}\, Y_{Q^\star,l,m}(\Omega_i)\, J_i}{J_i}
= \sum_{m'} N^{l}_{m',Q^\star,Q_1}\,\mathcal{D}^{l}_{m,m'}(\Omega_i)\,\tilde e_{m'-Q^\star}(X_i),
\qquad J_i = \prod_{j\neq i}(u_i v_j - u_j v_i),
```

a sum over a **Wigner-D matrix** ``\mathcal{D}^l_{m,m'}`` times **elementary symmetric
polynomials** ``\tilde e_r`` of the "vortex ratios" ``X_i``. This is the central object the code
evaluates — far cheaper and more numerically stable than the traditional mixed-derivative
projection.

The projection binds a **single vortex pair** per orbital, so the JK angular momentum is
``Q_1 = (N-1)/2``.

## Reaching higher fillings: the outer Jastrow

You might expect more vortices (larger ``p``) to require binding more pairs inside the
projection. It does not. The projection always binds a **single** vortex pair, and the remaining
vortices come from the Jastrow power as a holomorphic prefactor that stays in the LLL
automatically:

```math
\Psi = \Phi_1^{\,p-2}\; \mathcal{P}_{\mathrm{LLL}}\!\left[\Phi_n\,\Phi_1^2\right],
\qquad \Phi_1 = \prod_{j<k}(u_j v_k - u_k v_j).
```

So nothing special is needed for higher fillings — just pass the desired total vortex count `p`.
For example ``\nu = 2/9`` (``n = 2``, two pairs ``\Rightarrow p = 4``) is simply
`Ψproj(Qstar, 4, N, l_m_list)`. The
[higher-fillings tutorial](tutorials/03_higher_fillings.md) demonstrates this.

!!! note "Why a single bound pair?"
    Binding more than one pair inside the projection is unnecessary (the outer Jastrow reaches
    every filling) and numerically fragile. See [Theory & citation](theory.md) for the analysis.

## Notation in code

| Symbol | Code | Meaning |
|---|---|---|
| ``Q^\star`` | `Qstar::Rational{Int64}` | effective monopole strength, ``Q^\star = Q - p(N-1)/2`` |
| ``p``       | `p::Int64` | Jastrow power ``\prod(u_jv_k-u_kv_j)^p``; even (= 2× vortex pairs); ``\nu = n/(pn+1)`` |
| ``(l, m)``  | `l_m_list::Vector{NTuple{2,Rational}}` | occupied ``(L, L_z)`` orbitals (the filled ``\Lambda``-levels) |
| ``n``       | `n` | number of filled ``\Lambda``-levels |
| ``N``       | `system_size` | number of composite fermions |
| ``(u,v)``   | `U`, `V` | spinor coordinates of each particle |

The convenience builders [`cf_ground_state_lm`](@ref), [`cf_quasihole_lm`](@ref), and
[`cf_quasiparticle_lm`](@ref) construct `(Qstar, l_m_list)` for the standard states so you
rarely set them by hand.
