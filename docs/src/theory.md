# Theory & citation

## The method

The projection implemented here is the quaternion/rotation reformulation of the Jain–Kamilla
projection introduced in:

> M. Gattu, J. K. Jain, *Composite fermions on a sphere via quaternions / unlocking new regimes*,
> [arXiv:2412.09670](https://arxiv.org/abs/2412.09670).

In this formulation each lowest-Landau-level-projected composite-fermion orbital is a sum over a
Wigner-D matrix times elementary symmetric polynomials of the Jastrow "vortex ratios" — see
[Physics background](physics.md) for the working equations. This is both cheaper and more
numerically stable than the traditional mixed-derivative projection.

## The single-vortex-pair projection

The package binds a **single vortex pair** inside the Jain–Kamilla projection
(``Q_1 = (N-1)/2``), and reaches the full Jain sequence ``\nu = n/(2pn+1)`` through the outer
holomorphic Jastrow factor (set the global power to ``2p``; see
[Higher fillings](tutorials/03_higher_fillings.md)).

Binding more than one pair inside the projection is **mathematically valid at integer multiplicity
but unnecessary and fragile**: the outer-Jastrow construction reaches every filling using the
safe single-pair projection, while a higher multiplicity makes the elementary symmetric
polynomials develop higher-order self-poles that are well-behaved only at integer multiplicity
(and diverge for fractional values). The full derivation and the clustering analysis behind this
choice are kept in the repository:

- `derivation_jktype.tex` — the generalized projection derivation;
- `mathematica/` — the short-distance (clustering) analysis, with `README.md`.

## How to cite

If you use this package in published work, please cite the method paper:

```bibtex
@article{GattuJain2024,
  author  = {Gattu, Mytraya and Jain, Jainendra K.},
  title   = {Unlocking new regimes in the fractional quantum Hall effect},
  journal = {arXiv preprint},
  eprint  = {2412.09670},
  archivePrefix = {arXiv},
  year    = {2024},
}
```

and link to the software repository,
[github.com/mytraya-gattu/CompositeFermions](https://github.com/mytraya-gattu/CompositeFermions).
