# Mathematica clustering checks for JK-projected CF wavefunctions

Symbolic checks (N=4, disk) of the Jain–Kamilla projected composite-fermion
wavefunction and its short-distance (clustering) behaviour as two particles
coalesce, `z1 -> z2 + eps`, `eps -> 0`. These accompany `../derivation_jktype.tex`
and settle the `jk_type` ("multiple roots") question.

## Construction

Compact JK-projected orbital (disk):

    phi_{n,m}(z_i) = z_i^(n+m) * e_n( { 1/(z_i - z_j) : j != i }, each root repeated jk times )

with `jk = jk_type = p` (half the attached vortices). Full wavefunction

    psi = det[ phi ] * Prod_{i<j} (z_i - z_j)^(2 jk).

`e_n` over the jk-fold multiset is computed from the generating function
`[t^n] Prod_j (1 + t/(z_i-z_j))^jk`.

## Scripts

Run with `wolframscript -file <name>.wls`.

- **cluster.wls** — leading power of `eps` for the bare determinant (`detPow`)
  and the full `psi` (`psiPow = detPow + 2 jk`) over four occupation
  configurations and `jk = 1,2,3`. Validates C1 (all-LLL) = Laughlin `1/(2jk+1)`
  → `psiPow = 2jk+1`. Uses two generic numeric substitutions for robustness.

- **verify.wls** — proves the compact orbital equals the *true* `d^n` JK
  projection at the determinant level: `det[compact]/det[true]` is z-independent
  for aufbau configs (and z-dependent for a gappy config missing a lower
  Lambda level). Also shows the `1/(z1-z2)^2` double-pole coefficient in the
  true `d^2` projection equals `p(p-1)` (0 at jk=1 — why jk=1 is bit-exact —
  turning on for jk>=2).

- **frac.wls** — fractional `jk_type`. The continuation
  `[t^n] Prod (1+t a_j)^jk` exists for real `jk`, but `psi` *diverges /
  fails to vanish* at coincidence for non-integer `jk` (e.g. C3 in the window
  `jk in (1, 3/2)` gives `psiPow <= 0`). Integer `jk` is special: `J^jk` is a
  polynomial iff `jk` is an integer, which caps the per-pair self-pole.

- **fractrue.wls** — confirms `det[compact]/det[true]` stays constant even at
  fractional `jk` (e.g. 3/2, 5/2): the divergence is representation-independent,
  not an artifact of the ESP form.

## Take-away

`jk_type` must be a positive integer. At integer `jk` the projection is exact
(see `../derivation_jktype.tex`); fractional `jk` is a well-defined function but
not a valid LLL composite-fermion state (non-polynomial `J^jk`, and a
multivalued/anyonic Jastrow when `2 jk` is non-integer).
