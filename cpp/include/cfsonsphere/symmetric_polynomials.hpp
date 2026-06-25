// Elementary symmetric polynomials (ESP), each root taken with multiplicity `mult`.
//
// Direct port of the Julia kernel: hand-optimised recurrences for mult == 1 and mult == 2
// (the common ν = n/(2n±1) cases) and a general regularized binomial convolution for
// mult >= 3. Degree d is iterated high → low so the lower-degree entries read on the RHS
// still hold their previous-root values. dest[d] (0-based) holds the degree-d polynomial.
#pragma once

#include <complex>

#include "half_integer.hpp"

namespace cfs {

// Exact binomial coefficient C(n, k) (mult is small, so this is cheap and exact in double).
inline double binomial_coeff(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    double r = 1.0;
    for (int i = 0; i < k; ++i) {
        r = r * static_cast<double>(n - i) / static_cast<double>(i + 1);
    }
    return r;
}

// Regularized ESP: dest[d] = e_d(roots, multiplicity mult) * prod_{i=1..d} reg[i-1].
// dest has length b+1; roots has length num_roots; reg has length >= b.
inline void get_symmetric_polynomials(cdouble* dest, const cdouble* roots, int num_roots,
                                      int b, const double* reg, int mult) {
    dest[0] = cdouble(1.0, 0.0);
    if (b == 0) {
        return;
    }
    if (b == 1) {
        cdouble s(0.0, 0.0);
        for (int i = 0; i < num_roots; ++i) {
            s += roots[i];
        }
        dest[1] = static_cast<double>(mult) * s * reg[0];
        return;
    }
    for (int d = 1; d <= b; ++d) {
        dest[d] = cdouble(0.0, 0.0);
    }

    if (mult == 1) {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const int upper = std::min(i + 1, b);
            for (int d = upper; d >= 1; --d) {
                dest[d] += r * dest[d - 1] * reg[d - 1];
            }
        }
    } else if (mult == 2) {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const cdouble tr = 2.0 * r;
            const cdouble r2 = r * r;
            const int upper = std::min(2 * (i + 1), b);
            for (int d = upper; d >= 2; --d) {
                dest[d] += tr * reg[d - 1] * dest[d - 1] +
                           r2 * reg[d - 1] * reg[d - 2] * dest[d - 2];
            }
            dest[1] += tr * reg[0] * dest[0];
        }
    } else {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const int upper = std::min((i + 1) * mult, b);
            for (int d = upper; d >= 1; --d) {
                cdouble acc = dest[d];
                cdouble rk(1.0, 0.0);
                double regprod = 1.0;
                const int kmax = std::min(d, mult);
                for (int k = 1; k <= kmax; ++k) {
                    rk *= r;
                    regprod *= reg[d - k];
                    acc += binomial_coeff(mult, k) * rk * regprod * dest[d - k];
                }
                dest[d] = acc;
            }
        }
    }
}

// Unregularized ESP: dest[d] = e_d(roots, multiplicity mult).
inline void get_symmetric_polynomials(cdouble* dest, const cdouble* roots, int num_roots,
                                      int b, int mult) {
    dest[0] = cdouble(1.0, 0.0);
    if (b == 0) {
        return;
    }
    if (b == 1) {
        cdouble s(0.0, 0.0);
        for (int i = 0; i < num_roots; ++i) {
            s += roots[i];
        }
        dest[1] = static_cast<double>(mult) * s;
        return;
    }
    for (int d = 1; d <= b; ++d) {
        dest[d] = cdouble(0.0, 0.0);
    }

    if (mult == 1) {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const int upper = std::min(i + 1, b);
            for (int d = upper; d >= 1; --d) {
                dest[d] += r * dest[d - 1];
            }
        }
    } else if (mult == 2) {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const cdouble tr = 2.0 * r;
            const cdouble r2 = r * r;
            const int upper = std::min(2 * (i + 1), b);
            for (int d = upper; d >= 2; --d) {
                dest[d] += tr * dest[d - 1] + r2 * dest[d - 2];
            }
            dest[1] += tr * dest[0];
        }
    } else {
        for (int i = 0; i < num_roots; ++i) {
            const cdouble r = roots[i];
            const int upper = std::min((i + 1) * mult, b);
            for (int d = upper; d >= 1; --d) {
                cdouble acc = dest[d];
                cdouble rk(1.0, 0.0);
                const int kmax = std::min(d, mult);
                for (int k = 1; k <= kmax; ++k) {
                    rk *= r;
                    acc += binomial_coeff(mult, k) * rk * dest[d - k];
                }
                dest[d] = acc;
            }
        }
    }
}

}  // namespace cfs
