// Elementary symmetric polynomials (ESP) used by the Jain-Kamilla projection.
//
// The JK projection binds a single vortex pair into each orbital, so the ESP are over the
// N-1 distinct ratio roots (multiplicity one) and Q1 = (N-1)/2. Direct port of the Julia
// kernel: degree d is iterated high -> low so the lower-degree entries read on the RHS still
// hold their previous-root values. dest[d] (0-based) holds the degree-d polynomial.
#pragma once

#include <complex>

#include "half_integer.hpp"

namespace cfs {

// Regularized ESP: dest[d] = e_d(roots) * prod_{i=1..d} reg[i-1].
// dest has length b+1; roots has length num_roots; reg has length >= b.
inline void get_symmetric_polynomials(cdouble* dest, const cdouble* roots, int num_roots,
                                      int b, const double* reg) {
    dest[0] = cdouble(1.0, 0.0);
    if (b == 0) {
        return;
    }
    if (b == 1) {
        cdouble s(0.0, 0.0);
        for (int i = 0; i < num_roots; ++i) {
            s += roots[i];
        }
        dest[1] = s * reg[0];
        return;
    }
    for (int d = 1; d <= b; ++d) {
        dest[d] = cdouble(0.0, 0.0);
    }

    for (int i = 0; i < num_roots; ++i) {
        const cdouble r = roots[i];
        const int upper = std::min(i + 1, b);
        for (int d = upper; d >= 1; --d) {
            dest[d] += r * dest[d - 1] * reg[d - 1];
        }
    }
}

// Unregularized ESP: dest[d] = e_d(roots).
inline void get_symmetric_polynomials(cdouble* dest, const cdouble* roots, int num_roots,
                                      int b) {
    dest[0] = cdouble(1.0, 0.0);
    if (b == 0) {
        return;
    }
    if (b == 1) {
        cdouble s(0.0, 0.0);
        for (int i = 0; i < num_roots; ++i) {
            s += roots[i];
        }
        dest[1] = s;
        return;
    }
    for (int d = 1; d <= b; ++d) {
        dest[d] = cdouble(0.0, 0.0);
    }

    for (int i = 0; i < num_roots; ++i) {
        const cdouble r = roots[i];
        const int upper = std::min(i + 1, b);
        for (int d = upper; d >= 1; --d) {
            dest[d] += r * dest[d - 1];
        }
    }
}

}  // namespace cfs
