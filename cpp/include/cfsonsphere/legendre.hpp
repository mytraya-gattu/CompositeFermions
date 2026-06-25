// Legendre polynomials by the standard recurrence.
#pragma once

#include <vector>

namespace cfs {

// Fill res[0..kmax] with P_0(x) .. P_kmax(x). res must have size kmax+1.
inline void legendre_polynomials(std::vector<double>& res, double x, int kmax) {
    res[0] = 1.0;
    if (kmax == 0) {
        return;
    }
    res[1] = x;
    for (int k = 2; k <= kmax; ++k) {
        const double n = k - 1;
        res[k] = ((2.0 * n + 1.0) * x * res[k - 1] - n * res[k - 2]) / (n + 1.0);
    }
}

}  // namespace cfs
