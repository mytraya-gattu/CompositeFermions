// Complex log-determinant via LU (only the real part is used by the sampler; the imaginary
// part carries a 2πi branch ambiguity, as in Julia).
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "half_integer.hpp"

namespace cfs {

inline cdouble log_det(const Eigen::MatrixXcd& A) {
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(A);
    const Eigen::MatrixXcd& LU = lu.matrixLU();
    cdouble s(0.0, 0.0);
    for (int i = 0; i < A.rows(); ++i) {
        s += std::log(LU(i, i));
    }
    if (lu.permutationP().determinant() < 0) {
        s += cdouble(0.0, M_PI);  // permutation sign (affects imaginary part only)
    }
    return s;
}

}  // namespace cfs
