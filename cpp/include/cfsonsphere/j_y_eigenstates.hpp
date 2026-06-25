// Eigenstates of J_y in the J_z basis, used to build Wigner-d / monopole-harmonic elements.
//
// J_y is Hermitian with non-degenerate eigenvalues -L..L; we eigensolve and keep the
// eigenvector matrix V with columns sorted by ascending eigenvalue (so column
// m_index(2μ) ↔ eigenvalue μ), matching Julia's `eigen` (which sorts ascending by default).
// The table entry V[m1,μ]·conj(V[m2,μ]) is invariant under the per-column eigenvector phase,
// so C++/Eigen and Julia/LAPACK agree exactly despite differing phase conventions.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <map>

#include "half_integer.hpp"

namespace cfs {

// Cached eigenvector matrix of J_y at total angular momentum two_L (= 2L).
inline const Eigen::MatrixXcd& j_y_eigenvectors(int two_L) {
    static std::map<int, Eigen::MatrixXcd> cache;
    auto it = cache.find(two_L);
    if (it != cache.end()) {
        return it->second;
    }

    const int dim = multiplet_size(two_L);  // 2L+1
    const double l = 0.5 * two_L;
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);

    // <m+1|J_y|m> = sqrt(l(l+1)-m(m+1))/(2i) = -i/2 sqrt(...) on the subdiagonal; fill the
    // Hermitian conjugate on the superdiagonal so the matrix is exactly Hermitian.
    for (int two_m = -two_L; two_m <= two_L - 2; two_m += 2) {
        const double m = 0.5 * two_m;
        const int idx = m_index(two_m, two_L);
        const double val = std::sqrt(l * (l + 1.0) - m * (m + 1.0));
        M(idx + 1, idx) = cdouble(0.0, -0.5 * val);
        M(idx, idx + 1) = cdouble(0.0, 0.5 * val);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    auto res = cache.emplace(two_L, es.eigenvectors());
    return res.first->second;
}

// d[(μ, m1, m2)] = V[m1, μ] * conj(V[m2, μ]).
inline cdouble jy_table(int two_L, int two_mu, int two_m1, int two_m2) {
    const Eigen::MatrixXcd& V = j_y_eigenvectors(two_L);
    const int col = m_index(two_mu, two_L);
    return V(m_index(two_m1, two_L), col) * std::conj(V(m_index(two_m2, two_L), col));
}

}  // namespace cfs
