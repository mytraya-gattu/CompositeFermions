// Single-particle monopole harmonics Y_{Q,l,m}(θ,φ) — the unprojected orbital base functions.
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "half_integer.hpp"
#include "j_y_eigenstates.hpp"

namespace cfs {

// Returns Y_{Q,l,m}(θ,φ) for all m = -L..L (ascending), as a length-(2L+1) vector indexed by
// m_index(2m). two_L = 2L, two_Q = 2Q.
inline Eigen::VectorXcd calculate_ll(int two_L, int two_Q, double theta, double phi) {
    const int dim = multiplet_size(two_L);
    const double l = 0.5 * two_L;
    const Eigen::MatrixXcd& V = j_y_eigenvectors(two_L);
    const int qrow = m_index(two_Q, two_L);

    // fourier(i, j) = V[Q, μ_j] * conj(V[m_i, μ_j]).
    Eigen::MatrixXcd fourier(dim, dim);
    for (int j = 0; j < dim; ++j) {
        const cdouble vq = V(qrow, j);
        for (int i = 0; i < dim; ++i) {
            fourier(i, j) = vq * std::conj(V(i, j));
        }
    }

    // exp_theta[j] = exp(-i μ θ), μ = -l + j.
    Eigen::VectorXcd exp_theta(dim);
    for (int j = 0; j < dim; ++j) {
        const double mu = -l + static_cast<double>(j);
        exp_theta(j) = std::polar(1.0, -mu * theta);
    }

    Eigen::VectorXcd tmp = fourier * exp_theta;

    const double norm = std::sqrt((two_L + 1) / (4.0 * M_PI));
    Eigen::VectorXcd out(dim);
    for (int i = 0; i < dim; ++i) {
        const double m = -l + static_cast<double>(i);
        out(i) = tmp(i) * std::polar(1.0, m * phi) * norm;
    }
    return out;
}

}  // namespace cfs
