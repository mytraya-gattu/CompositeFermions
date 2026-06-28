// Jain-Kamilla projection coefficient and the Fourier-matrix block builder.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "half_integer.hpp"
#include "j_y_eigenstates.hpp"

namespace cfs {

// N^L_{m,Q*,Q1} (without the sqrt((2L+1)/4π), which is applied per block). All arguments are
// in two-units; the integer binomial arguments are formed as (sum±)/2.
inline double projection_coeff(int two_L, int two_Qstar, int two_Q1, int two_m) {
    const long Lm_Q = (two_L - two_Qstar) / 2;      // L - Q*
    const long m_Q = (two_m - two_Qstar) / 2;       // m - Q*
    const long L_Q = (two_L + two_Qstar) / 2;       // L + Q*
    const long L_m = (two_L + two_m) / 2;           // L + m
    return std::exp(log_binomial(two_Q1, Lm_Q) + log_binomial(Lm_Q, m_Q) +
                    0.5 * log_binomial(two_L, L_Q) -
                    log_binomial(two_Q1 + L_Q + 1, Lm_Q) -
                    0.5 * log_binomial(two_L, L_m));
}

// Fill the rows of `fourier_tot` (shape: norb*num_deg_max × num_mu_max, reshaped layout where
// row = orbital + norb*deg, col = global μ index) for all orbitals that share total angular
// momentum two_L. `orbital_indices` are their (global) row indices, `two_Lz_vals` their 2Lz.
inline void fill_fourier_block(Eigen::MatrixXcd& fourier_tot, int norb, int two_Lmax,
                               int two_Qstar, int N, int two_L,
                               const std::vector<int>& orbital_indices,
                               const std::vector<int>& two_Lz_vals) {
    const int two_Q1 = N - 1;
    const double normL = std::sqrt((two_L + 1) / (4.0 * M_PI));
    const int mu_offset = (two_Lmax - two_L) / 2;  // first global μ column for this L

    for (int two_Lzp = two_Qstar; two_Lzp <= two_L; two_Lzp += 2) {
        const int deg = (two_Lzp - two_Qstar) / 2;
        const double sign = (deg % 2 == 0) ? 1.0 : -1.0;  // (-1)^(Lzprime - Q*)
        const double coeff = projection_coeff(two_L, two_Qstar, two_Q1, two_Lzp) * sign;

        for (std::size_t a = 0; a < orbital_indices.size(); ++a) {
            const int row = orbital_indices[a] + norb * deg;
            const int two_Lz = two_Lz_vals[a];
            for (int two_mu = -two_L; two_mu <= two_L; two_mu += 2) {
                const int col = mu_offset + m_index(two_mu, two_L);
                fourier_tot(row, col) = jy_table(two_L, two_mu, two_Lzp, two_Lz) * coeff * normL;
            }
        }
    }
}

}  // namespace cfs
