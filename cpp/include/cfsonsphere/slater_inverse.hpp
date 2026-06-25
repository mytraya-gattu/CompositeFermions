// Sherman-Morrison maintenance of the Slater inverse for PsiUnproj, and an extended-Slater
// helper for fixed quasihole/quasiparticle orbital columns.
//
// These apply ONLY to PsiUnproj (single-particle orbitals): moving one particle changes one
// column of slater_det, so the inverse updates by a rank-1 formula. They are intentionally not
// provided for PsiProj, where a single move changes every column.
#pragma once

#include <Eigen/Dense>

#include "half_integer.hpp"
#include "projected_wavefunction.hpp"
#include "unprojected_wavefunction.hpp"

namespace cfs {

// inv(slater_det); call once after the first full update (requires a square slater_det).
inline void initialize_inverse(PsiUnproj& psi) {
    psi.slater_det_inv = psi.slater_det.inverse();
}

// det(S_next)/det(S_current) for a move of particle `iter` (O(N), unconjugated dot of the
// maintained inverse row with the new column).
inline cdouble slater_det_ratio(const PsiUnproj& cur, const PsiUnproj& nxt, int iter) {
    cdouble dr(0.0, 0.0);
    for (int k = 0; k < cur.N; ++k) {
        dr += cur.slater_det_inv(iter, k) * nxt.slater_det(k, iter);
    }
    return dr;
}

// Rank-1 Sherman-Morrison update of cur.slater_det_inv for an accepted move of particle
// `iter` to `nxt` (O(N^2)). Call BEFORE copy_from(cur, nxt, iter). `temp` is N-length scratch.
inline void update_inverse(PsiUnproj& cur, const PsiUnproj& nxt, int iter, cdouble det_ratio,
                           Eigen::VectorXcd& temp) {
    Eigen::MatrixXcd& Sinv = cur.slater_det_inv;
    temp.noalias() = Sinv * nxt.slater_det.col(iter);
    temp(iter) -= cdouble(1.0, 0.0);
    const cdouble invdr = cdouble(1.0, 0.0) / det_ratio;

    // Sinv -= (temp ⊗ Sinv.row(iter)) / det_ratio (unconjugated). Update row `iter` last so
    // the other rows read its original entries.
    const Eigen::RowVectorXcd rowiter = Sinv.row(iter);
    const int n = static_cast<int>(Sinv.rows());
    for (int a = 0; a < n; ++a) {
        if (a == iter) continue;
        Sinv.row(a) -= (temp(a) * invdr) * rowiter;
    }
    Sinv.row(iter) -= (temp(iter) * invdr) * rowiter;
}

// Fill the square (N+k)×(N+k) matrix Sfull with the electron block (first N columns) and k
// fixed QH/QP orbital columns, and return its LU. Works for PsiProj and PsiUnproj.
template <class Psi>
inline Eigen::PartialPivLU<Eigen::MatrixXcd> build_extended_slater(
    Eigen::MatrixXcd& Sfull, const Psi& psi, const Eigen::MatrixXcd& qh_columns) {
    Sfull.leftCols(psi.N) = psi.slater_det;
    Sfull.rightCols(qh_columns.cols()) = qh_columns;
    return Eigen::PartialPivLU<Eigen::MatrixXcd>(Sfull);
}

}  // namespace cfs
