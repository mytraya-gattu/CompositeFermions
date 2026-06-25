// Unprojected composite-fermion wavefunction PsiUnproj: det[Y_{Q*,l,m}(Ω_i)] · Jastrow^p.
//
// Built from single-particle monopole-harmonic orbitals, so moving one particle changes only
// one column of slater_det → the inverse can be tracked by Sherman-Morrison (slater_inverse.hpp).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "half_integer.hpp"
#include "j_y_eigenstates.hpp"
#include "spinor.hpp"

namespace cfs {

struct PsiUnproj {
    int two_Qstar;
    int p;
    int N;
    LMList l_m_list;
    int two_Lmax;
    int num_orbitals;
    int num_mu;

    Eigen::MatrixXcd fourier_tot;  // num_orbitals × num_mu
    Eigen::VectorXcd U, V;
    Eigen::MatrixXcd exp_theta;    // num_mu × N
    Eigen::MatrixXcd exp_phi;      // num_orbitals × N
    Eigen::MatrixXd dist_matrix;   // (N-1) × N
    Eigen::MatrixXcd slater_det;   // num_orbitals × N
    Eigen::MatrixXcd slater_det_inv;  // N × N (valid only when slater_det is square)
    cdouble jastrow_factor_log;

    PsiUnproj(int two_Qstar_, int p_, int N_, const LMList& l_m_list_)
        : two_Qstar(two_Qstar_), p(p_), N(N_), l_m_list(l_m_list_) {
        num_orbitals = static_cast<int>(l_m_list.size());
        two_Lmax = 0;
        for (auto& lm : l_m_list) two_Lmax = std::max(two_Lmax, lm.first);
        num_mu = two_Lmax + 1;

        fourier_tot = Eigen::MatrixXcd::Zero(num_orbitals, num_mu);
        for (int o = 0; o < num_orbitals; ++o) {
            const int two_L = l_m_list[o].first;
            const int two_Lz = l_m_list[o].second;
            const double normL = std::sqrt((two_L + 1) / (4.0 * M_PI));
            for (int two_mu = -two_L; two_mu <= two_L; two_mu += 2) {
                const int col = m_index(two_mu, two_Lmax);
                fourier_tot(o, col) = jy_table(two_L, two_mu, two_Qstar, two_Lz) * normL;
            }
        }

        U = Eigen::VectorXcd::Zero(N);
        V = Eigen::VectorXcd::Zero(N);
        exp_theta = Eigen::MatrixXcd::Zero(num_mu, N);
        exp_phi = Eigen::MatrixXcd::Zero(num_orbitals, N);
        dist_matrix = Eigen::MatrixXd::Zero(N - 1, N);
        slater_det = Eigen::MatrixXcd::Zero(num_orbitals, N);
        slater_det_inv = Eigen::MatrixXcd::Zero(N, N);
        jastrow_factor_log = cdouble(0.0, 0.0);
    }

    void fill_exp_columns(int prt, double theta, double phi) {
        for (int j = 0; j < num_mu; ++j) {
            const double mu = -0.5 * two_Lmax + j;
            exp_theta(j, prt) = std::polar(1.0, -mu * theta);
        }
        for (int o = 0; o < num_orbitals; ++o)
            exp_phi(o, prt) = std::polar(1.0, 0.5 * l_m_list[o].second * phi);
    }

    void update(const Eigen::VectorXd& theta, const Eigen::VectorXd& phi) {
        for (int prt = 0; prt < N; ++prt) {
            fill_exp_columns(prt, theta(prt), phi(prt));
            auto uv = u_v_generator(theta(prt), phi(prt));
            U(prt) = uv.first;
            V(prt) = uv.second;
        }
        slater_det = (fourier_tot * exp_theta).cwiseProduct(exp_phi);

        jastrow_factor_log = cdouble(0.0, 0.0);
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const cdouble dv = U(i) * V(j) - V(i) * U(j);
                jastrow_factor_log += static_cast<double>(p) * std::log(dv);
                dist_matrix(j - 1, i) = 2.0 * std::abs(dv);
                dist_matrix(i, j) = dist_matrix(j - 1, i);
            }
        }
    }

    void update(double theta, double phi, int iter) {
        auto uv = u_v_generator(theta, phi);
        const cdouble unew = uv.first, vnew = uv.second;
        const cdouble uold = U(iter), vold = V(iter);

        for (int i = 0; i < N; ++i) {
            if (i < iter) {
                const cdouble dv_old = U(i) * vold - V(i) * uold;
                const cdouble dv_new = U(i) * vnew - V(i) * unew;
                jastrow_factor_log += static_cast<double>(p) * std::log(dv_new / dv_old);
                dist_matrix(iter - 1, i) = 2.0 * std::abs(dv_new);
                dist_matrix(i, iter) = dist_matrix(iter - 1, i);
            } else if (i > iter) {
                const cdouble dv_old = -U(i) * vold + V(i) * uold;
                const cdouble dv_new = -U(i) * vnew + V(i) * unew;
                jastrow_factor_log += static_cast<double>(p) * std::log(dv_new / dv_old);
                dist_matrix(i - 1, iter) = 2.0 * std::abs(dv_new);
                dist_matrix(iter, i) = dist_matrix(i - 1, iter);
            }
        }
        U(iter) = unew;
        V(iter) = vnew;

        fill_exp_columns(iter, theta, phi);
        slater_det.col(iter) =
            (fourier_tot * exp_theta.col(iter)).cwiseProduct(exp_phi.col(iter));
    }

    void copy_from(const PsiUnproj& o) {
        U = o.U;
        V = o.V;
        exp_theta = o.exp_theta;
        exp_phi = o.exp_phi;
        dist_matrix = o.dist_matrix;
        slater_det = o.slater_det;
        slater_det_inv = o.slater_det_inv;
        jastrow_factor_log = o.jastrow_factor_log;
    }

    // Partial copy (only particle `iter` changed). Does NOT copy slater_det_inv: the
    // Sherman-Morrison helpers maintain the inverse on the accepted state.
    void copy_from(const PsiUnproj& o, int iter) {
        U(iter) = o.U(iter);
        V(iter) = o.V(iter);
        exp_theta.col(iter) = o.exp_theta.col(iter);
        exp_phi.col(iter) = o.exp_phi.col(iter);
        dist_matrix = o.dist_matrix;
        slater_det.col(iter) = o.slater_det.col(iter);
        jastrow_factor_log = o.jastrow_factor_log;
    }
};

}  // namespace cfs
