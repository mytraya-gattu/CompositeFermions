// Jain-Kamilla projected composite-fermion wavefunction PsiProj.
//
// The Slater determinant is built from projected (multi-particle) orbitals: moving one
// particle changes the elementary symmetric polynomials of every particle, hence every column
// of slater_det — so there is NO rank-1 update here (see PsiUnproj for that).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "half_integer.hpp"
#include "jk_projection.hpp"
#include "spinor.hpp"
#include "symmetric_polynomials.hpp"

namespace cfs {

struct PsiProj {
    int two_Qstar;
    int p;
    int N;            // system size
    LMList l_m_list;  // (two_L, two_Lz) per orbital
    int two_Lmax;

    int num_orbitals;
    int num_mu;   // 2 Lmax + 1
    int b;        // ESP order = Lmax - Q*
    int num_deg;  // b + 1

    Eigen::MatrixXcd fourier_tot;  // (num_orbitals*num_deg) × num_mu
    Eigen::VectorXcd U, V;
    Eigen::MatrixXcd exp_theta;    // num_mu × N
    Eigen::MatrixXcd exp_phi;      // num_orbitals × N
    Eigen::MatrixXd dist_matrix;   // (N-1) × N
    Eigen::MatrixXcd u_v_ratio;    // (N-1) × N
    Eigen::MatrixXcd esp;          // num_deg × N
    Eigen::VectorXd reg_coeffs;    // length b
    Eigen::MatrixXcd wigner_d;     // (num_orbitals*num_deg) × N
    std::vector<Eigen::MatrixXcd> wigner_D;  // N matrices, each num_orbitals × num_deg
    cdouble jastrow_factor_log;
    Eigen::MatrixXcd slater_det;   // num_orbitals × N

    PsiProj(int two_Qstar_, int p_, int N_, const LMList& l_m_list_)
        : two_Qstar(two_Qstar_), p(p_), N(N_), l_m_list(l_m_list_) {
        two_Lmax = 0;
        for (auto& lm : l_m_list) two_Lmax = std::max(two_Lmax, lm.first);
        num_orbitals = static_cast<int>(l_m_list.size());
        num_mu = two_Lmax + 1;
        b = (two_Lmax - two_Qstar) / 2;
        num_deg = b + 1;

        fourier_tot = Eigen::MatrixXcd::Zero(num_orbitals * num_deg, num_mu);
        // Group orbitals by total angular momentum and fill the Fourier block for each.
        std::vector<int> uniqueL;
        for (auto& lm : l_m_list)
            if (std::find(uniqueL.begin(), uniqueL.end(), lm.first) == uniqueL.end())
                uniqueL.push_back(lm.first);
        for (int two_L : uniqueL) {
            std::vector<int> idxs, lzs;
            for (int o = 0; o < num_orbitals; ++o)
                if (l_m_list[o].first == two_L) {
                    idxs.push_back(o);
                    lzs.push_back(l_m_list[o].second);
                }
            fill_fourier_block(fourier_tot, num_orbitals, two_Lmax, two_Qstar, N, two_L, idxs,
                               lzs);
        }

        U = Eigen::VectorXcd::Zero(N);
        V = Eigen::VectorXcd::Zero(N);
        exp_theta = Eigen::MatrixXcd::Zero(num_mu, N);
        exp_phi = Eigen::MatrixXcd::Zero(num_orbitals, N);
        dist_matrix = Eigen::MatrixXd::Zero(N - 1, N);
        u_v_ratio = Eigen::MatrixXcd::Zero(N - 1, N);
        esp = Eigen::MatrixXcd::Zero(num_deg, N);
        wigner_d = Eigen::MatrixXcd::Zero(num_orbitals * num_deg, N);
        wigner_D.assign(N, Eigen::MatrixXcd::Zero(num_orbitals, num_deg));
        slater_det = Eigen::MatrixXcd::Zero(num_orbitals, N);
        jastrow_factor_log = cdouble(0.0, 0.0);

        reg_coeffs = Eigen::VectorXd::Zero(b);
        for (int i = 1; i <= b; ++i)
            reg_coeffs(i - 1) = static_cast<double>(i) / ((N - 1) - i + 1);
    }

    // --- helpers shared by full and single-particle updates ---

    void compute_esp() {
        for (int e = 0; e < N; ++e)
            get_symmetric_polynomials(esp.col(e).data(), u_v_ratio.col(e).data(), N - 1, b,
                                      reg_coeffs.data());
    }

    void reshape_and_phase(int iter) {
        Eigen::Map<const Eigen::MatrixXcd> wd(wigner_d.col(iter).data(), num_orbitals, num_deg);
        wigner_D[iter] = wd;
        wigner_D[iter].array().colwise() *= exp_phi.col(iter).array();
    }

    // --- full update from all positions ---
    void update(const Eigen::VectorXd& theta, const Eigen::VectorXd& phi) {
        for (int j = 0; j < num_mu; ++j) {
            const double mu = -0.5 * two_Lmax + j;
            for (int prt = 0; prt < N; ++prt)
                exp_theta(j, prt) = std::polar(1.0, -mu * theta(prt));
        }
        for (int o = 0; o < num_orbitals; ++o) {
            const double lz = 0.5 * l_m_list[o].second;
            for (int prt = 0; prt < N; ++prt)
                exp_phi(o, prt) = std::polar(1.0, lz * phi(prt));
        }

        for (int prt = 0; prt < N; ++prt) {
            auto uv = u_v_generator(theta(prt), phi(prt));
            U(prt) = uv.first;
            V(prt) = uv.second;
        }

        jastrow_factor_log = cdouble(0.0, 0.0);
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const cdouble du = std::conj(U(i)) * U(j) + std::conj(V(i)) * V(j);
                const cdouble dv = U(i) * V(j) - V(i) * U(j);
                u_v_ratio(j - 1, i) = du / dv;
                u_v_ratio(i, j) = -std::conj(du) / dv;
                jastrow_factor_log += static_cast<double>(p) * std::log(dv);
                dist_matrix(j - 1, i) = 2.0 * std::abs(dv);
                dist_matrix(i, j) = dist_matrix(j - 1, i);
            }
        }

        compute_esp();
        wigner_d.noalias() = fourier_tot * exp_theta;
        for (int prt = 0; prt < N; ++prt) reshape_and_phase(prt);
        for (int prt = 0; prt < N; ++prt) slater_det.col(prt).noalias() = wigner_D[prt] * esp.col(prt);
    }

    // --- single-particle update (particle `iter` moves to (theta, phi)) ---
    void update(double theta, double phi, int iter) {
        for (int j = 0; j < num_mu; ++j) {
            const double mu = -0.5 * two_Lmax + j;
            exp_theta(j, iter) = std::polar(1.0, -mu * theta);
        }
        for (int o = 0; o < num_orbitals; ++o)
            exp_phi(o, iter) = std::polar(1.0, 0.5 * l_m_list[o].second * phi);

        auto uv = u_v_generator(theta, phi);
        const cdouble unew = uv.first, vnew = uv.second;

        for (int i = 0; i < N; ++i) {
            if (i < iter) {
                const cdouble dv_old = U(i) * V(iter) - V(i) * U(iter);
                const cdouble dv_new = U(i) * vnew - V(i) * unew;
                const cdouble du_new = std::conj(U(i)) * unew + std::conj(V(i)) * vnew;
                u_v_ratio(iter - 1, i) = du_new / dv_new;
                u_v_ratio(i, iter) = -std::conj(du_new) / dv_new;
                jastrow_factor_log += static_cast<double>(p) * std::log(dv_new / dv_old);
                dist_matrix(iter - 1, i) = 2.0 * std::abs(dv_new);
                dist_matrix(i, iter) = dist_matrix(iter - 1, i);
            } else if (i > iter) {
                const cdouble dv_old = -U(i) * V(iter) + V(i) * U(iter);
                const cdouble dv_new = -U(i) * vnew + V(i) * unew;
                const cdouble du_new = U(i) * std::conj(unew) + V(i) * std::conj(vnew);
                u_v_ratio(i - 1, iter) = du_new / dv_new;
                u_v_ratio(iter, i) = -std::conj(du_new) / dv_new;
                jastrow_factor_log += static_cast<double>(p) * std::log(dv_new / dv_old);
                dist_matrix(i - 1, iter) = 2.0 * std::abs(dv_new);
                dist_matrix(iter, i) = dist_matrix(i - 1, iter);
            }
        }
        U(iter) = unew;
        V(iter) = vnew;

        compute_esp();
        wigner_d.col(iter).noalias() = fourier_tot * exp_theta.col(iter);
        reshape_and_phase(iter);
        for (int e = 0; e < N; ++e) slater_det.col(e).noalias() = wigner_D[e] * esp.col(e);
    }

    void copy_from(const PsiProj& o) {
        dist_matrix = o.dist_matrix;
        exp_theta = o.exp_theta;
        exp_phi = o.exp_phi;
        U = o.U;
        V = o.V;
        jastrow_factor_log = o.jastrow_factor_log;
        slater_det = o.slater_det;
        esp = o.esp;
        u_v_ratio = o.u_v_ratio;
        wigner_d = o.wigner_d;
        wigner_D = o.wigner_D;
    }

    void copy_from(const PsiProj& o, int iter) {
        dist_matrix = o.dist_matrix;
        exp_theta.col(iter) = o.exp_theta.col(iter);
        exp_phi.col(iter) = o.exp_phi.col(iter);
        U(iter) = o.U(iter);
        V(iter) = o.V(iter);
        jastrow_factor_log = o.jastrow_factor_log;
        slater_det = o.slater_det;
        esp = o.esp;
        u_v_ratio = o.u_v_ratio;
        wigner_d.col(iter) = o.wigner_d.col(iter);
        wigner_D[iter] = o.wigner_D[iter];
    }
};

}  // namespace cfs
