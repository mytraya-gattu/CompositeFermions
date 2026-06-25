// Monte Carlo: RNG wrapper, sphere proposal, ARM step-size adaptation, the generic Gibbs
// thermalization driver, and density binning.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "half_integer.hpp"

namespace cfs {

// Thin RNG wrapper providing Gaussian (randn) and uniform-(0,1) (rand) draws.
struct Rng {
    std::mt19937_64 gen;
    std::normal_distribution<double> nd{0.0, 1.0};
    std::uniform_real_distribution<double> ud{0.0, 1.0};
    explicit Rng(std::uint64_t seed) : gen(seed) {}
    double randn() { return nd(gen); }
    double rand() { return ud(gen); }
};

// Uniform random points on the unit sphere: θ ∈ [0,π], φ ∈ (-π,π].
inline void rand_theta_phi(Rng& rng, int n, Eigen::VectorXd& theta, Eigen::VectorXd& phi) {
    theta.resize(n);
    phi.resize(n);
    for (int i = 0; i < n; ++i) {
        const double x = rng.randn(), y = rng.randn(), z = rng.randn();
        const double r = std::sqrt(x * x + y * y + z * z);
        theta(i) = std::acos(std::clamp(z / r, -1.0, 1.0));
        phi(i) = std::atan2(y, x);
    }
}

// Propose a move by rotating along a great circle by a Gaussian angle δθ ~ N(0,σ) in a
// uniformly random tangent direction (isotropic about the current point → symmetric proposal).
inline std::pair<double, double> proposal(Rng& rng, double theta, double phi, double sigma) {
    const double st = std::sin(theta), ct = std::cos(theta);
    const double sp = std::sin(phi), cp = std::cos(phi);
    const double rx = st * cp, ry = st * sp, rz = ct;

    double wx = rng.randn(), wy = rng.randn(), wz = rng.randn();
    const double d = wx * rx + wy * ry + wz * rz;
    double ex = wx - d * rx, ey = wy - d * ry, ez = wz - d * rz;
    const double en = std::sqrt(ex * ex + ey * ey + ez * ez);
    if (en < 1e-300) return {theta, phi};
    ex /= en; ey /= en; ez /= en;

    const double a = rng.randn() * sigma;
    const double sa = std::sin(a), ca = std::cos(a);
    const double x = ca * rx + sa * ex, y = ca * ry + sa * ey, z = ca * rz + sa * ez;
    return {std::acos(std::clamp(z, -1.0, 1.0)), std::atan2(y, x)};
}

inline std::pair<double, double> arm_parameters(double ideal, double r) {
    double a = 1.0, b = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const double c = std::pow(a * ideal + b, r);
        a = std::pow(a * ideal + b, 1.0 / r) - c;
        b = c;
    }
    return {a, b};
}

inline double arm_scale_factor(double p, double p_i, double a, double b) {
    return std::log(a * p_i + b) / std::log(a * p + b);
}

// Generic Gibbs thermalization. `logpdf(psi)` returns a real scalar; tunes σ toward 50%
// acceptance. Returns (sampling_iter, σ, dt_seconds, acceptance_rate).
template <class Psi, class LogPdf>
std::tuple<int, double, double, double> gibbs_thermalization(
    Rng& rng, Psi& cur, Psi& nxt, Eigen::VectorXd& theta_cur, Eigen::VectorXd& phi_cur,
    Eigen::VectorXd& theta_nxt, Eigen::VectorXd& phi_nxt, double sigma_init, LogPdf logpdf,
    long num_therm) {
    const double target = 0.50;
    auto ab = arm_parameters(target, 3.0);
    const double a = ab.first, b = ab.second;

    long accepted = 0;
    double sigma = sigma_init;
    const int Nsys = cur.N;

    cur.update(theta_cur, phi_cur);
    nxt.copy_from(cur);
    double logpdf_cur = logpdf(cur);

    std::vector<long> schedule(25);
    for (int k = 0; k < 25; ++k)
        schedule[k] = std::lround(std::exp(
            std::log(10.0) + (std::log(static_cast<double>(num_therm)) - std::log(10.0)) * k / 24.0));

    int it = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long mc = 1; mc <= num_therm; ++mc) {
        auto pr = proposal(rng, theta_cur(it), phi_cur(it), sigma);
        theta_nxt(it) = pr.first;
        phi_nxt(it) = pr.second;
        nxt.update(theta_nxt(it), phi_nxt(it), it);
        const double logpdf_nxt = logpdf(nxt);

        if (logpdf_nxt - logpdf_cur >= std::log(rng.rand())) {
            theta_cur(it) = theta_nxt(it);
            phi_cur(it) = phi_nxt(it);
            cur.copy_from(nxt, it);
            logpdf_cur = logpdf_nxt;
            ++accepted;
        } else {
            theta_nxt(it) = theta_cur(it);
            phi_nxt(it) = phi_cur(it);
            nxt.copy_from(cur, it);
        }

        if (std::find(schedule.begin(), schedule.end(), mc) != schedule.end())
            sigma *= arm_scale_factor(static_cast<double>(accepted) / mc, target, a, b);

        it = (it + 1) % Nsys;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {it, sigma, dt, static_cast<double>(accepted) / num_therm};
}

// 1D polar-angle density histogram; θmesh sorted ascending.
inline void update_density(const std::vector<double>& theta_mesh, const Eigen::VectorXd& theta,
                           std::vector<double>& accumulated) {
    for (int i = 0; i < theta.size(); ++i) {
        const int lb = static_cast<int>(
            std::lower_bound(theta_mesh.begin(), theta_mesh.end(), theta(i)) - theta_mesh.begin());
        const int idx = lb - 1;
        if (idx >= 0 && idx < static_cast<int>(accumulated.size())) accumulated[idx] += 1.0;
    }
}

}  // namespace cfs
