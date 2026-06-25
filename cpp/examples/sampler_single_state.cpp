// Example: density and pair-correlation of a projected CF Slater-determinant state (PsiProj)
// at filling ν = n/(2np+1), via Metropolis-Hastings-Gibbs Monte Carlo.
//
// Build (from cpp/): cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build build
// Run:               ./build/sampler_single_state [N] [n] [p] [num_therm] [num_steps] [seed]
#include <cfsonsphere/cfsonsphere.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cfs;

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 4;
    const int n = argc > 2 ? std::atoi(argv[2]) : 1;
    const int p = argc > 3 ? std::atoi(argv[3]) : 1;
    const long num_therm = argc > 4 ? std::atol(argv[4]) : 200000;
    const long num_steps = argc > 5 ? std::atol(argv[5]) : 1000000;
    const std::uint64_t seed = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 1;

    auto [twoQ, lm] = cf_ground_state_lm(N, n, p);
    const int Nsys = static_cast<int>(lm.size());

    PsiProj cur(twoQ, p, Nsys, lm);
    PsiProj nxt(twoQ, p, Nsys, lm);

    auto logpdf = [](const PsiProj& psi) {
        return 2.0 * (log_det(psi.slater_det) + psi.jastrow_factor_log).real();
    };

    Rng rng(seed);
    Eigen::VectorXd th, ph;
    rand_theta_phi(rng, Nsys, th, ph);
    Eigen::VectorXd th_n = th, ph_n = ph;

    double sigma = M_PI / std::sqrt(12.0);
    auto [it, sigma_t, dt_therm, acc_therm] =
        gibbs_thermalization(rng, cur, nxt, th, ph, th_n, ph_n, sigma, logpdf, num_therm);
    sigma = sigma_t;
    std::printf("thermalization: acceptance=%.3f  sigma=%.4f  dt=%.2fs\n", acc_therm, sigma, dt_therm);

    // Observables: equal-area polar density, and pair-distance distribution.
    const int n_theta = 200;
    std::vector<double> theta_mesh(n_theta);
    for (int i = 0; i < n_theta; ++i)
        theta_mesh[i] = std::acos(1.0 - 2.0 * i / (n_theta - 1));  // 0 .. π
    std::vector<double> density(n_theta - 1, 0.0);

    const int n_r = 500;
    const double dr = 2.0 / n_r;
    std::vector<double> pair_corr(n_r, 0.0);

    double logpdf_cur = logpdf(cur);
    long naccept = 0;
    for (long mc = 1; mc <= num_steps; ++mc) {
        auto pr = proposal(rng, th(it), ph(it), sigma);
        nxt.update(pr.first, pr.second, it);
        const double logpdf_nxt = logpdf(nxt);
        if (logpdf_nxt - logpdf_cur >= std::log(rng.rand())) {
            th(it) = pr.first;
            ph(it) = pr.second;
            cur.copy_from(nxt, it);
            logpdf_cur = logpdf_nxt;
            ++naccept;
        } else {
            nxt.copy_from(cur, it);
        }
        update_density(theta_mesh, th, density);
        for (int i = 0; i < Nsys - 1; ++i)
            for (int j = i + 1; j < Nsys; ++j) {
                int bin = static_cast<int>(cur.dist_matrix(j - 1, i) / dr);
                if (bin >= 0 && bin < n_r) pair_corr[bin] += 1.0;
            }
        it = (it + 1) % Nsys;
    }

    for (auto& d : density) d /= num_steps;
    for (auto& c : pair_corr) c /= num_steps;

    save_csv("density.csv", density);
    save_csv("pair_correlation.csv", pair_corr);
    std::printf("sampling: acceptance=%.3f  (wrote density.csv, pair_correlation.csv)\n",
                static_cast<double>(naccept) / num_steps);
    return 0;
}
