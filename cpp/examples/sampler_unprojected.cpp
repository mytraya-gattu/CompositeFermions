// Example: unprojected CF state (PsiUnproj) sampled with Sherman-Morrison inverse updates.
//
// Phase 1 thermalizes with the generic logdet-based driver (tunes σ); phase 2 samples with
// the O(N) ratio + O(N²) rank-1 inverse update.
//
// Build (from cpp/): cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew && cmake --build build
// Run:               ./build/sampler_unprojected [N] [n] [p] [num_therm] [num_steps] [seed]
#include <cfsonsphere/cfsonsphere.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cfs;

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 9;
    const int n = argc > 2 ? std::atoi(argv[2]) : 3;
    const int p = argc > 3 ? std::atoi(argv[3]) : 1;
    const long num_therm = argc > 4 ? std::atol(argv[4]) : 200000;
    const long num_steps = argc > 5 ? std::atol(argv[5]) : 1000000;
    const std::uint64_t seed = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 1;

    auto [twoQ, lm] = cf_ground_state_lm(N, n, p);
    const int Nsys = static_cast<int>(lm.size());

    PsiUnproj cur(twoQ, p, Nsys, lm);
    PsiUnproj nxt(twoQ, p, Nsys, lm);

    auto logpdf = [](const PsiUnproj& psi) {
        return 2.0 * (log_det(psi.slater_det) + psi.jastrow_factor_log).real();
    };

    Rng rng(seed);
    Eigen::VectorXd th, ph;
    rand_theta_phi(rng, Nsys, th, ph);
    Eigen::VectorXd th_n = th, ph_n = ph;

    // Phase 1: thermalize + tune σ with the generic (logdet) driver.
    double sigma = M_PI / std::sqrt(12.0);
    auto [it, sigma_t, dt_therm, acc_therm] =
        gibbs_thermalization(rng, cur, nxt, th, ph, th_n, ph_n, sigma, logpdf, num_therm);
    sigma = sigma_t;
    std::printf("thermalization: acceptance=%.3f  sigma=%.4f  dt=%.2fs\n", acc_therm, sigma, dt_therm);

    // Phase 2: Sherman-Morrison-accelerated sampling.
    initialize_inverse(cur);
    Eigen::VectorXcd temp(Nsys);

    const int n_theta = 200;
    std::vector<double> theta_mesh(n_theta);
    for (int i = 0; i < n_theta; ++i) theta_mesh[i] = std::acos(1.0 - 2.0 * i / (n_theta - 1));
    std::vector<double> density(n_theta - 1, 0.0);

    long naccept = 0;
    for (long mc = 1; mc <= num_steps; ++mc) {
        auto pr = proposal(rng, th(it), ph(it), sigma);
        nxt.update(pr.first, pr.second, it);
        const cdouble dr = slater_det_ratio(cur, nxt, it);
        const double dlog =
            2.0 * (std::log(dr) + nxt.jastrow_factor_log - cur.jastrow_factor_log).real();
        if (dlog >= std::log(rng.rand())) {
            th(it) = pr.first;
            ph(it) = pr.second;
            update_inverse(cur, nxt, it, dr, temp);
            cur.copy_from(nxt, it);
            ++naccept;
        } else {
            nxt.copy_from(cur, it);
        }
        update_density(theta_mesh, th, density);
        it = (it + 1) % Nsys;
    }

    for (auto& d : density) d /= num_steps;
    save_csv("density_unprojected.csv", density);
    std::printf("sampling: acceptance=%.3f  (wrote density_unprojected.csv)\n",
                static_cast<double>(naccept) / num_steps);
    return 0;
}
