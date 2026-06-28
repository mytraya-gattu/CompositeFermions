// Minimal downstream consumer of the header-only cfsonsphere library.
//
// Builds a Jain-Kamilla projected CF ground state (the single-bound-pair projection;
// higher fillings ν = n/(2p̃n+1) come from the global Jastrow power p = 2p̃) and evaluates
// log Ψ at one configuration. See ../sampler_single_state.cpp for a full Monte Carlo driver.
#include <cfsonsphere/cfsonsphere.hpp>

#include <cstdio>

using namespace cfs;

int main() {
    const int N = 10, n = 2, p = 2;            // ν = 2/5 ground state
    auto [twoQ, lm] = cf_ground_state_lm(N, n, p);
    const int Nsys = static_cast<int>(lm.size());

    PsiProj psi(twoQ, p, Nsys, lm);

    // One deterministic configuration on the sphere.
    Eigen::VectorXd theta(Nsys), phi(Nsys);
    for (int i = 0; i < Nsys; ++i) {
        theta(i) = M_PI * (i + 0.5) / Nsys;
        phi(i)   = 2.0 * M_PI * i / Nsys - M_PI;
    }
    psi.update(theta, phi);

    const auto logpsi = log_det(psi.slater_det) + psi.jastrow_factor_log;
    std::printf("N=%d n=%d p=%d  orbitals=%d  Re(logPsi)=%.6f\n",
                N, n, p, Nsys, logpsi.real());
    return 0;
}
