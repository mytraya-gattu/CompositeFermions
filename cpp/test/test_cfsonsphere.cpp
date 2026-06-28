// Native structural tests for the CFsOnSphere C++ core (no Julia needed).
#include <cfsonsphere/cfsonsphere.hpp>

#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace cfs;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                           \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                 \
    do {                                                                      \
        ++g_checks;                                                           \
        double _d = std::abs((a) - (b));                                      \
        if (!(_d <= (tol))) {                                                 \
            std::printf("FAIL %s:%d  |%s - %s| = %.3e > %.3e\n", __FILE__,    \
                        __LINE__, #a, #b, _d, (double)(tol));                 \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static Eigen::VectorXd fixed_theta(int N, unsigned seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Eigen::VectorXd t(N);
    for (int i = 0; i < N; ++i) t(i) = std::acos(2.0 * u(g) - 1.0);
    return t;
}
static Eigen::VectorXd fixed_phi(int N, unsigned seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Eigen::VectorXd p(N);
    for (int i = 0; i < N; ++i) p(i) = 2.0 * M_PI * u(g) - M_PI;
    return p;
}

static void test_esp_branches() {
    std::mt19937_64 g(1);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int nroots = 12, b = 8;
    std::vector<cdouble> roots(nroots);
    for (auto& r : roots) r = cdouble(nd(g), nd(g));

    {
        const int Npart = 13;
        std::vector<double> reg(b);
        for (int i = 1; i <= b; ++i) reg[i - 1] = double(i) / ((Npart - 1) - i + 1);

        std::vector<cdouble> dest(b + 1);
        get_symmetric_polynomials(dest.data(), roots.data(), nroots, b, reg.data());

        // Ground truth: coefficients of prod_i (1 + r_i x), scaled by prod reg.
        std::vector<cdouble> g(nroots + 1, cdouble(0.0, 0.0));
        g[0] = cdouble(1.0, 0.0);
        int deg = 0;
        for (auto& r : roots) {
            for (int j = deg; j >= 0; --j) g[j + 1] += r * g[j];
            ++deg;
        }
        double acc = 1.0, maxd = 0;
        for (int d = 0; d <= b; ++d) {
            if (d >= 1) acc *= reg[d - 1];
            maxd = std::max(maxd, std::abs(dest[d] - g[d] * acc));
        }
        CHECK_NEAR(maxd, 0.0, 1e-9);
    }
}

static void test_psiproj_consistency() {
    auto [twoQ, lm] = cf_ground_state_lm(10, 2, 1);  // N=10, n=2 -> 10 orbitals
    const int N = 10;
    PsiProj psi(twoQ, 1, N, lm);
    auto th = fixed_theta(N, 11), ph = fixed_phi(N, 22);
    psi.update(th, ph);
    Eigen::MatrixXcd S0 = psi.slater_det;
    // moving each particle to its own position must be a no-op
    for (int i = 0; i < N; ++i) psi.update(th(i), ph(i), i);
    CHECK_NEAR((psi.slater_det - S0).cwiseAbs().maxCoeff(), 0.0, 1e-10);
}

static void test_psiunproj_orbitals() {
    auto [twoQ, lm] = cf_ground_state_lm(9, 3, 1);
    const int N = 9;
    CHECK((int)lm.size() == N);
    PsiUnproj psi(twoQ, 1, N, lm);
    auto th = fixed_theta(N, 3), ph = fixed_phi(N, 4);
    psi.update(th, ph);

    Eigen::MatrixXcd Sref(N, N);
    for (int i = 0; i < N; ++i)
        for (int o = 0; o < N; ++o) {
            Eigen::VectorXcd col = calculate_ll(lm[o].first, twoQ, th(i), ph(i));
            Sref(o, i) = col(m_index(lm[o].second, lm[o].first));
        }
    CHECK_NEAR((psi.slater_det - Sref).cwiseAbs().maxCoeff(), 0.0, 1e-10);

    // incremental vs full
    PsiUnproj psi2(twoQ, 1, N, lm);
    psi2.update(th, ph);
    Eigen::VectorXd th3 = th, ph3 = ph;
    Rng rng(99);
    for (int i = 0; i < N; ++i) {
        auto pr = proposal(rng, th3(i), ph3(i), 0.5);
        th3(i) = pr.first;
        ph3(i) = pr.second;
        psi2.update(th3(i), ph3(i), i);
    }
    PsiUnproj psi3(twoQ, 1, N, lm);
    psi3.update(th3, ph3);
    CHECK_NEAR((psi2.slater_det - psi3.slater_det).cwiseAbs().maxCoeff(), 0.0, 1e-10);
    CHECK_NEAR(std::abs((psi2.jastrow_factor_log - psi3.jastrow_factor_log).real()), 0.0, 1e-9);
}

static void test_sherman_morrison() {
    auto [twoQ, lm] = cf_ground_state_lm(9, 3, 2);
    const int N = 9;
    PsiUnproj cur(twoQ, 2, N, lm), nxt(twoQ, 2, N, lm);
    Rng rng(4);
    Eigen::VectorXd th, ph;
    rand_theta_phi(rng, N, th, ph);
    cur.update(th, ph);
    nxt.copy_from(cur);
    initialize_inverse(cur);
    CHECK_NEAR((cur.slater_det_inv - cur.slater_det.inverse()).cwiseAbs().maxCoeff(), 0.0, 1e-9);

    Eigen::VectorXcd temp(N);
    double sigma = 0.4;
    int it = 0, naccept = 0;
    for (int step = 1; step <= 3000; ++step) {
        auto pr = proposal(rng, th(it), ph(it), sigma);
        nxt.update(pr.first, pr.second, it);
        cdouble dr = slater_det_ratio(cur, nxt, it);
        if (step % 500 == 0) {
            cdouble dr_bf = log_det(nxt.slater_det) - log_det(cur.slater_det);
            CHECK_NEAR(std::abs(std::log(dr).real() - dr_bf.real()), 0.0, 1e-6);
        }
        double dlog = 2.0 * (std::log(dr) + nxt.jastrow_factor_log - cur.jastrow_factor_log).real();
        if (dlog >= std::log(rng.rand())) {
            th(it) = pr.first;
            ph(it) = pr.second;
            update_inverse(cur, nxt, it, dr, temp);
            cur.copy_from(nxt, it);
            ++naccept;
        } else {
            nxt.copy_from(cur, it);
        }
        it = (it + 1) % N;
    }
    CHECK(naccept > 0);
    CHECK_NEAR((cur.slater_det_inv - cur.slater_det.inverse()).cwiseAbs().maxCoeff(), 0.0, 1e-7);
}

static void test_extended_slater() {
    auto [twoQ, lm] = cf_quasiparticle_lm(9, 3, 1);  // N+1 = 10 orbitals
    const int N = 9;
    CHECK((int)lm.size() == N + 1);
    PsiUnproj psi(twoQ, 1, N, lm);
    CHECK(psi.slater_det.rows() == N + 1 && psi.slater_det.cols() == N);
    Rng rng(7);
    Eigen::VectorXd th, ph;
    rand_theta_phi(rng, N, th, ph);
    psi.update(th, ph);

    const double th_qh = 0.7, ph_qh = 1.1;
    Eigen::MatrixXcd qh(N + 1, 1);
    for (int o = 0; o <= N; ++o) {
        Eigen::VectorXcd col = calculate_ll(lm[o].first, twoQ, th_qh, ph_qh);
        qh(o, 0) = col(m_index(lm[o].second, lm[o].first));
    }

    Eigen::MatrixXcd Sfull(N + 1, N + 1);
    auto lu = build_extended_slater(Sfull, psi, qh);
    Eigen::MatrixXcd ref(N + 1, N + 1);
    ref.leftCols(N) = psi.slater_det;
    ref.rightCols(1) = qh;
    CHECK_NEAR((Sfull - ref).cwiseAbs().maxCoeff(), 0.0, 0.0);
    CHECK_NEAR(std::abs(log_det(Sfull).real() - log_det(ref).real()), 0.0, 1e-8);

    // amplitude: transpose(Sfull) \ e_end == row `end` of inv(Sfull)
    Eigen::VectorXcd e_end = Eigen::VectorXcd::Zero(N + 1);
    e_end(N) = 1.0;
    Eigen::VectorXcd a = Sfull.transpose().partialPivLu().solve(e_end);
    Eigen::VectorXcd ref_row = Sfull.inverse().row(N).transpose();
    CHECK_NEAR((a - ref_row).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

static void test_proposal() {
    Rng rng(2);
    const double th0 = 0.9, ph0 = 0.3, sigma = 0.3;
    for (int i = 0; i < 5000; ++i) {
        auto pr = proposal(rng, th0, ph0, sigma);
        CHECK(pr.first >= 0.0 && pr.first <= M_PI);
        CHECK(pr.second >= -M_PI && pr.second <= M_PI);
    }
    const double rx = std::sin(th0) * std::cos(ph0), ry = std::sin(th0) * std::sin(ph0),
                 rz = std::cos(th0);
    const long ns = 200000;
    double acc = 0;
    for (long i = 0; i < ns; ++i) {
        auto pr = proposal(rng, th0, ph0, sigma);
        const double x = std::sin(pr.first) * std::cos(pr.second),
                     y = std::sin(pr.first) * std::sin(pr.second), z = std::cos(pr.first);
        acc += x * rx + y * ry + z * rz;
    }
    CHECK_NEAR(acc / ns, std::exp(-sigma * sigma / 2.0), 5e-3);  // E[cos δθ] = e^{-σ²/2}
}

static void test_lambda_builders() {
    auto [q1, lm1] = cf_ground_state_lm(9, 3, 1);
    CHECK(q1 == 0);
    CHECK((int)lm1.size() == 9);
    auto [q2, lm2] = cf_quasihole_lm(9, 3, 1);
    CHECK((int)lm2.size() == 8);
    auto [q3, lm3] = cf_quasiparticle_lm(9, 3, 1);
    CHECK((int)lm3.size() == 10);
}

int main() {
    test_esp_branches();
    test_psiproj_consistency();
    test_psiunproj_orbitals();
    test_sherman_morrison();
    test_extended_slater();
    test_proposal();
    test_lambda_builders();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
