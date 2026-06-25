// Large-scale cross-check of log Ψ between C++ and Julia over many systems × K configs.
// Reads the manifest + configs + reference log Ψ written by cpp/reference/dump_logpsi.jl.
// Compares Re(log Ψ) = log|Ψ| (physical, unambiguous) and the phase e^{i Im(log Ψ)}
// (robust to the 2πk branch ambiguity). Usage: test_logpsi <reference_dir>
#include <cfsonsphere/cfsonsphere.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace cfs;

static std::string DIR;
static std::string path(const std::string& f) { return DIR + "/" + f; }

struct Result {
    double max_re, max_phase, us_per_eval;
};

// Read one system; compares Re/phase and times the compute (update + log_det) only.
template <class Psi>
static Result run(Psi& psi, int N, int Kconf, const std::string& tag) {
    std::ifstream cfg(path("cfg_" + tag + ".csv"));
    std::ifstream lp(path("logpsi_" + tag + ".csv"));
    // Read all configs into memory first so the timed region excludes file I/O.
    std::vector<Eigen::VectorXd> thetas(Kconf, Eigen::VectorXd(N)), phis(Kconf, Eigen::VectorXd(N));
    std::vector<cdouble> refs(Kconf);
    for (int k = 0; k < Kconf; ++k) {
        for (int i = 0; i < N; ++i) cfg >> thetas[k](i);
        for (int i = 0; i < N; ++i) cfg >> phis[k](i);
        double re_ref, im_ref;
        lp >> re_ref >> im_ref;
        refs[k] = cdouble(re_ref, im_ref);
    }

    psi.update(thetas[0], phis[0]);  // warm up caches (untimed)

    double max_re = 0.0, max_phase = 0.0;
    std::vector<cdouble> out(Kconf);
    // timing: average over R passes (matches the Julia measurement)
    const int R = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < Kconf; ++k) {
            psi.update(thetas[k], phis[k]);
            out[k] = log_det(psi.slater_det) + psi.jastrow_factor_log;
        }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R;

    for (int k = 0; k < Kconf; ++k) {
        max_re = std::max(max_re, std::abs(out[k].real() - refs[k].real()));
        max_phase = std::max(max_phase, std::abs(std::polar(1.0, out[k].imag()) -
                                                 std::polar(1.0, refs[k].imag())));
    }
    return {max_re, max_phase, 1e6 * secs / Kconf};
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "reference";
    std::ifstream man(path("logpsi_systems.csv"));
    if (!man.good()) {
        std::printf("manifest not found in '%s' (run cpp/reference/dump_logpsi.jl); skipping\n",
                    DIR.c_str());
        return 0;
    }

    // Julia per-eval timings (us), if available.
    std::map<std::string, double> julia_us;
    {
        std::ifstream tf(path("timing_julia.csv"));
        std::string tag;
        long kk;
        double tot, per;
        while (tf >> tag >> kk >> tot >> per) julia_us[tag] = per;
    }

    const double re_tol = 1e-9, phase_tol = 1e-7;
    int failures = 0;
    long total = 0;
    std::printf("%-9s %-7s %4s %6s %11s %12s | %9s %9s %6s\n", "tag", "kind", "N", "K",
                "max|dRe|", "max|dphase|", "cpp(us)", "jl(us)", "jl/cpp");
    std::string line;
    while (std::getline(man, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag, kind;
        int N, n, p, jk, Kconf;
        ss >> tag >> kind >> N >> n >> p >> jk >> Kconf;

        auto [twoQ, lm] = cf_ground_state_lm(N, n, p);
        const int Nsys = static_cast<int>(lm.size());
        Result r;
        if (kind == "proj") {
            PsiProj psi(twoQ, p, Nsys, lm, jk);
            r = run(psi, Nsys, Kconf, tag);
        } else {
            PsiUnproj psi(twoQ, p, Nsys, lm);
            r = run(psi, Nsys, Kconf, tag);
        }
        total += Kconf;
        const bool ok = (r.max_re <= re_tol) && (r.max_phase <= phase_tol);
        const double jl = julia_us.count(tag) ? julia_us[tag] : 0.0;
        std::printf("%-9s %-7s %4d %6d %11.2e %12.2e | %9.2f %9.2f %6.2f %s\n", tag.c_str(),
                    kind.c_str(), Nsys, Kconf, r.max_re, r.max_phase, r.us_per_eval, jl,
                    jl > 0 ? jl / r.us_per_eval : 0.0, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }
    std::printf("\n%ld configs compared across systems; %d system(s) failed\n", total, failures);
    std::printf("Timing: us/eval for full log Ψ (update + logdet + jastrow); jl/cpp > 1 means C++ faster.\n");
    return failures == 0 ? 0 : 1;
}
