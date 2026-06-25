// Convenience builders for the (two_Qstar, l_m_list) of the standard CF states.
//
// Mirrors the Julia helpers in two-units: two_Qstar = 2*(N/n - n)/2 = N/n - n (requires n | N),
// and orbitals run over Λ-levels |Qstar| .. |Qstar|+|n|-1. Returns the SIGNED two_Qstar
// (the orbital monopole strength) and the (two_L, two_Lz) list.
#pragma once

#include <cassert>
#include <cstdlib>
#include <utility>

#include "half_integer.hpp"

namespace cfs {

inline int two_qstar(int N, int n) {
    assert(N % n == 0 && "cf_*_lm requires n | N for an integer 2*Qstar");
    return N / n - n;  // = 2*Qstar
}

// Ground state: all Lz filled in Λ-levels |Qstar| .. |Qstar|+|n|-1.
inline std::pair<int, LMList> cf_ground_state_lm(int N, int n, int /*p*/) {
    const int twoQ = two_qstar(N, n);
    const int absQ = std::abs(twoQ);
    const int levels = std::abs(n);
    LMList lm;
    for (int k = 0; k < levels; ++k) {
        const int two_L = absQ + 2 * k;
        for (int two_Lz = -two_L; two_Lz <= two_L; two_Lz += 2) lm.emplace_back(two_L, two_Lz);
    }
    return {twoQ, lm};
}

// One quasihole: drop the top orbital (Lqh, Lqh) of the highest occupied Λ-level.
inline std::pair<int, LMList> cf_quasihole_lm(int N, int n, int /*p*/) {
    const int twoQ = two_qstar(N, n);
    const int absQ = std::abs(twoQ);
    const int levels = std::abs(n);
    const int two_Lqh = absQ + 2 * (levels - 1);
    LMList lm;
    for (int k = 0; k < levels; ++k) {
        const int two_L = absQ + 2 * k;
        for (int two_Lz = -two_L; two_Lz <= two_L; two_Lz += 2)
            if (!(two_L == two_Lqh && two_Lz == two_Lqh)) lm.emplace_back(two_L, two_Lz);
    }
    return {twoQ, lm};
}

// One quasiparticle: add the orbital (Lqp, Lqp) in the next Λ-level.
inline std::pair<int, LMList> cf_quasiparticle_lm(int N, int n, int /*p*/) {
    const int twoQ = two_qstar(N, n);
    const int absQ = std::abs(twoQ);
    const int levels = std::abs(n);
    const int two_Lqp = absQ + 2 * levels;
    LMList lm;
    for (int k = 0; k < levels; ++k) {
        const int two_L = absQ + 2 * k;
        for (int two_Lz = -two_L; two_Lz <= two_L; two_Lz += 2) lm.emplace_back(two_L, two_Lz);
    }
    lm.emplace_back(two_Lqp, two_Lqp);
    return {twoQ, lm};
}

}  // namespace cfs
