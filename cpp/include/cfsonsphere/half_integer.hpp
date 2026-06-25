// Half-integer angular-momentum bookkeeping.
//
// Quantum-Hall angular momenta (Q*, L, Lz, μ) are half-integers. Following the standard
// convention we store every one of them as an `int` equal to TWICE its value (`two_L = 2L`,
// etc.). Quantities that are guaranteed integer (L-Q*, L+Q*, L+m, 2L, 2Q1) are formed as
// (two_a ± two_b)/2 with exact integer division (operands always share parity), and array
// indices like (μ + L) become (two_mu + two_L)/2.
#pragma once

#include <cmath>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

namespace cfs {

using cdouble = std::complex<double>;

// Occupied single-particle orbitals as (two_L, two_Lz) pairs (each = 2× the half-integer).
using LMList = std::vector<std::pair<int, int>>;

// 0-based index of a magnetic quantum number two_m within the multiplet of total 2L
// (m runs -L, -L+1, ..., L, i.e. two_m runs -two_L, -two_L+2, ..., two_L → 2L+1 = two_L+1 values).
inline int m_index(int two_m, int two_L) {
    return (two_m + two_L) / 2;
}

// Number of m values in a multiplet of total angular momentum two_L (= 2L+1).
inline int multiplet_size(int two_L) {
    return two_L + 1;
}

// log C(n, k) via lgamma; returns -inf when n < k (mirrors Julia custom_logbinomial,
// which only guards n - k >= 0).
inline double log_binomial(long n, long k) {
    if (n - k < 0) {
        return -std::numeric_limits<double>::infinity();
    }
    return std::lgamma(static_cast<double>(n) + 1.0) -
           std::lgamma(static_cast<double>(k) + 1.0) -
           std::lgamma(static_cast<double>(n - k) + 1.0);
}

}  // namespace cfs
