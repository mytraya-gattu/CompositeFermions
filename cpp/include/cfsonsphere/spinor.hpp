// Spinor (CP^1) coordinates on the sphere.
#pragma once

#include <cmath>
#include <complex>

#include "half_integer.hpp"

namespace cfs {

// (u, v) = (cos(θ/2) e^{+iφ/2}, sin(θ/2) e^{-iφ/2}) for a single particle.
inline std::pair<cdouble, cdouble> u_v_generator(double theta, double phi) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {c * std::polar(1.0, 0.5 * phi), s * std::polar(1.0, -0.5 * phi)};
}

}  // namespace cfs
