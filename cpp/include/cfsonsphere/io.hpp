// Minimal CSV output helpers (no JLD2/HDF5 dependency).
#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace cfs {

// Write a real vector as a single-column CSV.
inline void save_csv(const std::string& path, const std::vector<double>& v) {
    std::ofstream f(path);
    f << std::setprecision(17);
    for (double x : v) f << x << "\n";
}

inline void save_csv(const std::string& path, const Eigen::VectorXd& v) {
    std::ofstream f(path);
    f << std::setprecision(17);
    for (int i = 0; i < v.size(); ++i) f << v(i) << "\n";
}

// Write two paired columns (e.g. grid, value) as CSV.
inline void save_csv(const std::string& path, const std::vector<double>& x,
                     const std::vector<double>& y) {
    std::ofstream f(path);
    f << std::setprecision(17);
    for (std::size_t i = 0; i < x.size(); ++i) f << x[i] << "," << y[i] << "\n";
}

}  // namespace cfs
