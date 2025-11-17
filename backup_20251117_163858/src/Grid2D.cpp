#include "Grid2D.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace ELSPricer {

Grid2D::Grid2D(double S1_min, double S1_max, int N1,
               double S2_min, double S2_max, int N2,
               double T, int Nt)
    : S1_min_(S1_min), S1_max_(S1_max), N1_(N1),
      S2_min_(S2_min), S2_max_(S2_max), N2_(N2),
      T_(T), Nt_(Nt)
{
    // Calculate step sizes
    dS1_ = (S1_max_ - S1_min_) / (N1_ - 1);
    dS2_ = (S2_max_ - S2_min_) / (N2_ - 1);
    dt_ = T_ / Nt_;

    // Create S1 grid
    S1_.resize(N1_);
    for (int i = 0; i < N1_; ++i) {
        S1_[i] = S1_min_ + i * dS1_;
    }

    // Create S2 grid
    S2_.resize(N2_);
    for (int j = 0; j < N2_; ++j) {
        S2_[j] = S2_min_ + j * dS2_;
    }

    // Create time grid (0 to T)
    t_.resize(Nt_ + 1);
    for (int n = 0; n <= Nt_; ++n) {
        t_[n] = n * dt_;
    }
}

int Grid2D::findS1Index(double S1) const {
    // Find nearest index
    int idx = static_cast<int>(std::round((S1 - S1_min_) / dS1_));
    return std::max(0, std::min(N1_ - 1, idx));
}

int Grid2D::findS2Index(double S2) const {
    int idx = static_cast<int>(std::round((S2 - S2_min_) / dS2_));
    return std::max(0, std::min(N2_ - 1, idx));
}

double Grid2D::interpolate(const std::vector<double>& V, double S1, double S2) const {
    // Bilinear interpolation
    // V is stored in row-major order: V[i * N2 + j] = V(S1[i], S2[j])

    // Find surrounding grid points
    double x = (S1 - S1_min_) / dS1_;
    double y = (S2 - S2_min_) / dS2_;

    int i0 = static_cast<int>(std::floor(x));
    int j0 = static_cast<int>(std::floor(y));

    // Clamp to grid boundaries
    i0 = std::max(0, std::min(N1_ - 2, i0));
    j0 = std::max(0, std::min(N2_ - 2, j0));

    int i1 = i0 + 1;
    int j1 = j0 + 1;

    // Interpolation weights
    double wx = x - i0;
    double wy = y - j0;

    // Bilinear interpolation
    double v00 = V[i0 * N2_ + j0];
    double v01 = V[i0 * N2_ + j1];
    double v10 = V[i1 * N2_ + j0];
    double v11 = V[i1 * N2_ + j1];

    double v0 = v00 * (1 - wy) + v01 * wy;
    double v1 = v10 * (1 - wy) + v11 * wy;

    return v0 * (1 - wx) + v1 * wx;
}

void Grid2D::printInfo() const {
    std::cout << "Grid2D Configuration\n";
    std::cout << "====================\n";
    std::cout << "S1: [" << S1_min_ << ", " << S1_max_ << "] with " << N1_
              << " points (dS1=" << dS1_ << ")\n";
    std::cout << "S2: [" << S2_min_ << ", " << S2_max_ << "] with " << N2_
              << " points (dS2=" << dS2_ << ")\n";
    std::cout << "Time: [0, " << T_ << "] with " << Nt_
              << " steps (dt=" << dt_ << ")\n";
    std::cout << "Total grid points: " << (N1_ * N2_) << "\n";
}

std::unique_ptr<Grid2D> createAdaptiveGrid(
    double S1_0, double S2_0, double T,
    int N1, int N2, int Nt,
    double spaceFactor)
{
    double S1_min = 0.0;
    double S1_max = S1_0 * spaceFactor;
    double S2_min = 0.0;
    double S2_max = S2_0 * spaceFactor;

    return std::make_unique<Grid2D>(
        S1_min, S1_max, N1,
        S2_min, S2_max, N2,
        T, Nt
    );
}

} // namespace ELSPricer
