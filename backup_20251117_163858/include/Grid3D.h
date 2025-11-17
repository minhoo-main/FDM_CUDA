#pragma once

#include <vector>
#include <iostream>
#include <cmath>

namespace ELSPricer {

/**
 * 3D Grid for 3-asset FDM
 * WARNING: Memory intensive! Use with caution.
 */
class Grid3D {
public:
    Grid3D(double S1_max, double S2_max, double S3_max, double T,
           int N1, int N2, int N3, int Nt)
        : S1_max_(S1_max), S2_max_(S2_max), S3_max_(S3_max), T_(T),
          N1_(N1), N2_(N2), N3_(N3), Nt_(Nt)
    {
        // Grid spacing
        dS1_ = S1_max_ / (N1_ - 1);
        dS2_ = S2_max_ / (N2_ - 1);
        dS3_ = S3_max_ / (N3_ - 1);
        dt_ = T_ / Nt_;

        // Initialize grids
        S1_.resize(N1_);
        S2_.resize(N2_);
        S3_.resize(N3_);
        time_.resize(Nt_ + 1);

        for (int i = 0; i < N1_; ++i) S1_[i] = i * dS1_;
        for (int j = 0; j < N2_; ++j) S2_[j] = j * dS2_;
        for (int k = 0; k < N3_; ++k) S3_[k] = k * dS3_;
        for (int n = 0; n <= Nt_; ++n) time_[n] = n * dt_;
    }

    // Memory estimate
    size_t getMemoryUsageGB() const {
        size_t total_points = (size_t)N1_ * N2_ * N3_ * Nt_;
        return total_points * sizeof(double) / (1024 * 1024 * 1024);
    }

    void printInfo() const {
        std::cout << "\n=== 3D Grid Configuration ===" << std::endl;
        std::cout << "Grid: " << N1_ << " × " << N2_ << " × " << N3_
                  << " × " << Nt_ << std::endl;
        std::cout << "Total points: " << (long long)N1_ * N2_ * N3_ * Nt_
                  << std::endl;
        std::cout << "Memory required: ~" << getMemoryUsageGB() << " GB"
                  << std::endl;

        if (getMemoryUsageGB() > 8) {
            std::cout << "⚠️  WARNING: High memory usage!" << std::endl;
            std::cout << "   Consider using sparse grids or Monte Carlo"
                      << std::endl;
        }
    }

    // Getters
    int getN1() const { return N1_; }
    int getN2() const { return N2_; }
    int getN3() const { return N3_; }
    int getNt() const { return Nt_; }

    double getDS1() const { return dS1_; }
    double getDS2() const { return dS2_; }
    double getDS3() const { return dS3_; }
    double getDt() const { return dt_; }

    const std::vector<double>& getS1() const { return S1_; }
    const std::vector<double>& getS2() const { return S2_; }
    const std::vector<double>& getS3() const { return S3_; }
    const std::vector<double>& getTime() const { return time_; }

    // Index helpers
    inline size_t index(int i, int j, int k) const {
        return (i * N2_ + j) * N3_ + k;
    }

private:
    double S1_max_, S2_max_, S3_max_, T_;
    int N1_, N2_, N3_, Nt_;
    double dS1_, dS2_, dS3_, dt_;

    std::vector<double> S1_, S2_, S3_, time_;
};

} // namespace ELSPricer