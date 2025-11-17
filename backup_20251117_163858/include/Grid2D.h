#pragma once

#include <vector>
#include <memory>

namespace ELSPricer {

/**
 * 2D Grid for Finite Difference Method
 *
 * Represents a uniform 2D grid for solving 2D Black-Scholes PDE
 * Grid points: (S1[i], S2[j]) for i=0..N1-1, j=0..N2-1
 */
class Grid2D {
public:
    // Constructor
    Grid2D(double S1_min, double S1_max, int N1,
           double S2_min, double S2_max, int N2,
           double T, int Nt);

    // Accessors
    inline int getN1() const { return N1_; }
    inline int getN2() const { return N2_; }
    inline int getNt() const { return Nt_; }

    inline double getS1Min() const { return S1_min_; }
    inline double getS1Max() const { return S1_max_; }
    inline double getS2Min() const { return S2_min_; }
    inline double getS2Max() const { return S2_max_; }

    inline double getdS1() const { return dS1_; }
    inline double getdS2() const { return dS2_; }
    inline double getdt() const { return dt_; }
    inline double getT() const { return T_; }

    inline const std::vector<double>& getS1() const { return S1_; }
    inline const std::vector<double>& getS2() const { return S2_; }
    inline const std::vector<double>& getTime() const { return t_; }

    // Get grid value at (i, j)
    double getS1(int i) const { return S1_[i]; }
    double getS2(int j) const { return S2_[j]; }
    double getTime(int n) const { return t_[n]; }

    // Find nearest grid index
    int findS1Index(double S1) const;
    int findS2Index(double S2) const;

    // Bilinear interpolation at arbitrary point
    double interpolate(const std::vector<double>& V, double S1, double S2) const;

    // Print grid info
    void printInfo() const;

private:
    // Grid parameters
    double S1_min_, S1_max_;
    double S2_min_, S2_max_;
    int N1_, N2_;
    double T_;
    int Nt_;

    // Derived parameters
    double dS1_, dS2_, dt_;

    // Grid points
    std::vector<double> S1_;  // S1 grid points [N1]
    std::vector<double> S2_;  // S2 grid points [N2]
    std::vector<double> t_;   // Time grid points [Nt+1]
};

/**
 * Create adaptive grid centered around initial prices
 */
std::unique_ptr<Grid2D> createAdaptiveGrid(
    double S1_0, double S2_0, double T,
    int N1 = 100, int N2 = 100, int Nt = 200,
    double spaceFactor = 3.0
);

} // namespace ELSPricer
