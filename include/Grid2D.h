#include "precision.h"
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
    Grid2D(Real S1_min, Real S1_max, int N1,
           Real S2_min, Real S2_max, int N2,
           Real T, int Nt);

    // Accessors
    inline int getN1() const { return N1_; }
    inline int getN2() const { return N2_; }
    inline int getNt() const { return Nt_; }

    inline Real getS1Min() const { return S1_min_; }
    inline Real getS1Max() const { return S1_max_; }
    inline Real getS2Min() const { return S2_min_; }
    inline Real getS2Max() const { return S2_max_; }

    inline Real getdS1() const { return dS1_; }
    inline Real getdS2() const { return dS2_; }
    inline Real getdt() const { return dt_; }
    inline Real getT() const { return T_; }

    inline const std::vector<Real>& getS1() const { return S1_; }
    inline const std::vector<Real>& getS2() const { return S2_; }
    inline const std::vector<Real>& getTime() const { return t_; }

    // Get grid value at (i, j)
    Real getS1(int i) const { return S1_[i]; }
    Real getS2(int j) const { return S2_[j]; }
    Real getTime(int n) const { return t_[n]; }

    // Find nearest grid index
    int findS1Index(Real S1) const;
    int findS2Index(Real S2) const;

    // Bilinear interpolation at arbitrary point
    Real interpolate(const std::vector<Real>& V, Real S1, Real S2) const;

    // Print grid info
    void printInfo() const;

private:
    // Grid parameters
    Real S1_min_, S1_max_;
    Real S2_min_, S2_max_;
    int N1_, N2_;
    Real T_;
    int Nt_;

    // Derived parameters
    Real dS1_, dS2_, dt_;

    // Grid points
    std::vector<Real> S1_;  // S1 grid points [N1]
    std::vector<Real> S2_;  // S2 grid points [N2]
    std::vector<Real> t_;   // Time grid points [Nt+1]
};

/**
 * Create adaptive grid centered around initial prices
 */
std::unique_ptr<Grid2D> createAdaptiveGrid(
    Real S1_0, Real S2_0, Real T,
    int N1 = 100, int N2 = 100, int Nt = 200,
    Real spaceFactor = 3.0
);

} // namespace ELSPricer
