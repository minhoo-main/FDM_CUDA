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
    Grid2D(float S1_min, float S1_max, int N1,
           float S2_min, float S2_max, int N2,
           float T, int Nt);

    // Accessors
    inline int getN1() const { return N1_; }
    inline int getN2() const { return N2_; }
    inline int getNt() const { return Nt_; }

    inline float getS1Min() const { return S1_min_; }
    inline float getS1Max() const { return S1_max_; }
    inline float getS2Min() const { return S2_min_; }
    inline float getS2Max() const { return S2_max_; }

    inline float getdS1() const { return dS1_; }
    inline float getdS2() const { return dS2_; }
    inline float getdt() const { return dt_; }
    inline float getT() const { return T_; }

    inline const std::vector<float>& getS1() const { return S1_; }
    inline const std::vector<float>& getS2() const { return S2_; }
    inline const std::vector<float>& getTime() const { return t_; }

    // Get grid value at (i, j)
    float getS1(int i) const { return S1_[i]; }
    float getS2(int j) const { return S2_[j]; }
    float getTime(int n) const { return t_[n]; }

    // Find nearest grid index
    int findS1Index(float S1) const;
    int findS2Index(float S2) const;

    // Bilinear interpolation at arbitrary point
    float interpolate(const std::vector<float>& V, float S1, float S2) const;

    // Print grid info
    void printInfo() const;

private:
    // Grid parameters
    float S1_min_, S1_max_;
    float S2_min_, S2_max_;
    int N1_, N2_;
    float T_;
    int Nt_;

    // Derived parameters
    float dS1_, dS2_, dt_;

    // Grid points
    std::vector<float> S1_;  // S1 grid points [N1]
    std::vector<float> S2_;  // S2 grid points [N2]
    std::vector<float> t_;   // Time grid points [Nt+1]
};

/**
 * Create adaptive grid centered around initial prices
 */
std::unique_ptr<Grid2D> createAdaptiveGrid(
    float S1_0, float S2_0, float T,
    int N1 = 100, int N2 = 100, int Nt = 200,
    float spaceFactor = 3.0
);

} // namespace ELSPricer
