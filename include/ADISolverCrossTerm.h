#ifndef ADISOLVER_CROSSTERM_H
#define ADISOLVER_CROSSTERM_H

#include "Grid2D.h"
#include "ELSProduct.h"
#include <vector>

namespace ELSPricer {

/**
 * ADI Solver with Explicit Cross-Term Correction
 *
 * Handles the mixed derivative term ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2
 * using explicit treatment (added to RHS)
 *
 * Algorithm:
 *   1. Compute cross-term explicitly: C = ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2
 *   2. RHS1 = V + 0.5*dt*C
 *   3. Solve S1 implicit: (I - 0.5*dt*L1) V_half = RHS1
 *   4. Compute cross-term for V_half: C_half
 *   5. RHS2 = V_half + 0.5*dt*C_half
 *   6. Solve S2 implicit: (I - 0.5*dt*L2) V_new = RHS2
 */
class ADISolverCrossTerm {
public:
    ADISolverCrossTerm(const Grid2D& grid, const ELSProduct& product);

    std::vector<float> solve(const std::vector<float>& V_T);
    std::vector<float> solveWithEarlyRedemption(
        const std::vector<float>& V_T,
        const ELSProduct& product);

private:
    void precomputeCoefficients();
    void computeCrossTerm(const std::vector<float>& V, std::vector<float>& cross);
    void addCrossTermToRHS(const std::vector<float>& V, std::vector<float>& RHS);

    void thomasAlgorithm(
        const std::vector<float>& lower,
        const std::vector<float>& diag,
        const std::vector<float>& upper,
        const std::vector<float>& rhs,
        std::vector<float>& solution);

    void solveS1Direction(const std::vector<float>& V_in, std::vector<float>& V_out);
    void solveS2Direction(const std::vector<float>& V_in, std::vector<float>& V_out);
    void applyBoundaryConditions(std::vector<float>& V);
    void applyEarlyRedemption(
        std::vector<float>& V,
        int obsIdx,
        const ELSProduct& product);

private:
    const Grid2D& grid_;
    const ELSProduct& product_;

    int N1_, N2_, Nt_;
    float dS1_, dS2_, dt_;
    float sigma1_, sigma2_, rho_, r_, q1_, q2_;

    // Tridiagonal coefficients
    std::vector<float> alpha1_, beta1_, gamma1_;
    std::vector<float> alpha2_, beta2_, gamma2_;

    // Cross-term coefficients (precomputed)
    std::vector<float> cross_coef_;  // ρ σ1 σ2 S1[i] S2[j] / (4 dS1 dS2)
};

struct PricingResultCrossTerm {
    float price;
    std::vector<float> grid;
    float computeTime;
};

PricingResultCrossTerm priceELSCrossTerm(
    const ELSProduct& product,
    int N1, int N2, int Nt,
    bool verbose = false);

} // namespace ELSPricer

#endif // ADISOLVER_CROSSTERM_H
