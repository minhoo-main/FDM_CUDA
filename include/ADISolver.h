#pragma once

#include "Grid2D.h"
#include "ELSProduct.h"
#include <vector>
#include <functional>

namespace ELSPricer {

/**
 * ADI (Alternating Direction Implicit) Solver for 2D Black-Scholes PDE
 *
 * Solves the 2D BS PDE using ADI method:
 * - Each time step is split into 2 half-steps
 * - Half-step 1: S1 direction implicit, S2 direction explicit
 * - Half-step 2: S2 direction implicit, S1 direction explicit
 * - Each half-step reduces to tridiagonal systems (Thomas algorithm)
 */
class ADISolver {
public:
    // Constructor
    ADISolver(const Grid2D& grid, const ELSProduct& product);

    // Solve the PDE
    // V_T: Terminal payoff (flattened: V[i * N2 + j])
    // Returns: V_0 at t=0 (flattened)
    std::vector<float> solve(const std::vector<float>& V_T);

    // Solve with early redemption callbacks
    std::vector<float> solveWithEarlyRedemption(
        const std::vector<float>& V_T,
        const ELSProduct& product
    );

protected:
    // Grid and product
    const Grid2D& grid_;
    const ELSProduct& product_;

    int N1_, N2_, Nt_;
    float dS1_, dS2_, dt_;

    // PDE coefficients
    float sigma1_, sigma2_, rho_;
    float r_, q1_, q2_;

    // Tridiagonal coefficients (precomputed)
    std::vector<float> alpha1_, beta1_, gamma1_;  // S1 direction
    std::vector<float> alpha2_, beta2_, gamma2_;  // S2 direction

    // Initialize coefficients
    void precomputeCoefficients();

    // ADI half-steps
    void solveS1Direction(const std::vector<float>& V_in, std::vector<float>& V_out);
    void solveS2Direction(const std::vector<float>& V_in, std::vector<float>& V_out);

    // Apply boundary conditions
    void applyBoundaryConditions(std::vector<float>& V);

    // Apply early redemption
    void applyEarlyRedemption(
        std::vector<float>& V,
        int obsIdx,
        const ELSProduct& product
    );

    // Thomas algorithm for tridiagonal system
    static void thomasAlgorithm(
        const std::vector<float>& lower,
        const std::vector<float>& diag,
        const std::vector<float>& upper,
        const std::vector<float>& rhs,
        std::vector<float>& solution
    );
};

/**
 * Helper function to price ELS using ADI solver
 */
struct PricingResult {
    float price;
    std::vector<float> priceGrid;
    float computeTime;
};

PricingResult priceELS(
    const ELSProduct& product,
    int N1 = 100,
    int N2 = 100,
    int Nt = 200,
    bool verbose = true
);

} // namespace ELSPricer
