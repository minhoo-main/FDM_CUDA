#pragma once

#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <vector>

namespace ELSPricer {
namespace CUDA {

/**
 * GPU-accelerated ADI Solver using CUDA
 *
 * Key optimization: Batched Tridiagonal Solver
 * - Solves N2 independent tridiagonal systems in parallel for S1 direction
 * - Solves N1 independent tridiagonal systems in parallel for S2 direction
 * - Expected speedup: 10-50x over CPU for large grids
 */
class CUDAADISolver {
public:
    // Constructor
    CUDAADISolver(const Grid2D& grid, const ELSProduct& product);

    // Destructor
    ~CUDAADISolver();

    // Solve the PDE on GPU
    std::vector<double> solve(const std::vector<double>& V_T);

    // Solve with early redemption
    std::vector<double> solveWithEarlyRedemption(
        const std::vector<double>& V_T,
        const ELSProduct& product
    );

private:
    const Grid2D& grid_;
    const ELSProduct& product_;

    int N1_, N2_, Nt_;
    double dS1_, dS2_, dt_;
    double sigma1_, sigma2_, rho_;
    double r_, q1_, q2_;

    // Device pointers
    double* d_V_;           // Current solution on device
    double* d_V_half_;      // Intermediate solution
    double* d_V_transposed_; // Transposed matrix for S1 direction (reused)
    double* d_S1_;          // S1 grid values on device
    double* d_S2_;          // S2 grid values on device
    double* d_alpha1_;      // S1 tridiagonal coefficients
    double* d_beta1_;
    double* d_gamma1_;
    double* d_alpha2_;      // S2 tridiagonal coefficients
    double* d_beta2_;
    double* d_gamma2_;

    // Host coefficients
    std::vector<double> alpha1_, beta1_, gamma1_;
    std::vector<double> alpha2_, beta2_, gamma2_;

    // Initialize GPU memory and coefficients
    void initialize();
    void cleanup();
    void precomputeCoefficients();

    // GPU operations
    void copyToDevice(const std::vector<double>& V_host);
    void copyFromDevice(std::vector<double>& V_host);

    void solveS1DirectionGPU();
    void solveS2DirectionGPU();
    void applyBoundaryConditionsGPU();
    void applyEarlyRedemptionGPU(int obsIdx);
};

/**
 * Price ELS using GPU-accelerated solver
 */
PricingResult priceELSGPU(
    const ELSProduct& product,
    int N1 = 100,
    int N2 = 100,
    int Nt = 200,
    bool verbose = true
);

} // namespace CUDA
} // namespace ELSPricer
