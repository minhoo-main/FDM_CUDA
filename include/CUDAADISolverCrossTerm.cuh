#ifndef CUDA_ADISOLVER_CROSSTERM_CUH
#define CUDA_ADISOLVER_CROSSTERM_CUH

#include "Grid2D.h"
#include "ELSProduct.h"
#include <vector>

namespace ELSPricer {
namespace CUDA {

/**
 * CUDA-accelerated ADI Solver with Explicit Cross-Term Correction
 *
 * GPU implementation of ADI with mixed derivative term
 * ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2 handled explicitly
 *
 * Uses CUDA kernels for:
 *   - Cross-term computation (highly parallel)
 *   - Batched tridiagonal solves
 *   - Boundary conditions
 */
class CUDAADISolverCrossTerm {
public:
    CUDAADISolverCrossTerm(const Grid2D& grid, const ELSProduct& product);
    ~CUDAADISolverCrossTerm();

    std::vector<float> solve(const std::vector<float>& V_T);
    std::vector<float> solveWithEarlyRedemption(
        const std::vector<float>& V_T,
        const ELSProduct& product);

private:
    void initialize();
    void cleanup();
    void precomputeCoefficients();

    void copyToDevice(const std::vector<float>& V_host);
    void copyFromDevice(std::vector<float>& V_host);

    void computeCrossTermGPU(const float* d_V_in, float* d_cross_out);
    void addCrossTermToRHSGPU(const float* d_V, float* d_RHS);
    void solveS1DirectionGPU(const float* d_RHS, float* d_V_out);
    void solveS2DirectionGPU(const float* d_RHS, float* d_V_out);
    void applyBoundaryConditionsGPU(float* d_V);

private:
    const Grid2D& grid_;
    const ELSProduct& product_;

    int N1_, N2_, Nt_;
    float dS1_, dS2_, dt_;
    float sigma1_, sigma2_, rho_, r_, q1_, q2_;

    // Host arrays
    std::vector<float> alpha1_, beta1_, gamma1_;
    std::vector<float> alpha2_, beta2_, gamma2_;
    std::vector<float> cross_coef_;

    // Device pointers
    float* d_V_;
    float* d_V_half_;
    float* d_V_transposed_;
    float* d_RHS1_;
    float* d_RHS2_;
    float* d_cross_;
    float* d_S1_;
    float* d_S2_;
    float* d_cross_coef_;

    float* d_alpha1_;
    float* d_beta1_;
    float* d_gamma1_;
    float* d_alpha2_;
    float* d_beta2_;
    float* d_gamma2_;
};

struct PricingResultCrossTermGPU {
    float price;
    std::vector<float> grid;
    float computeTime;
};

PricingResultCrossTermGPU priceELSCrossTermGPU(
    const ELSProduct& product,
    int N1, int N2, int Nt,
    bool verbose = false);

} // namespace CUDA
} // namespace ELSPricer

#endif // CUDA_ADISOLVER_CROSSTERM_CUH
