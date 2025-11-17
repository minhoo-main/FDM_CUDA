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

    std::vector<double> solve(const std::vector<double>& V_T);
    std::vector<double> solveWithEarlyRedemption(
        const std::vector<double>& V_T,
        const ELSProduct& product);

private:
    void initialize();
    void cleanup();
    void precomputeCoefficients();

    void copyToDevice(const std::vector<double>& V_host);
    void copyFromDevice(std::vector<double>& V_host);

    void computeCrossTermGPU(const double* d_V_in, double* d_cross_out);
    void addCrossTermToRHSGPU(const double* d_V, double* d_RHS);
    void solveS1DirectionGPU(const double* d_RHS, double* d_V_out);
    void solveS2DirectionGPU(const double* d_RHS, double* d_V_out);
    void applyBoundaryConditionsGPU(double* d_V);

private:
    const Grid2D& grid_;
    const ELSProduct& product_;

    int N1_, N2_, Nt_;
    double dS1_, dS2_, dt_;
    double sigma1_, sigma2_, rho_, r_, q1_, q2_;

    // Host arrays
    std::vector<double> alpha1_, beta1_, gamma1_;
    std::vector<double> alpha2_, beta2_, gamma2_;
    std::vector<double> cross_coef_;

    // Device pointers
    double* d_V_;
    double* d_V_half_;
    double* d_V_transposed_;
    double* d_RHS1_;
    double* d_RHS2_;
    double* d_cross_;
    double* d_S1_;
    double* d_S2_;
    double* d_cross_coef_;

    double* d_alpha1_;
    double* d_beta1_;
    double* d_gamma1_;
    double* d_alpha2_;
    double* d_beta2_;
    double* d_gamma2_;
};

struct PricingResultCrossTermGPU {
    double price;
    std::vector<double> grid;
    double computeTime;
};

PricingResultCrossTermGPU priceELSCrossTermGPU(
    const ELSProduct& product,
    int N1, int N2, int Nt,
    bool verbose = false);

} // namespace CUDA
} // namespace ELSPricer

#endif // CUDA_ADISOLVER_CROSSTERM_CUH
