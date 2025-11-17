#include "precision.h"
#include "CUDAADISolverCrossTerm.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ELSPricer {
namespace CUDA {

// Forward declarations from batched_thomas.cu
void batchedThomas(
    const double* d_lower,
    const double* d_diag,
    const double* d_upper,
    const double* d_rhs,
    double* d_solution,
    int N,
    int batch_size);

void applyBoundaryConditions(double* d_V, int N1, int N2);
void transpose(const double* d_input, double* d_output, int rows, int cols);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * CUDA Kernel: Compute cross-term ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2
 *
 * Uses 4-point stencil for mixed derivative:
 * ∂²V/∂S1∂S2 ≈ [V(i+1,j+1) - V(i+1,j-1) - V(i-1,j+1) + V(i-1,j-1)] / (4 dS1 dS2)
 */
__global__ void computeCrossTermKernel(
    const double* d_V,
    const double* d_cross_coef,
    double* d_cross,
    int N1,
    int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute for interior points
    if (i >= 1 && i < N1 - 1 && j >= 1 && j < N2 - 1) {
        int idx = i * N2 + j;

        // 4-point stencil for mixed derivative
        ELSPricer::Real mixed_deriv =
            d_V[(i+1) * N2 + (j+1)] - d_V[(i+1) * N2 + (j-1)] -
            d_V[(i-1) * N2 + (j+1)] + d_V[(i-1) * N2 + (j-1)];

        // cross = coef * mixed_deriv
        // where coef = ρ σ1 σ2 S1[i] S2[j] / (4 dS1 dS2)
        d_cross[idx] = d_cross_coef[idx] * mixed_deriv;
    }
    else if (i < N1 && j < N2) {
        // Boundary points: cross-term = 0
        d_cross[i * N2 + j] = 0.0;
    }
}

/**
 * CUDA Kernel: Add cross-term to RHS
 * RHS = V + 0.5 * dt * cross_term
 */
__global__ void addCrossTermKernel(
    const double* d_V,
    const double* d_cross,
    double* d_RHS,
    ELSPricer::Real dt,
    int N1,
    int N2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N1 && j < N2) {
        int idx = i * N2 + j;
        d_RHS[idx] = d_V[idx] + 0.5 * dt * d_cross[idx];
    }
}

CUDAADISolverCrossTerm::CUDAADISolverCrossTerm(
    const Grid2D& grid,
    const ELSProduct& product)
    : grid_(grid), product_(product),
      d_V_(nullptr), d_V_half_(nullptr), d_V_transposed_(nullptr),
      d_RHS1_(nullptr), d_RHS2_(nullptr), d_cross_(nullptr),
      d_S1_(nullptr), d_S2_(nullptr), d_cross_coef_(nullptr),
      d_alpha1_(nullptr), d_beta1_(nullptr), d_gamma1_(nullptr),
      d_alpha2_(nullptr), d_beta2_(nullptr), d_gamma2_(nullptr)
{
    N1_ = grid.getN1();
    N2_ = grid.getN2();
    Nt_ = grid.getNt();
    dS1_ = grid.getdS1();
    dS2_ = grid.getdS2();
    dt_ = grid.getdt();

    sigma1_ = product.getSigma1();
    sigma2_ = product.getSigma2();
    rho_ = product.getRho();
    r_ = product.getR();
    q1_ = product.getQ1();
    q2_ = product.getQ2();

    initialize();
}

CUDAADISolverCrossTerm::~CUDAADISolverCrossTerm() {
    cleanup();
}

void CUDAADISolverCrossTerm::precomputeCoefficients() {
    alpha1_.resize(N1_ - 1);
    beta1_.resize(N1_);
    gamma1_.resize(N1_ - 1);

    alpha2_.resize(N2_ - 1);
    beta2_.resize(N2_);
    gamma2_.resize(N2_ - 1);

    // Precompute cross-term coefficients
    cross_coef_.resize(N1_ * N2_);
    const auto& S1 = grid_.getS1();
    const auto& S2 = grid_.getS2();

    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            cross_coef_[i * N2_ + j] =
                rho_ * sigma1_ * sigma2_ * S1[i] * S2[j] / (4.0 * dS1_ * dS2_);
        }
    }

    // S1 direction coefficients
    for (int i = 1; i < N1_ - 1; ++i) {
        ELSPricer::Real S1 = grid_.getS1(i);
        ELSPricer::Real a1 = 0.5 * sigma1_ * sigma1_ * S1 * S1 / (dS1_ * dS1_);
        ELSPricer::Real b1 = (r_ - q1_) * S1 / (2.0 * dS1_);

        alpha1_[i - 1] = -0.5 * dt_ * (a1 - b1);
        beta1_[i] = 1.0 + dt_ * (a1 + 0.5 * r_);
        gamma1_[i] = -0.5 * dt_ * (a1 + b1);
    }
    beta1_[0] = 1.0;
    beta1_[N1_ - 1] = 1.0;

    // S2 direction coefficients
    for (int j = 1; j < N2_ - 1; ++j) {
        ELSPricer::Real S2 = grid_.getS2(j);
        ELSPricer::Real a2 = 0.5 * sigma2_ * sigma2_ * S2 * S2 / (dS2_ * dS2_);
        ELSPricer::Real b2 = (r_ - q2_) * S2 / (2.0 * dS2_);

        alpha2_[j - 1] = -0.5 * dt_ * (a2 - b2);
        beta2_[j] = 1.0 + dt_ * (a2 + 0.5 * r_);
        gamma2_[j] = -0.5 * dt_ * (a2 + b2);
    }
    beta2_[0] = 1.0;
    beta2_[N2_ - 1] = 1.0;
}

void CUDAADISolverCrossTerm::initialize() {
    // Precompute coefficients on host
    precomputeCoefficients();

    // Allocate device memory
    size_t grid_size = N1_ * N2_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_V_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_V_half_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_V_transposed_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_RHS1_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_RHS2_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_cross_, grid_size));

    // Allocate and copy S1, S2 grids
    CUDA_CHECK(cudaMalloc(&d_S1_, N1_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S2_, N2_ * sizeof(double)));

    const auto& S1 = grid_.getS1();
    const auto& S2 = grid_.getS2();
    CUDA_CHECK(cudaMemcpy(d_S1_, S1.data(), N1_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S2_, S2.data(), N2_ * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate and copy cross-term coefficients
    CUDA_CHECK(cudaMalloc(&d_cross_coef_, grid_size));
    CUDA_CHECK(cudaMemcpy(d_cross_coef_, cross_coef_.data(), grid_size, cudaMemcpyHostToDevice));

    // Allocate coefficient arrays
    CUDA_CHECK(cudaMalloc(&d_alpha1_, (N1_ - 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta1_, N1_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gamma1_, (N1_ - 1) * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_alpha2_, (N2_ - 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta2_, N2_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gamma2_, (N2_ - 1) * sizeof(double)));

    // Copy coefficients to device
    CUDA_CHECK(cudaMemcpy(d_alpha1_, alpha1_.data(), (N1_ - 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta1_, beta1_.data(), N1_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma1_, gamma1_.data(), (N1_ - 1) * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_alpha2_, alpha2_.data(), (N2_ - 1) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta2_, beta2_.data(), N2_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma2_, gamma2_.data(), (N2_ - 1) * sizeof(double), cudaMemcpyHostToDevice));

    // Print GPU info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "✓ GPU Initialized (Cross-Term): " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
}

void CUDAADISolverCrossTerm::cleanup() {
    if (d_V_) CUDA_CHECK(cudaFree(d_V_));
    if (d_V_half_) CUDA_CHECK(cudaFree(d_V_half_));
    if (d_V_transposed_) CUDA_CHECK(cudaFree(d_V_transposed_));
    if (d_RHS1_) CUDA_CHECK(cudaFree(d_RHS1_));
    if (d_RHS2_) CUDA_CHECK(cudaFree(d_RHS2_));
    if (d_cross_) CUDA_CHECK(cudaFree(d_cross_));
    if (d_S1_) CUDA_CHECK(cudaFree(d_S1_));
    if (d_S2_) CUDA_CHECK(cudaFree(d_S2_));
    if (d_cross_coef_) CUDA_CHECK(cudaFree(d_cross_coef_));
    if (d_alpha1_) CUDA_CHECK(cudaFree(d_alpha1_));
    if (d_beta1_) CUDA_CHECK(cudaFree(d_beta1_));
    if (d_gamma1_) CUDA_CHECK(cudaFree(d_gamma1_));
    if (d_alpha2_) CUDA_CHECK(cudaFree(d_alpha2_));
    if (d_beta2_) CUDA_CHECK(cudaFree(d_beta2_));
    if (d_gamma2_) CUDA_CHECK(cudaFree(d_gamma2_));
}

void CUDAADISolverCrossTerm::copyToDevice(const std::vector<ELSPricer::Real>& V_host) {
    CUDA_CHECK(cudaMemcpy(d_V_, V_host.data(), N1_ * N2_ * sizeof(double), cudaMemcpyHostToDevice));
}

void CUDAADISolverCrossTerm::copyFromDevice(std::vector<ELSPricer::Real>& V_host) {
    V_host.resize(N1_ * N2_);
    CUDA_CHECK(cudaMemcpy(V_host.data(), d_V_, N1_ * N2_ * sizeof(double), cudaMemcpyDeviceToHost));
}

void CUDAADISolverCrossTerm::computeCrossTermGPU(
    const double* d_V_in,
    double* d_cross_out)
{
    dim3 block(16, 16);
    dim3 grid((N1_ + 15) / 16, (N2_ + 15) / 16);

    computeCrossTermKernel<<<grid, block>>>(
        d_V_in, d_cross_coef_, d_cross_out, N1_, N2_);

    CUDA_CHECK(cudaGetLastError());
}

void CUDAADISolverCrossTerm::addCrossTermToRHSGPU(
    const double* d_V,
    double* d_RHS)
{
    // First compute cross-term
    computeCrossTermGPU(d_V, d_cross_);

    // Then add to RHS
    dim3 block(16, 16);
    dim3 grid((N1_ + 15) / 16, (N2_ + 15) / 16);

    addCrossTermKernel<<<grid, block>>>(d_V, d_cross_, d_RHS, dt_, N1_, N2_);

    CUDA_CHECK(cudaGetLastError());
}

void CUDAADISolverCrossTerm::solveS1DirectionGPU(
    const double* d_RHS,
    double* d_V_out)
{
    // Transpose RHS: [N1 x N2] -> [N2 x N1]
    transpose(d_RHS, d_V_transposed_, N1_, N2_);

    // Solve N2 tridiagonal systems in parallel
    batchedThomas(d_alpha1_, d_beta1_, d_gamma1_, d_V_transposed_, d_V_half_, N1_, N2_);

    // Transpose back: [N2 x N1] -> [N1 x N2]
    transpose(d_V_half_, d_V_out, N2_, N1_);
}

void CUDAADISolverCrossTerm::solveS2DirectionGPU(
    const double* d_RHS,
    double* d_V_out)
{
    // V is already in row-major format [N1 x N2]
    batchedThomas(d_alpha2_, d_beta2_, d_gamma2_, d_RHS, d_V_out, N2_, N1_);
}

void CUDAADISolverCrossTerm::applyBoundaryConditionsGPU(double* d_V) {
    applyBoundaryConditions(d_V, N1_, N2_);
}

std::vector<ELSPricer::Real> CUDAADISolverCrossTerm::solve(const std::vector<ELSPricer::Real>& V_T) {
    // Copy initial data to device
    copyToDevice(V_T);

    // Time stepping with cross-term
    for (int n = Nt_ - 1; n >= 0; --n) {
        // Step 1: Add cross-term to RHS1
        addCrossTermToRHSGPU(d_V_, d_RHS1_);

        // Step 2: S1 direction implicit
        solveS1DirectionGPU(d_RHS1_, d_V_half_);

        // Step 3: Add cross-term to RHS2
        addCrossTermToRHSGPU(d_V_half_, d_RHS2_);

        // Step 4: S2 direction implicit
        solveS2DirectionGPU(d_RHS2_, d_V_);

        // Boundary conditions
        applyBoundaryConditionsGPU(d_V_);
    }

    // Copy result back
    std::vector<ELSPricer::Real> V_0;
    copyFromDevice(V_0);

    return V_0;
}

std::vector<ELSPricer::Real> CUDAADISolverCrossTerm::solveWithEarlyRedemption(
    const std::vector<ELSPricer::Real>& V_T,
    const ELSProduct& product)
{
    copyToDevice(V_T);

    const auto& obsDates = product.getObservationDates();
    const auto& timeGrid = grid_.getTime();

    // Find observation indices
    std::vector<int> obsIndices;
    for (ELSPricer::Real obsDate : obsDates) {
        int idx = 0;
        ELSPricer::Real minDiff = std::abs(timeGrid[0] - obsDate);
        for (int n = 1; n <= Nt_; ++n) {
            ELSPricer::Real diff = std::abs(timeGrid[n] - obsDate);
            if (diff < minDiff) {
                minDiff = diff;
                idx = n;
            }
        }
        obsIndices.push_back(idx);
    }

    int obsIdx = static_cast<int>(obsDates.size()) - 1;

    // Time stepping (backward from Nt_ to 1)
    for (int n = Nt_; n >= 1; --n) {
        // Check early redemption (simplified - using CPU for now)
        if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
            // Copy to host, apply redemption, copy back
            std::vector<ELSPricer::Real> V_host;
            copyFromDevice(V_host);

            const auto& S1 = grid_.getS1();
            const auto& S2 = grid_.getS2();
            const auto& barriers = product.getRedemptionBarriers();
            const auto& coupons = product.getCoupons();

            for (int i = 0; i < N1_; ++i) {
                for (int j = 0; j < N2_; ++j) {
                    auto result = product.checkEarlyRedemption(S1[i], S2[j], obsIdx);
                    if (result.isRedeemed) {
                        V_host[i * N2_ + j] = result.payoff;
                    }
                }
            }

            copyToDevice(V_host);
            --obsIdx;
        }

        // ADI steps with cross-term
        if (n < Nt_) {
            addCrossTermToRHSGPU(d_V_, d_RHS1_);
            solveS1DirectionGPU(d_RHS1_, d_V_half_);
            addCrossTermToRHSGPU(d_V_half_, d_RHS2_);
            solveS2DirectionGPU(d_RHS2_, d_V_);
            applyBoundaryConditionsGPU(d_V_);
        }
    }

    std::vector<ELSPricer::Real> V_0;
    copyFromDevice(V_0);

    return V_0;
}

PricingResultCrossTermGPU priceELSCrossTermGPU(
    const ELSProduct& product,
    int N1, int N2, int Nt,
    bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Create grid
    auto grid = createAdaptiveGrid(
        product.getS1_0(),
        product.getS2_0(),
        product.getMaturity(),
        N1, N2, Nt
    );

    if (verbose) {
        std::cout << "\n=== ELS Pricing (GPU ADI with Cross-Term) ===\n";
        grid->printInfo();
    }

    // Create terminal payoff
    std::vector<ELSPricer::Real> V_T(N1 * N2);
    const auto& S1 = grid->getS1();
    const auto& S2 = grid->getS2();

    bool kiOccurred = false;

    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            V_T[i * N2 + j] = product.payoffAtMaturity(S1[i], S2[j], kiOccurred);
        }
    }

    // Solve with GPU cross-term
    CUDAADISolverCrossTerm solver(*grid, product);
    auto V_0 = solver.solveWithEarlyRedemption(V_T, product);

    // Extract price at (S1_0, S2_0)
    int i0 = grid->findS1Index(product.getS1_0());
    int j0 = grid->findS2Index(product.getS2_0());
    ELSPricer::Real price = V_0[i0 * N2 + j0];

    auto end = std::chrono::high_resolution_clock::now();
    ELSPricer::Real computeTime = std::chrono::duration<ELSPricer::Real>(end - start).count();

    if (verbose) {
        std::cout << "\n--- Pricing Result (GPU with Cross-Term) ---\n";
        std::cout << "ELS Price: " << price << "\n";
        std::cout << "Compute Time: " << computeTime << " seconds\n";
    }

    return PricingResultCrossTermGPU{price, V_0, computeTime};
}

} // namespace CUDA
} // namespace ELSPricer
