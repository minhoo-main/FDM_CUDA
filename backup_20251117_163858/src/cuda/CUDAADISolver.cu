#include "CUDAADISolver.cuh"
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
void applyEarlyRedemption(
    double* d_V,
    const double* d_S1,
    const double* d_S2,
    double S1_0,
    double S2_0,
    double barrier,
    double principal,
    double coupon,
    int N1,
    int N2);

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

CUDAADISolver::CUDAADISolver(const Grid2D& grid, const ELSProduct& product)
    : grid_(grid), product_(product),
      d_V_(nullptr), d_V_half_(nullptr), d_V_transposed_(nullptr),
      d_S1_(nullptr), d_S2_(nullptr),
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

CUDAADISolver::~CUDAADISolver() {
    cleanup();
}

void CUDAADISolver::precomputeCoefficients() {
    alpha1_.resize(N1_ - 1);
    beta1_.resize(N1_);
    gamma1_.resize(N1_ - 1);

    alpha2_.resize(N2_ - 1);
    beta2_.resize(N2_);
    gamma2_.resize(N2_ - 1);

    // S1 direction
    for (int i = 1; i < N1_ - 1; ++i) {
        double S1 = grid_.getS1(i);
        double a1 = 0.5 * sigma1_ * sigma1_ * S1 * S1 / (dS1_ * dS1_);
        double b1 = (r_ - q1_) * S1 / (2.0 * dS1_);

        alpha1_[i - 1] = -0.5 * dt_ * (a1 - b1);
        beta1_[i] = 1.0 + dt_ * (a1 + 0.5 * r_);
        gamma1_[i] = -0.5 * dt_ * (a1 + b1);
    }
    beta1_[0] = 1.0;
    beta1_[N1_ - 1] = 1.0;

    // S2 direction
    for (int j = 1; j < N2_ - 1; ++j) {
        double S2 = grid_.getS2(j);
        double a2 = 0.5 * sigma2_ * sigma2_ * S2 * S2 / (dS2_ * dS2_);
        double b2 = (r_ - q2_) * S2 / (2.0 * dS2_);

        alpha2_[j - 1] = -0.5 * dt_ * (a2 - b2);
        beta2_[j] = 1.0 + dt_ * (a2 + 0.5 * r_);
        gamma2_[j] = -0.5 * dt_ * (a2 + b2);
    }
    beta2_[0] = 1.0;
    beta2_[N2_ - 1] = 1.0;
}

void CUDAADISolver::initialize() {
    // Precompute coefficients on host
    precomputeCoefficients();

    // Allocate device memory
    size_t grid_size = N1_ * N2_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_V_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_V_half_, grid_size));
    CUDA_CHECK(cudaMalloc(&d_V_transposed_, grid_size));  // Pre-allocate transpose buffer

    // Allocate and copy S1, S2 grids to device
    CUDA_CHECK(cudaMalloc(&d_S1_, N1_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_S2_, N2_ * sizeof(double)));

    const auto& S1 = grid_.getS1();
    const auto& S2 = grid_.getS2();
    CUDA_CHECK(cudaMemcpy(d_S1_, S1.data(), N1_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S2_, S2.data(), N2_ * sizeof(double), cudaMemcpyHostToDevice));

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
    std::cout << "✓ GPU Initialized: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
}

void CUDAADISolver::cleanup() {
    if (d_V_) CUDA_CHECK(cudaFree(d_V_));
    if (d_V_half_) CUDA_CHECK(cudaFree(d_V_half_));
    if (d_V_transposed_) CUDA_CHECK(cudaFree(d_V_transposed_));
    if (d_S1_) CUDA_CHECK(cudaFree(d_S1_));
    if (d_S2_) CUDA_CHECK(cudaFree(d_S2_));
    if (d_alpha1_) CUDA_CHECK(cudaFree(d_alpha1_));
    if (d_beta1_) CUDA_CHECK(cudaFree(d_beta1_));
    if (d_gamma1_) CUDA_CHECK(cudaFree(d_gamma1_));
    if (d_alpha2_) CUDA_CHECK(cudaFree(d_alpha2_));
    if (d_beta2_) CUDA_CHECK(cudaFree(d_beta2_));
    if (d_gamma2_) CUDA_CHECK(cudaFree(d_gamma2_));
}

void CUDAADISolver::copyToDevice(const std::vector<double>& V_host) {
    CUDA_CHECK(cudaMemcpy(d_V_, V_host.data(), N1_ * N2_ * sizeof(double), cudaMemcpyHostToDevice));
}

void CUDAADISolver::copyFromDevice(std::vector<double>& V_host) {
    V_host.resize(N1_ * N2_);
    CUDA_CHECK(cudaMemcpy(V_host.data(), d_V_, N1_ * N2_ * sizeof(double), cudaMemcpyDeviceToHost));
}

void CUDAADISolver::solveS1DirectionGPU() {
    // For S1 direction, we need to solve N2 independent systems
    // Each system has size N1
    // V is stored as [N1 x N2] row-major
    // We need to transpose to [N2 x N1] for batched solve

    // Use pre-allocated transpose buffer (no malloc/free overhead)
    // Transpose: [N1 x N2] -> [N2 x N1]
    transpose(d_V_, d_V_transposed_, N1_, N2_);

    // Solve N2 tridiagonal systems in parallel
    batchedThomas(d_alpha1_, d_beta1_, d_gamma1_, d_V_transposed_, d_V_half_, N1_, N2_);

    // Transpose back: [N2 x N1] -> [N1 x N2]
    transpose(d_V_half_, d_V_half_, N2_, N1_);
}

void CUDAADISolver::solveS2DirectionGPU() {
    // For S2 direction, we solve N1 independent systems
    // Each system has size N2
    // V is already in row-major format [N1 x N2]

    batchedThomas(d_alpha2_, d_beta2_, d_gamma2_, d_V_half_, d_V_, N2_, N1_);
}

void CUDAADISolver::applyBoundaryConditionsGPU() {
    applyBoundaryConditions(d_V_, N1_, N2_);
}

std::vector<double> CUDAADISolver::solve(const std::vector<double>& V_T) {
    // Copy initial data to device
    copyToDevice(V_T);

    // Time stepping
    for (int n = Nt_ - 1; n >= 0; --n) {
        solveS1DirectionGPU();
        solveS2DirectionGPU();
        applyBoundaryConditionsGPU();
    }

    // Copy result back
    std::vector<double> V_0;
    copyFromDevice(V_0);

    return V_0;
}

std::vector<double> CUDAADISolver::solveWithEarlyRedemption(
    const std::vector<double>& V_T,
    const ELSProduct& product)
{
    copyToDevice(V_T);

    const auto& obsDates = product.getObservationDates();
    const auto& timeGrid = grid_.getTime();

    // Find observation indices
    std::vector<int> obsIndices;
    for (double obsDate : obsDates) {
        int idx = 0;
        double minDiff = std::abs(timeGrid[0] - obsDate);
        for (int n = 1; n <= Nt_; ++n) {
            double diff = std::abs(timeGrid[n] - obsDate);
            if (diff < minDiff) {
                minDiff = diff;
                idx = n;
            }
        }
        obsIndices.push_back(idx);
    }

    int obsIdx = static_cast<int>(obsDates.size()) - 1;

    // Time stepping (backward from Nt_ to 1)
    // Note: We go from Nt_ to 1 (not 0) because:
    // - At n=Nt_ we only check early redemption (no PDE step)
    // - PDE steps go from n=Nt_-1 down to n=1
    for (int n = Nt_; n >= 1; --n) {
        // Check early redemption BEFORE PDE step
        if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
            const auto& barriers = product.getRedemptionBarriers();
            const auto& coupons = product.getCoupons();

            applyEarlyRedemption(
                d_V_,
                d_S1_,
                d_S2_,
                product.getS1_0(),
                product.getS2_0(),
                barriers[obsIdx],
                product.getPrincipal(),
                coupons[obsIdx],
                N1_,
                N2_
            );

            --obsIdx;
        }

        // ADI steps (except at n=Nt_ which is terminal condition)
        if (n < Nt_) {
            solveS1DirectionGPU();
            solveS2DirectionGPU();
            applyBoundaryConditionsGPU();
        }
    }

    std::vector<double> V_0;
    copyFromDevice(V_0);

    return V_0;
}

PricingResult priceELSGPU(
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
        std::cout << "\n=== ELS Pricing (GPU CUDA Solver) ===\n";
        grid->printInfo();
    }

    // Create terminal payoff
    std::vector<double> V_T(N1 * N2);
    const auto& S1 = grid->getS1();
    const auto& S2 = grid->getS2();

    bool kiOccurred = false;
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            V_T[i * N2 + j] = product.payoffAtMaturity(S1[i], S2[j], kiOccurred);
        }
    }

    // Solve on GPU
    CUDAADISolver solver(*grid, product);

    // Ensure GPU is synchronized before timing
    cudaDeviceSynchronize();
    auto compute_start = std::chrono::high_resolution_clock::now();

    auto V_0 = solver.solveWithEarlyRedemption(V_T, product);

    // Synchronize GPU before stopping timer
    cudaDeviceSynchronize();
    auto compute_end = std::chrono::high_resolution_clock::now();

    // Extract price
    int i0 = grid->findS1Index(product.getS1_0());
    int j0 = grid->findS2Index(product.getS2_0());
    double price = V_0[i0 * N2 + j0];

    auto end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end - start).count();
    double computeTime = std::chrono::duration<double>(compute_end - compute_start).count();

    if (verbose) {
        std::cout << "\n--- Pricing Result (GPU) ---\n";
        std::cout << "ELS Price: " << price << "\n";
        std::cout << "Compute Time (pure): " << computeTime << " seconds\n";
        std::cout << "Total Time (with setup): " << totalTime << " seconds\n";
    }

    return PricingResult{price, V_0, computeTime};
}

} // namespace CUDA
} // namespace ELSPricer
