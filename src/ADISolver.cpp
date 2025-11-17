#include "ADISolver.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace ELSPricer {

ADISolver::ADISolver(const Grid2D& grid, const ELSProduct& product)
    : grid_(grid), product_(product)
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

    precomputeCoefficients();
}

void ADISolver::precomputeCoefficients() {
    // Allocate arrays
    alpha1_.resize(N1_ - 1);
    beta1_.resize(N1_);
    gamma1_.resize(N1_ - 1);

    alpha2_.resize(N2_ - 1);
    beta2_.resize(N2_);
    gamma2_.resize(N2_ - 1);

    // S1 direction coefficients
    for (int i = 1; i < N1_ - 1; ++i) {
        Real S1 = grid_.getS1(i);

        Real a1 = 0.5 * sigma1_ * sigma1_ * S1 * S1 / (dS1_ * dS1_);
        Real b1 = (r_ - q1_) * S1 / (2.0 * dS1_);

        // Tridiagonal coefficients
        alpha1_[i - 1] = -0.5 * dt_ * (a1 - b1);  // lower diagonal
        beta1_[i] = 1.0 + dt_ * (a1 + 0.5 * r_);  // main diagonal
        gamma1_[i] = -0.5 * dt_ * (a1 + b1);      // upper diagonal
    }

    // Boundary conditions
    beta1_[0] = 1.0;
    beta1_[N1_ - 1] = 1.0;

    // S2 direction coefficients
    for (int j = 1; j < N2_ - 1; ++j) {
        Real S2 = grid_.getS2(j);

        Real a2 = 0.5 * sigma2_ * sigma2_ * S2 * S2 / (dS2_ * dS2_);
        Real b2 = (r_ - q2_) * S2 / (2.0 * dS2_);

        alpha2_[j - 1] = -0.5 * dt_ * (a2 - b2);
        beta2_[j] = 1.0 + dt_ * (a2 + 0.5 * r_);
        gamma2_[j] = -0.5 * dt_ * (a2 + b2);
    }

    beta2_[0] = 1.0;
    beta2_[N2_ - 1] = 1.0;
}

void ADISolver::thomasAlgorithm(
    const std::vector<Real>& lower,
    const std::vector<Real>& diag,
    const std::vector<Real>& upper,
    const std::vector<Real>& rhs,
    std::vector<Real>& solution)
{
    int n = diag.size();
    solution.resize(n);

    std::vector<Real> c_prime(n - 1);
    std::vector<Real> d_prime(n);

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (int i = 1; i < n - 1; ++i) {
        Real denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        c_prime[i] = upper[i] / denom;
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    // Last row
    int i = n - 1;
    Real denom = diag[i] - lower[i - 1] * c_prime[i - 1];
    d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;

    // Backward substitution
    solution[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
}

void ADISolver::solveS1Direction(const std::vector<Real>& V_in, std::vector<Real>& V_out) {
    V_out.resize(N1_ * N2_);

    // For each S2 slice, solve tridiagonal system in S1 direction
    for (int j = 0; j < N2_; ++j) {
        std::vector<Real> rhs(N1_);
        std::vector<Real> sol(N1_);

        // Extract column
        for (int i = 0; i < N1_; ++i) {
            rhs[i] = V_in[i * N2_ + j];
        }

        // Boundary conditions
        rhs[0] = 0.0;
        // rhs[N1_ - 1] is kept as is

        // Solve tridiagonal system
        thomasAlgorithm(alpha1_, beta1_, gamma1_, rhs, sol);

        // Store result
        for (int i = 0; i < N1_; ++i) {
            V_out[i * N2_ + j] = sol[i];
        }
    }
}

void ADISolver::solveS2Direction(const std::vector<Real>& V_in, std::vector<Real>& V_out) {
    V_out.resize(N1_ * N2_);

    // For each S1 slice, solve tridiagonal system in S2 direction
    for (int i = 0; i < N1_; ++i) {
        std::vector<Real> rhs(N2_);
        std::vector<Real> sol(N2_);

        // Extract row
        for (int j = 0; j < N2_; ++j) {
            rhs[j] = V_in[i * N2_ + j];
        }

        // Boundary conditions
        rhs[0] = 0.0;
        // rhs[N2_ - 1] is kept as is

        // Solve tridiagonal system
        thomasAlgorithm(alpha2_, beta2_, gamma2_, rhs, sol);

        // Store result
        for (int j = 0; j < N2_; ++j) {
            V_out[i * N2_ + j] = sol[j];
        }
    }
}

void ADISolver::applyBoundaryConditions(std::vector<Real>& V) {
    // S1 = 0: V = 0
    for (int j = 0; j < N2_; ++j) {
        V[0 * N2_ + j] = 0.0;
    }

    // S2 = 0: V = 0
    for (int i = 0; i < N1_; ++i) {
        V[i * N2_ + 0] = 0.0;
    }

    // S1 = S1_max: Linear extrapolation
    for (int j = 0; j < N2_; ++j) {
        V[(N1_ - 1) * N2_ + j] = 2.0 * V[(N1_ - 2) * N2_ + j] - V[(N1_ - 3) * N2_ + j];
    }

    // S2 = S2_max: Linear extrapolation
    for (int i = 0; i < N1_; ++i) {
        V[i * N2_ + (N2_ - 1)] = 2.0 * V[i * N2_ + (N2_ - 2)] - V[i * N2_ + (N2_ - 3)];
    }
}

void ADISolver::applyEarlyRedemption(
    std::vector<Real>& V,
    int obsIdx,
    const ELSProduct& product)
{
    const auto& S1 = grid_.getS1();
    const auto& S2 = grid_.getS2();

    int redeemed_count = 0;
    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            auto result = product.checkEarlyRedemption(S1[i], S2[j], obsIdx);
            if (result.isRedeemed) {
                // Early redemption is mandatory, not optional
                // When condition is met, investor receives the payoff immediately
                V[i * N2_ + j] = result.payoff;
                redeemed_count++;
            }
        }
    }

    // Debug output
    Real percentage = 100.0 * redeemed_count / (N1_ * N2_);
    std::cout << "  [DEBUG] Observation " << obsIdx
              << " (t=" << product.getObservationDates()[obsIdx] << "): "
              << redeemed_count << " / " << (N1_ * N2_)
              << " points redeemed (" << std::fixed << std::setprecision(1)
              << percentage << "%)" << std::endl;  // Force flush
}

std::vector<Real> ADISolver::solve(const std::vector<Real>& V_T) {
    std::vector<Real> V = V_T;
    std::vector<Real> V_half(N1_ * N2_);

    // Time stepping (backward in time: T -> 0)
    for (int n = Nt_ - 1; n >= 0; --n) {
        // Half-step 1: S1 direction implicit
        solveS1Direction(V, V_half);

        // Half-step 2: S2 direction implicit
        solveS2Direction(V_half, V);

        // Boundary conditions
        applyBoundaryConditions(V);
    }

    return V;
}

std::vector<Real> ADISolver::solveWithEarlyRedemption(
    const std::vector<Real>& V_T,
    const ELSProduct& product)
{
    std::vector<Real> V = V_T;
    std::vector<Real> V_half(N1_ * N2_);

    const auto& obsDates = product.getObservationDates();
    const auto& timeGrid = grid_.getTime();

    // Convert observation dates to time indices
    std::vector<int> obsIndices;
    std::cout << "[DEBUG] Matching observation dates to timesteps:\n";
    for (Real obsDate : obsDates) {
        int idx = 0;
        Real minDiff = std::abs(timeGrid[0] - obsDate);
        for (int n = 1; n <= Nt_; ++n) {
            Real diff = std::abs(timeGrid[n] - obsDate);
            if (diff < minDiff) {
                minDiff = diff;
                idx = n;
            }
        }
        obsIndices.push_back(idx);
        std::cout << "  t=" << obsDate << " -> timestep " << idx
                  << " (actual t=" << timeGrid[idx] << ")" << std::endl;
    }

    int obsIdx = static_cast<int>(obsDates.size()) - 1;

    // Time stepping (backward from Nt_ to 1)
    // Note: We go from Nt_ to 1 (not 0) because:
    // - At n=Nt_ we only check early redemption (no PDE step)
    // - PDE steps go from n=Nt_-1 down to n=1
    for (int n = Nt_; n >= 1; --n) {
        // Check early redemption BEFORE PDE step
        if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
            std::cout << "[DEBUG] Applying early redemption at timestep " << n << std::endl;
            applyEarlyRedemption(V, obsIdx, product);
            --obsIdx;
        }

        // ADI steps (except at n=Nt_ which is terminal condition)
        if (n < Nt_) {
            solveS1Direction(V, V_half);
            solveS2Direction(V_half, V);
            applyBoundaryConditions(V);
        }
    }

    return V;
}

PricingResult priceELS(
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
        std::cout << "\n=== ELS Pricing (CPU ADI Solver) ===\n";
        grid->printInfo();
    }

    // Create terminal payoff
    std::vector<Real> V_T(N1 * N2);
    const auto& S1 = grid->getS1();
    const auto& S2 = grid->getS2();

    // TODO: Track KI properly (for now assume no KI)
    bool kiOccurred = false;

    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            V_T[i * N2 + j] = product.payoffAtMaturity(S1[i], S2[j], kiOccurred);
        }
    }

    // Solve
    ADISolver solver(*grid, product);
    auto V_0 = solver.solveWithEarlyRedemption(V_T, product);

    // Extract price at (S1_0, S2_0)
    int i0 = grid->findS1Index(product.getS1_0());
    int j0 = grid->findS2Index(product.getS2_0());
    Real price = V_0[i0 * N2 + j0];

    auto end = std::chrono::high_resolution_clock::now();
    Real computeTime = std::chrono::duration<Real>(end - start).count();

    if (verbose) {
        std::cout << "\n--- Pricing Result ---\n";
        std::cout << "ELS Price: " << price << "\n";
        std::cout << "Compute Time: " << computeTime << " seconds\n";
    }

    return PricingResult{price, V_0, computeTime};
}

} // namespace ELSPricer
