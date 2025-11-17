#include "ADISolverCrossTerm.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace ELSPricer {

ADISolverCrossTerm::ADISolverCrossTerm(const Grid2D& grid, const ELSProduct& product)
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

void ADISolverCrossTerm::precomputeCoefficients() {
    // Allocate arrays
    alpha1_.resize(N1_ - 1);
    beta1_.resize(N1_);
    gamma1_.resize(N1_ - 1);

    alpha2_.resize(N2_ - 1);
    beta2_.resize(N2_);
    gamma2_.resize(N2_ - 1);

    // Precompute cross-term coefficients: ρ σ1 σ2 S1[i] S2[j] / (4 dS1 dS2)
    cross_coef_.resize(N1_ * N2_);
    const auto& S1 = grid_.getS1();
    const auto& S2 = grid_.getS2();

    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            cross_coef_[i * N2_ + j] =
                rho_ * sigma1_ * sigma2_ * S1[i] * S2[j] / (4.0 * dS1_ * dS2_);
        }
    }

    // S1 direction coefficients (same as before)
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

    // S2 direction coefficients (same as before)
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

void ADISolverCrossTerm::computeCrossTerm(
    const std::vector<double>& V,
    std::vector<double>& cross)
{
    cross.resize(N1_ * N2_, 0.0);

    // Compute mixed derivative: ∂²V/∂S1∂S2 using 4-point stencil
    // ∂²V/∂S1∂S2 ≈ [V(i+1,j+1) - V(i+1,j-1) - V(i-1,j+1) + V(i-1,j-1)] / (4 dS1 dS2)

    for (int i = 1; i < N1_ - 1; ++i) {
        for (int j = 1; j < N2_ - 1; ++j) {
            double mixed_deriv =
                V[(i+1) * N2_ + (j+1)] - V[(i+1) * N2_ + (j-1)] -
                V[(i-1) * N2_ + (j+1)] + V[(i-1) * N2_ + (j-1)];

            // cross = ρ σ1 σ2 S1 S2 ∂²V/∂S1∂S2
            cross[i * N2_ + j] = cross_coef_[i * N2_ + j] * mixed_deriv;
        }
    }
}

void ADISolverCrossTerm::addCrossTermToRHS(
    const std::vector<double>& V,
    std::vector<double>& RHS)
{
    std::vector<double> cross;
    computeCrossTerm(V, cross);

    RHS.resize(N1_ * N2_);
    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            // RHS = V + 0.5 * dt * cross_term
            RHS[i * N2_ + j] = V[i * N2_ + j] + 0.5 * dt_ * cross[i * N2_ + j];
        }
    }
}

void ADISolverCrossTerm::thomasAlgorithm(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
    std::vector<double>& solution)
{
    int n = diag.size();
    solution.resize(n);

    std::vector<double> c_prime(n - 1);
    std::vector<double> d_prime(n);

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (int i = 1; i < n - 1; ++i) {
        double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        c_prime[i] = upper[i] / denom;
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    int i = n - 1;
    double denom = diag[i] - lower[i - 1] * c_prime[i - 1];
    d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;

    // Backward substitution
    solution[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
}

void ADISolverCrossTerm::solveS1Direction(
    const std::vector<double>& V_in,
    std::vector<double>& V_out)
{
    V_out.resize(N1_ * N2_);

    for (int j = 0; j < N2_; ++j) {
        std::vector<double> rhs(N1_);
        std::vector<double> sol(N1_);

        for (int i = 0; i < N1_; ++i) {
            rhs[i] = V_in[i * N2_ + j];
        }

        rhs[0] = 0.0;

        thomasAlgorithm(alpha1_, beta1_, gamma1_, rhs, sol);

        for (int i = 0; i < N1_; ++i) {
            V_out[i * N2_ + j] = sol[i];
        }
    }
}

void ADISolverCrossTerm::solveS2Direction(
    const std::vector<double>& V_in,
    std::vector<double>& V_out)
{
    V_out.resize(N1_ * N2_);

    for (int i = 0; i < N1_; ++i) {
        std::vector<double> rhs(N2_);
        std::vector<double> sol(N2_);

        for (int j = 0; j < N2_; ++j) {
            rhs[j] = V_in[i * N2_ + j];
        }

        rhs[0] = 0.0;

        thomasAlgorithm(alpha2_, beta2_, gamma2_, rhs, sol);

        for (int j = 0; j < N2_; ++j) {
            V_out[i * N2_ + j] = sol[j];
        }
    }
}

void ADISolverCrossTerm::applyBoundaryConditions(std::vector<double>& V) {
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

void ADISolverCrossTerm::applyEarlyRedemption(
    std::vector<double>& V,
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
                V[i * N2_ + j] = result.payoff;
                redeemed_count++;
            }
        }
    }

    double percentage = 100.0 * redeemed_count / (N1_ * N2_);
    std::cout << "  [DEBUG] Observation " << obsIdx
              << " (t=" << product.getObservationDates()[obsIdx] << "): "
              << redeemed_count << " / " << (N1_ * N2_)
              << " points redeemed (" << std::fixed << std::setprecision(1)
              << percentage << "%)" << std::endl;
}

std::vector<double> ADISolverCrossTerm::solve(const std::vector<double>& V_T) {
    std::vector<double> V = V_T;
    std::vector<double> RHS1(N1_ * N2_);
    std::vector<double> V_half(N1_ * N2_);
    std::vector<double> RHS2(N1_ * N2_);

    // Time stepping (backward in time: T -> 0)
    for (int n = Nt_ - 1; n >= 0; --n) {
        // Step 1: Add cross-term to RHS
        addCrossTermToRHS(V, RHS1);

        // Step 2: S1 direction implicit
        solveS1Direction(RHS1, V_half);

        // Step 3: Add cross-term for V_half to RHS
        addCrossTermToRHS(V_half, RHS2);

        // Step 4: S2 direction implicit
        solveS2Direction(RHS2, V);

        // Boundary conditions
        applyBoundaryConditions(V);
    }

    return V;
}

std::vector<double> ADISolverCrossTerm::solveWithEarlyRedemption(
    const std::vector<double>& V_T,
    const ELSProduct& product)
{
    std::vector<double> V = V_T;
    std::vector<double> RHS1(N1_ * N2_);
    std::vector<double> V_half(N1_ * N2_);
    std::vector<double> RHS2(N1_ * N2_);

    const auto& obsDates = product.getObservationDates();
    const auto& timeGrid = grid_.getTime();

    // Convert observation dates to time indices
    std::vector<int> obsIndices;
    std::cout << "[DEBUG] Matching observation dates to timesteps:\n";
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
        std::cout << "  t=" << obsDate << " -> timestep " << idx
                  << " (actual t=" << timeGrid[idx] << ")" << std::endl;
    }

    int obsIdx = static_cast<int>(obsDates.size()) - 1;

    // Time stepping (backward from Nt_ to 1)
    for (int n = Nt_; n >= 1; --n) {
        // Check early redemption BEFORE PDE step
        if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
            std::cout << "[DEBUG] Applying early redemption at timestep " << n << std::endl;
            applyEarlyRedemption(V, obsIdx, product);
            --obsIdx;
        }

        // ADI steps with cross-term
        if (n < Nt_) {
            addCrossTermToRHS(V, RHS1);
            solveS1Direction(RHS1, V_half);
            addCrossTermToRHS(V_half, RHS2);
            solveS2Direction(RHS2, V);
            applyBoundaryConditions(V);
        }
    }

    return V;
}

PricingResultCrossTerm priceELSCrossTerm(
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
        std::cout << "\n=== ELS Pricing (CPU ADI with Cross-Term) ===\n";
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

    // Solve with cross-term
    ADISolverCrossTerm solver(*grid, product);
    auto V_0 = solver.solveWithEarlyRedemption(V_T, product);

    // Extract price at (S1_0, S2_0)
    int i0 = grid->findS1Index(product.getS1_0());
    int j0 = grid->findS2Index(product.getS2_0());
    double price = V_0[i0 * N2 + j0];

    auto end = std::chrono::high_resolution_clock::now();
    double computeTime = std::chrono::duration<double>(end - start).count();

    if (verbose) {
        std::cout << "\n--- Pricing Result (with Cross-Term) ---\n";
        std::cout << "ELS Price: " << price << "\n";
        std::cout << "Compute Time: " << computeTime << " seconds\n";
    }

    return PricingResultCrossTerm{price, V_0, computeTime};
}

} // namespace ELSPricer
