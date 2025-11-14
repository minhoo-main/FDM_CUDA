#include "ELSProduct.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ELSPricer {

ELSProduct::ELSProduct(
    double principal, double maturity,
    const std::vector<double>& observationDates,
    const std::vector<double>& redemptionBarriers,
    const std::vector<double>& coupons,
    double kiBarrier,
    double S1_0, double S2_0,
    double sigma1, double sigma2, double rho,
    double r, double q1, double q2,
    bool worstOf)
    : principal_(principal), maturity_(maturity),
      observationDates_(observationDates),
      redemptionBarriers_(redemptionBarriers),
      coupons_(coupons),
      kiBarrier_(kiBarrier),
      S1_0_(S1_0), S2_0_(S2_0),
      sigma1_(sigma1), sigma2_(sigma2), rho_(rho),
      r_(r), q1_(q1), q2_(q2),
      worstOf_(worstOf)
{
    initializeDefaults();
}

void ELSProduct::initializeDefaults() {
    // Set default observation dates if not provided (6-month intervals)
    if (observationDates_.empty()) {
        int numObs = static_cast<int>(maturity_ * 2); // every 6 months
        for (int i = 1; i <= numObs; ++i) {
            observationDates_.push_back(i * 0.5);
        }
    }

    // Set default step-down barriers if not provided
    if (redemptionBarriers_.empty()) {
        redemptionBarriers_ = {0.95, 0.95, 0.90, 0.85, 0.80, 0.75};
        redemptionBarriers_.resize(observationDates_.size(), 0.75);
    }

    // Set default coupons if not provided (8% annual)
    if (coupons_.empty()) {
        double annualCoupon = 8.0;
        for (double t : observationDates_) {
            coupons_.push_back(annualCoupon * t);
        }
    }
}

double ELSProduct::performance(double S1, double S2) const {
    double perf1 = S1 / S1_0_;
    double perf2 = S2 / S2_0_;

    if (worstOf_) {
        return std::min(perf1, perf2);
    } else {
        return std::max(perf1, perf2);
    }
}

double ELSProduct::payoffAtMaturity(double S1, double S2, bool kiOccurred) const {
    double perf = performance(S1, S2);

    if (kiOccurred) {
        // KI occurred: min(principal, principal * performance)
        return principal_ * std::min(1.0, perf);
    } else {
        // No KI: principal + final coupon
        return principal_ + coupons_.back();
    }
}

ELSProduct::RedemptionResult ELSProduct::checkEarlyRedemption(
    double S1, double S2, int obsIdx) const
{
    RedemptionResult result{false, 0.0};

    if (obsIdx < 0 || obsIdx >= static_cast<int>(observationDates_.size())) {
        return result;
    }

    double perf = performance(S1, S2);
    double barrier = redemptionBarriers_[obsIdx];

    if (perf >= barrier) {
        result.isRedeemed = true;
        result.payoff = principal_ + coupons_[obsIdx];
    }

    return result;
}

bool ELSProduct::checkKnockIn(double S1, double S2) const {
    double perf = performance(S1, S2);
    return perf < kiBarrier_;
}

void ELSProduct::printInfo() const {
    std::cout << "\nELS Product (Step-Down, " << (worstOf_ ? "Worst-of" : "Best-of") << ")\n";
    std::cout << "================================\n";
    std::cout << "Principal: " << principal_ << "\n";
    std::cout << "Maturity: " << maturity_ << " years\n";
    std::cout << "KI Barrier: " << (kiBarrier_ * 100) << "%\n\n";

    std::cout << "Observation Dates and Conditions:\n";
    for (size_t i = 0; i < observationDates_.size(); ++i) {
        std::cout << "  " << std::setw(2) << (i + 1) << ". "
                  << "T=" << std::setw(4) << observationDates_[i]
                  << "  Barrier=" << std::setw(5) << (redemptionBarriers_[i] * 100) << "%"
                  << "  Coupon=" << std::setw(6) << coupons_[i] << "%\n";
    }

    std::cout << "\nUnderlying Assets:\n";
    std::cout << "  S1: " << S1_0_ << " (σ=" << sigma1_ << ", q=" << q1_ << ")\n";
    std::cout << "  S2: " << S2_0_ << " (σ=" << sigma2_ << ", q=" << q2_ << ")\n";
    std::cout << "  Correlation: " << rho_ << "\n";

    std::cout << "\nMarket Parameters:\n";
    std::cout << "  Risk-free rate: " << r_ << "\n";
}

ELSProduct createSampleELS() {
    return ELSProduct(
        100.0,  // principal
        3.0,    // maturity
        {0.5, 1.0, 1.5, 2.0, 2.5, 3.0},  // observation dates
        {0.95, 0.95, 0.90, 0.85, 0.80, 0.75},  // redemption barriers
        {4.0, 8.0, 12.0, 16.0, 20.0, 24.0},  // coupons (8% annual)
        0.50,   // KI barrier
        100.0,  // S1_0
        100.0,  // S2_0
        0.25,   // sigma1
        0.30,   // sigma2
        0.50,   // rho
        0.03,   // r
        0.02,   // q1
        0.015,  // q2
        true    // worst-of
    );
}

} // namespace ELSPricer
