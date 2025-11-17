#include "precision.h"
#pragma once

#include <vector>
#include <string>

namespace ELSPricer {

/**
 * Step-Down ELS (Equity Linked Securities) Product
 *
 * Features:
 * - Multiple early redemption dates (Step-Down barriers)
 * - Knock-In barrier
 * - Worst-of structure
 */
class ELSProduct {
public:
    // Constructor
    ELSProduct(
        Real principal = 100.0,
        Real maturity = 3.0,
        const std::vector<Real>& observationDates = {},
        const std::vector<Real>& redemptionBarriers = {},
        const std::vector<Real>& coupons = {},
        Real kiBarrier = 0.50,
        Real S1_0 = 100.0,
        Real S2_0 = 100.0,
        Real sigma1 = 0.25,
        Real sigma2 = 0.30,
        Real rho = 0.50,
        Real r = 0.03,
        Real q1 = 0.02,
        Real q2 = 0.015,
        bool worstOf = true
    );

    // Accessors
    inline Real getPrincipal() const { return principal_; }
    inline Real getMaturity() const { return maturity_; }
    inline Real getS1_0() const { return S1_0_; }
    inline Real getS2_0() const { return S2_0_; }
    inline Real getSigma1() const { return sigma1_; }
    inline Real getSigma2() const { return sigma2_; }
    inline Real getRho() const { return rho_; }
    inline Real getR() const { return r_; }
    inline Real getQ1() const { return q1_; }
    inline Real getQ2() const { return q2_; }
    inline Real getKIBarrier() const { return kiBarrier_; }
    inline bool isWorstOf() const { return worstOf_; }

    inline const std::vector<Real>& getObservationDates() const { return observationDates_; }
    inline const std::vector<Real>& getRedemptionBarriers() const { return redemptionBarriers_; }
    inline const std::vector<Real>& getCoupons() const { return coupons_; }

    // Payoff calculation
    Real payoffAtMaturity(Real S1, Real S2, bool kiOccurred) const;

    // Early redemption check
    struct RedemptionResult {
        bool isRedeemed;
        Real payoff;
    };
    RedemptionResult checkEarlyRedemption(Real S1, Real S2, int obsIdx) const;

    // Knock-in check
    bool checkKnockIn(Real S1, Real S2) const;

    // Performance (Worst-of or Best-of)
    Real performance(Real S1, Real S2) const;

    // Print product info
    void printInfo() const;

private:
    // Product parameters
    Real principal_;
    Real maturity_;

    std::vector<Real> observationDates_;
    std::vector<Real> redemptionBarriers_;
    std::vector<Real> coupons_;

    Real kiBarrier_;

    // Underlying assets
    Real S1_0_, S2_0_;
    Real sigma1_, sigma2_;
    Real rho_;

    // Market parameters
    Real r_;   // risk-free rate
    Real q1_;  // dividend yield 1
    Real q2_;  // dividend yield 2

    bool worstOf_;  // Worst-of (true) or Best-of (false)

    // Initialize default values
    void initializeDefaults();
};

/**
 * Create a sample Step-Down ELS product
 */
ELSProduct createSampleELS();

} // namespace ELSPricer
