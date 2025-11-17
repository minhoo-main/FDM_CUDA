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
        double principal = 100.0,
        double maturity = 3.0,
        const std::vector<double>& observationDates = {},
        const std::vector<double>& redemptionBarriers = {},
        const std::vector<double>& coupons = {},
        double kiBarrier = 0.50,
        double S1_0 = 100.0,
        double S2_0 = 100.0,
        double sigma1 = 0.25,
        double sigma2 = 0.30,
        double rho = 0.50,
        double r = 0.03,
        double q1 = 0.02,
        double q2 = 0.015,
        bool worstOf = true
    );

    // Accessors
    inline double getPrincipal() const { return principal_; }
    inline double getMaturity() const { return maturity_; }
    inline double getS1_0() const { return S1_0_; }
    inline double getS2_0() const { return S2_0_; }
    inline double getSigma1() const { return sigma1_; }
    inline double getSigma2() const { return sigma2_; }
    inline double getRho() const { return rho_; }
    inline double getR() const { return r_; }
    inline double getQ1() const { return q1_; }
    inline double getQ2() const { return q2_; }
    inline double getKIBarrier() const { return kiBarrier_; }
    inline bool isWorstOf() const { return worstOf_; }

    inline const std::vector<double>& getObservationDates() const { return observationDates_; }
    inline const std::vector<double>& getRedemptionBarriers() const { return redemptionBarriers_; }
    inline const std::vector<double>& getCoupons() const { return coupons_; }

    // Payoff calculation
    double payoffAtMaturity(double S1, double S2, bool kiOccurred) const;

    // Early redemption check
    struct RedemptionResult {
        bool isRedeemed;
        double payoff;
    };
    RedemptionResult checkEarlyRedemption(double S1, double S2, int obsIdx) const;

    // Knock-in check
    bool checkKnockIn(double S1, double S2) const;

    // Performance (Worst-of or Best-of)
    double performance(double S1, double S2) const;

    // Print product info
    void printInfo() const;

private:
    // Product parameters
    double principal_;
    double maturity_;

    std::vector<double> observationDates_;
    std::vector<double> redemptionBarriers_;
    std::vector<double> coupons_;

    double kiBarrier_;

    // Underlying assets
    double S1_0_, S2_0_;
    double sigma1_, sigma2_;
    double rho_;

    // Market parameters
    double r_;   // risk-free rate
    double q1_;  // dividend yield 1
    double q2_;  // dividend yield 2

    bool worstOf_;  // Worst-of (true) or Best-of (false)

    // Initialize default values
    void initializeDefaults();
};

/**
 * Create a sample Step-Down ELS product
 */
ELSProduct createSampleELS();

} // namespace ELSPricer
