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
        float principal = 100.0,
        float maturity = 3.0,
        const std::vector<float>& observationDates = {},
        const std::vector<float>& redemptionBarriers = {},
        const std::vector<float>& coupons = {},
        float kiBarrier = 0.50,
        float S1_0 = 100.0,
        float S2_0 = 100.0,
        float sigma1 = 0.25,
        float sigma2 = 0.30,
        float rho = 0.50,
        float r = 0.03,
        float q1 = 0.02,
        float q2 = 0.015,
        bool worstOf = true
    );

    // Accessors
    inline float getPrincipal() const { return principal_; }
    inline float getMaturity() const { return maturity_; }
    inline float getS1_0() const { return S1_0_; }
    inline float getS2_0() const { return S2_0_; }
    inline float getSigma1() const { return sigma1_; }
    inline float getSigma2() const { return sigma2_; }
    inline float getRho() const { return rho_; }
    inline float getR() const { return r_; }
    inline float getQ1() const { return q1_; }
    inline float getQ2() const { return q2_; }
    inline float getKIBarrier() const { return kiBarrier_; }
    inline bool isWorstOf() const { return worstOf_; }

    inline const std::vector<float>& getObservationDates() const { return observationDates_; }
    inline const std::vector<float>& getRedemptionBarriers() const { return redemptionBarriers_; }
    inline const std::vector<float>& getCoupons() const { return coupons_; }

    // Payoff calculation
    float payoffAtMaturity(float S1, float S2, bool kiOccurred) const;

    // Early redemption check
    struct RedemptionResult {
        bool isRedeemed;
        float payoff;
    };
    RedemptionResult checkEarlyRedemption(float S1, float S2, int obsIdx) const;

    // Knock-in check
    bool checkKnockIn(float S1, float S2) const;

    // Performance (Worst-of or Best-of)
    float performance(float S1, float S2) const;

    // Print product info
    void printInfo() const;

private:
    // Product parameters
    float principal_;
    float maturity_;

    std::vector<float> observationDates_;
    std::vector<float> redemptionBarriers_;
    std::vector<float> coupons_;

    float kiBarrier_;

    // Underlying assets
    float S1_0_, S2_0_;
    float sigma1_, sigma2_;
    float rho_;

    // Market parameters
    float r_;   // risk-free rate
    float q1_;  // dividend yield 1
    float q2_;  // dividend yield 2

    bool worstOf_;  // Worst-of (true) or Best-of (false)

    // Initialize default values
    void initializeDefaults();
};

/**
 * Create a sample Step-Down ELS product
 */
ELSProduct createSampleELS();

} // namespace ELSPricer
