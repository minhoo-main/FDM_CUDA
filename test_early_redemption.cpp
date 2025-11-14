#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>
#include <iomanip>

using namespace ELSPricer;

int main() {
    // Simple test: S1=S2=100 (at the money), check 0.5년 observation

    ELSProduct product(
        100.0, 3.0,
        {0.5, 1.0, 1.5, 2.0, 2.5, 3.0},
        {0.85, 0.85, 0.80, 0.80, 0.75, 0.70},
        {4.0, 8.0, 12.0, 16.0, 20.0, 24.0},
        0.45,
        100.0, 100.0,
        0.152, 0.404, 0.61,
        0.03477, 0.015, 0.02,
        true
    );

    // At S1=100, S2=100 → worst=100%
    // Barrier=85% → Should redeem!
    auto result = product.checkEarlyRedemption(100.0, 100.0, 0);

    std::cout << "Test at S1=100, S2=100 (observation 0, barrier 85%):\n";
    std::cout << "  Is Redeemed: " << (result.isRedeemed ? "YES" : "NO") << "\n";
    std::cout << "  Payoff: " << result.payoff << "\n";

    // Test at S1=80, S2=80 → worst=80%
    // Barrier=85% → Should NOT redeem
    result = product.checkEarlyRedemption(80.0, 80.0, 0);

    std::cout << "\nTest at S1=80, S2=80 (observation 0, barrier 85%):\n";
    std::cout << "  Is Redeemed: " << (result.isRedeemed ? "YES" : "NO") << "\n";
    std::cout << "  Payoff: " << result.payoff << "\n";

    // Now test full pricing with a small grid
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Full pricing test (10×10×50):\n\n";

    auto test_result = priceELS(product, 10, 10, 50, true);

    std::cout << "\nPrice: " << test_result.price << "\n";

    return 0;
}
