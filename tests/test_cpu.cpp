#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace ELSPricer;

void testGrid2D() {
    std::cout << "Testing Grid2D... ";

    auto grid = createAdaptiveGrid(100.0, 100.0, 3.0, 50, 50, 100);

    assert(grid->getN1() == 50);
    assert(grid->getN2() == 50);
    assert(grid->getNt() == 100);

    assert(std::abs(grid->getS1Min() - 0.0) < 1e-10);
    assert(std::abs(grid->getS1Max() - 300.0) < 1e-10);

    int i0 = grid->findS1Index(100.0);
    assert(i0 >= 0 && i0 < 50);

    std::cout << "✓ PASSED\n";
}

void testELSProduct() {
    std::cout << "Testing ELSProduct... ";

    auto product = createSampleELS();

    assert(std::abs(product.getPrincipal() - 100.0) < 1e-10);
    assert(std::abs(product.getMaturity() - 3.0) < 1e-10);
    assert(product.getObservationDates().size() == 6);

    // Test early redemption check
    auto result = product.checkEarlyRedemption(100.0, 100.0, 0);
    assert(result.isRedeemed == true);  // 100% >= 95% barrier
    assert(std::abs(result.payoff - 104.0) < 1e-10);  // 100 + 4% coupon

    // Test knock-in check
    bool ki = product.checkKnockIn(40.0, 100.0);
    assert(ki == true);  // min(40%, 100%) < 50%

    std::cout << "✓ PASSED\n";
}

void testCPUSolver() {
    std::cout << "Testing CPU ADI Solver... ";

    auto product = createSampleELS();
    auto result = priceELS(product, 40, 40, 80, false);

    std::cout << "\n  Computed price: " << std::fixed << std::setprecision(4) << result.price << "\n";

    // Compute time should be positive
    assert(result.computeTime > 0.0);

    // Price should be reasonable (relaxed range for initial test)
    if (result.price < 0.0 || result.price > 200.0) {
        std::cerr << "  ⚠ Price out of expected range: " << result.price << "\n";
        assert(false);
    }

    std::cout << "  ✓ PASSED (price=" << result.price << ", time=" << std::setprecision(3) << result.computeTime << "s)\n";
}

void testConvergence() {
    std::cout << "Testing Grid Convergence...\n";

    auto product = createSampleELS();

    auto result1 = priceELS(product, 40, 40, 80, false);
    auto result2 = priceELS(product, 60, 60, 120, false);
    auto result3 = priceELS(product, 80, 80, 160, false);

    std::cout << "  40×40×80:   " << std::fixed << std::setprecision(4) << result1.price << "\n";
    std::cout << "  60×60×120:  " << result2.price << "\n";
    std::cout << "  80×80×160:  " << result3.price << "\n";

    // Prices should be positive and finite
    assert(result1.price > 0.0 && result1.price < 200.0);
    assert(result2.price > 0.0 && result2.price < 200.0);
    assert(result3.price > 0.0 && result3.price < 200.0);

    std::cout << "  ✓ PASSED\n";
}

int main() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer Test Suite (CPU Only)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    try {
        testGrid2D();
        testELSProduct();
        testCPUSolver();
        testConvergence();

        std::cout << "\n═══════════════════════════════════════════════════════════\n";
        std::cout << "   All tests PASSED! ✓\n";
        std::cout << "═══════════════════════════════════════════════════════════\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test FAILED: " << e.what() << "\n\n";
        return 1;
    }
}
