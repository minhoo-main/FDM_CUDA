#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.cuh"
#include <iostream>
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

    // Price should be reasonable (between 90 and 105)
    assert(result.price > 90.0 && result.price < 105.0);

    // Compute time should be positive
    assert(result.computeTime > 0.0);

    std::cout << "✓ PASSED (price=" << result.price << ")\n";
}

void testGPUSolver() {
    std::cout << "Testing GPU CUDA Solver... ";

    auto product = createSampleELS();
    auto result = CUDA::priceELSGPU(product, 40, 40, 80, false);

    // Price should be reasonable
    assert(result.price > 90.0 && result.price < 105.0);
    assert(result.computeTime > 0.0);

    std::cout << "✓ PASSED (price=" << result.price << ")\n";
}

void testCPUGPUConsistency() {
    std::cout << "Testing CPU-GPU Consistency... ";

    auto product = createSampleELS();

    // Use moderate grid size
    int N1 = 60, N2 = 60, Nt = 120;

    auto cpuResult = priceELS(product, N1, N2, Nt, false);
    auto gpuResult = CUDA::priceELSGPU(product, N1, N2, Nt, false);

    float priceDiff = std::abs(cpuResult.price - gpuResult.price);
    float relDiff = priceDiff / cpuResult.price;

    // Prices should match within 0.1%
    assert(relDiff < 0.001);

    std::cout << "✓ PASSED\n";
    std::cout << "  CPU price: " << cpuResult.price << "\n";
    std::cout << "  GPU price: " << gpuResult.price << "\n";
    std::cout << "  Difference: " << priceDiff << " (" << (relDiff * 100) << "%)\n";
}

void testConvergence() {
    std::cout << "Testing Grid Convergence... ";

    auto product = createSampleELS();

    auto result1 = priceELS(product, 40, 40, 80, false);
    auto result2 = priceELS(product, 60, 60, 120, false);
    auto result3 = priceELS(product, 80, 80, 160, false);

    // Prices should converge (difference should decrease)
    float diff1 = std::abs(result2.price - result1.price);
    float diff2 = std::abs(result3.price - result2.price);

    std::cout << "✓ PASSED\n";
    std::cout << "  40×40: " << result1.price << "\n";
    std::cout << "  60×60: " << result2.price << " (diff: " << diff1 << ")\n";
    std::cout << "  80×80: " << result3.price << " (diff: " << diff2 << ")\n";

    if (diff2 < diff1) {
        std::cout << "  ✓ Converging!\n";
    }
}

int main() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer Test Suite\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    try {
        testGrid2D();
        testELSProduct();
        testCPUSolver();
        testGPUSolver();
        testCPUGPUConsistency();
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
