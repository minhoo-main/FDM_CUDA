#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "ADISolverCrossTerm.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace ELSPricer;

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   CPU Cross-Term Implementation Test\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    auto product = createSampleELS();

    std::cout << "Product Configuration:\n";
    std::cout << "  Maturity: " << product.getMaturity() << " years\n";
    std::cout << "  Correlation (ρ): " << product.getRho() << "\n";
    std::cout << "  σ1 = " << product.getSigma1() << ", σ2 = " << product.getSigma2() << "\n";
    std::cout << "  S1_0 = " << product.getS1_0() << ", S2_0 = " << product.getS2_0() << "\n\n";

    // Test different grid sizes
    std::vector<std::pair<int, int>> test_cases = {
        {100, 200},
        {100, 500},
        {200, 1000}
    };

    std::cout << std::string(95, '=') << "\n";
    std::cout << std::setw(12) << "Grid"
              << std::setw(15) << "Simple ADI"
              << std::setw(15) << "Cross-Term"
              << std::setw(15) << "Price Diff"
              << std::setw(15) << "Diff %"
              << std::setw(15) << "Time Overhead\n";
    std::cout << std::string(95, '=') << "\n";

    for (const auto& test : test_cases) {
        int N = test.first;
        int Nt = test.second;

        std::cout << std::setw(7) << N << "×" << N
                  << " Nt=" << std::setw(4) << Nt << std::flush;

        // Test 1: Simple ADI (no cross-term)
        auto result_simple = priceELS(product, N, N, Nt, false);

        // Test 2: ADI with Cross-Term
        auto result_cross = priceELSCrossTerm(product, N, N, Nt, false);

        // Calculate differences
        float price_diff = result_cross.price - result_simple.price;
        float price_diff_pct = 100.0 * std::abs(price_diff) / result_simple.price;
        float time_overhead = 100.0 * (result_cross.computeTime - result_simple.computeTime)
                               / result_simple.computeTime;

        std::cout << std::setw(15) << std::fixed << std::setprecision(6) << result_simple.price
                  << std::setw(15) << result_cross.price
                  << std::setw(15) << std::setprecision(4) << price_diff
                  << std::setw(14) << std::setprecision(3) << price_diff_pct << "%"
                  << std::setw(13) << std::setprecision(1) << time_overhead << "%\n";
    }

    std::cout << std::string(95, '=') << "\n\n";

    // Detailed test with medium grid
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Detailed Analysis (200×200, Nt=1000)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    int N_detail = 200;
    int Nt_detail = 1000;

    std::cout << "Running Simple ADI...\n";
    auto simple = priceELS(product, N_detail, N_detail, Nt_detail, false);

    std::cout << "\nRunning Cross-Term ADI...\n";
    auto cross = priceELSCrossTerm(product, N_detail, N_detail, Nt_detail, false);

    std::cout << "\n--- Results Comparison ---\n\n";
    std::cout << std::setw(25) << "Method"
              << std::setw(18) << "Price"
              << std::setw(15) << "Time (s)"
              << std::setw(15) << "Grid Points\n";
    std::cout << std::string(73, '-') << "\n";

    long long total_points = (long long)N_detail * N_detail * Nt_detail;

    std::cout << std::setw(25) << "Simple ADI"
              << std::setw(18) << std::setprecision(8) << simple.price
              << std::setw(15) << std::setprecision(4) << simple.computeTime
              << std::setw(15) << total_points << "\n";

    std::cout << std::setw(25) << "Cross-Term ADI"
              << std::setw(18) << cross.price
              << std::setw(15) << cross.computeTime
              << std::setw(15) << total_points << "\n";

    std::cout << std::string(73, '-') << "\n\n";

    // Analysis
    float price_diff = cross.price - simple.price;
    float price_diff_pct = 100.0 * std::abs(price_diff) / simple.price;
    float time_overhead = 100.0 * (cross.computeTime - simple.computeTime) / simple.computeTime;

    std::cout << "Analysis:\n";
    std::cout << "  Price Difference: " << std::setprecision(6) << price_diff
              << " (" << std::setprecision(3) << price_diff_pct << "%)\n";
    std::cout << "  Time Overhead: +" << std::setprecision(2) << time_overhead << "%\n";
    std::cout << "  Throughput (Simple): " << std::setprecision(2)
              << (total_points / 1e6 / simple.computeTime) << " M points/sec\n";
    std::cout << "  Throughput (Cross): " << std::setprecision(2)
              << (total_points / 1e6 / cross.computeTime) << " M points/sec\n\n";

    // Interpretation
    std::cout << "Interpretation:\n";
    if (price_diff_pct < 0.1) {
        std::cout << "  ✓ Price difference < 0.1%\n";
        std::cout << "    → Cross-term effect is NEGLIGIBLE for ρ = " << product.getRho() << "\n";
        std::cout << "    → Simple ADI is sufficient for this correlation\n";
    } else if (price_diff_pct < 0.5) {
        std::cout << "  ⚠ Price difference " << std::setprecision(2) << price_diff_pct << "%\n";
        std::cout << "    → Cross-term has MODERATE effect for ρ = " << product.getRho() << "\n";
        std::cout << "    → Consider cross-term for high-accuracy requirements\n";
    } else {
        std::cout << "  ⚠⚠ Price difference " << std::setprecision(2) << price_diff_pct << "%\n";
        std::cout << "    → Cross-term has SIGNIFICANT effect for ρ = " << product.getRho() << "\n";
        std::cout << "    → Cross-term correction is REQUIRED\n";
    }

    std::cout << "\n";
    if (time_overhead < 10) {
        std::cout << "  ✓ Time overhead < 10% - Very efficient!\n";
    } else if (time_overhead < 20) {
        std::cout << "  ✓ Time overhead < 20% - Acceptable performance\n";
    } else {
        std::cout << "  ⚠ Time overhead " << std::setprecision(1) << time_overhead
                  << "% - Consider if accuracy gain justifies cost\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Test Complete\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";

    return 0;
}
