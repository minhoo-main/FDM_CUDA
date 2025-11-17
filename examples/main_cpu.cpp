#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>
#include <iomanip>

using namespace ELSPricer;

void printHeader() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer - C++ CPU Implementation\n";
    std::cout << "   2D FDM ADI Solver (CPU Only)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
}

void runBenchmark(const ELSProduct& product) {
    std::cout << "\n┌─────────────────────────────────────┐\n";
    std::cout << "│   CPU Benchmarks                    │\n";
    std::cout << "└─────────────────────────────────────┘\n\n";

    struct GridSize {
        int N1, N2, Nt;
        std::string name;
    };

    std::vector<GridSize> sizes = {
        {50, 50, 100, "Small"},
        {100, 100, 200, "Medium"},
        {150, 150, 300, "Large"}
    };

    std::cout << std::setw(10) << "Grid"
              << std::setw(18) << "Size"
              << std::setw(15) << "Price"
              << std::setw(15) << "Time (s)"
              << std::setw(18) << "Points/sec\n";
    std::cout << std::string(76, '-') << "\n";

    for (const auto& size : sizes) {
        auto result = priceELS(product, size.N1, size.N2, size.Nt, false);

        long long totalPoints = (long long)size.N1 * size.N2 * size.Nt;
        Real pointsPerSec = totalPoints / result.computeTime;

        std::cout << std::setw(10) << size.name
                  << std::setw(5) << size.N1 << "×" << std::setw(3) << size.N2 << "×" << std::setw(4) << size.Nt
                  << std::setw(15) << std::fixed << std::setprecision(4) << result.price
                  << std::setw(15) << std::setprecision(3) << result.computeTime
                  << std::setw(18) << std::scientific << std::setprecision(2) << pointsPerSec << "\n";
    }
}

int main() {
    printHeader();

    // Create sample ELS product
    auto product = createSampleELS();
    product.printInfo();

    try {
        runBenchmark(product);

        std::cout << "\n═══════════════════════════════════════════════════════════\n";
        std::cout << "   Pricing completed successfully!\n";
        std::cout << "═══════════════════════════════════════════════════════════\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
