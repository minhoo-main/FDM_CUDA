#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.cuh"
#include <iostream>
#include <iomanip>

using namespace ELSPricer;

void printHeader() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer - C++/CUDA Implementation\n";
    std::cout << "   2D FDM ADI Solver with GPU Acceleration\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
}

void runCPUBenchmark(const ELSProduct& product) {
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
              << std::setw(15) << "Size"
              << std::setw(12) << "Price"
              << std::setw(15) << "Time (s)"
              << std::setw(15) << "Points/sec\n";
    std::cout << std::string(67, '-') << "\n";

    for (const auto& size : sizes) {
        auto result = priceELS(product, size.N1, size.N2, size.Nt, false);

        int totalPoints = size.N1 * size.N2 * size.Nt;
        Real pointsPerSec = totalPoints / result.computeTime;

        std::cout << std::setw(10) << size.name
                  << std::setw(8) << size.N1 << "×" << size.N2 << "×" << size.Nt
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.price
                  << std::setw(15) << std::setprecision(3) << result.computeTime
                  << std::setw(15) << std::scientific << std::setprecision(2) << pointsPerSec << "\n";
    }
}

void runGPUBenchmark(const ELSProduct& product) {
    std::cout << "\n┌─────────────────────────────────────┐\n";
    std::cout << "│   GPU Benchmarks                    │\n";
    std::cout << "└─────────────────────────────────────┘\n\n";

    struct GridSize {
        int N1, N2, Nt;
        std::string name;
    };

    std::vector<GridSize> sizes = {
        {50, 50, 100, "Small"},
        {100, 100, 200, "Medium"},
        {150, 150, 300, "Large"},
        {200, 200, 500, "Very Large"}
    };

    std::cout << std::setw(10) << "Grid"
              << std::setw(15) << "Size"
              << std::setw(12) << "Price"
              << std::setw(15) << "Time (s)"
              << std::setw(15) << "Points/sec\n";
    std::cout << std::string(67, '-') << "\n";

    for (const auto& size : sizes) {
        auto result = CUDA::priceELSGPU(product, size.N1, size.N2, size.Nt, false);

        int totalPoints = size.N1 * size.N2 * size.Nt;
        Real pointsPerSec = totalPoints / result.computeTime;

        std::cout << std::setw(10) << size.name
                  << std::setw(8) << size.N1 << "×" << size.N2 << "×" << size.Nt
                  << std::setw(12) << std::fixed << std::setprecision(4) << result.price
                  << std::setw(15) << std::setprecision(3) << result.computeTime
                  << std::setw(15) << std::scientific << std::setprecision(2) << pointsPerSec << "\n";
    }
}

void runComparison(const ELSProduct& product) {
    std::cout << "\n┌─────────────────────────────────────┐\n";
    std::cout << "│   CPU vs GPU Comparison             │\n";
    std::cout << "└─────────────────────────────────────┘\n\n";

    int N1 = 100, N2 = 100, Nt = 200;

    std::cout << "Grid: " << N1 << " × " << N2 << " × " << Nt << "\n\n";

    // CPU
    std::cout << "Running CPU solver...\n";
    auto cpuResult = priceELS(product, N1, N2, Nt, false);

    // GPU
    std::cout << "Running GPU solver...\n";
    auto gpuResult = CUDA::priceELSGPU(product, N1, N2, Nt, false);

    // Results
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << std::setw(20) << "Method"
              << std::setw(15) << "Price"
              << std::setw(15) << "Time (s)"
              << std::setw(15) << "Speedup\n";
    std::cout << std::string(60, '-') << "\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(20) << "CPU"
              << std::setw(15) << cpuResult.price
              << std::setw(15) << std::setprecision(3) << cpuResult.computeTime
              << std::setw(15) << "1.00×\n";

    Real speedup = cpuResult.computeTime / gpuResult.computeTime;
    std::cout << std::setw(20) << "GPU (CUDA)"
              << std::setw(15) << std::setprecision(4) << gpuResult.price
              << std::setw(15) << std::setprecision(3) << gpuResult.computeTime
              << std::setw(14) << std::setprecision(2) << speedup << "×\n";

    std::cout << std::string(60, '=') << "\n";

    // Price difference
    Real priceDiff = std::abs(cpuResult.price - gpuResult.price);
    Real relDiff = priceDiff / cpuResult.price * 100.0;
    std::cout << "\nPrice difference: " << priceDiff
              << " (" << relDiff << "%)\n";

    if (speedup > 1.0) {
        std::cout << "\n✓ GPU is " << std::setprecision(1) << speedup
                  << "× faster than CPU!\n";
    } else {
        std::cout << "\n⚠ CPU is faster for this grid size\n";
        std::cout << "  (GPU overhead dominates for small grids)\n";
    }
}

int main(int argc, char** argv) {
    printHeader();

    // Create sample ELS product
    auto product = createSampleELS();
    product.printInfo();

    // Parse command line arguments
    bool runCPU = true;
    bool runGPU = true;
    bool runCompare = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu-only") {
            runGPU = false;
        } else if (arg == "--gpu-only") {
            runCPU = false;
        } else if (arg == "--compare") {
            runCompare = true;
            runCPU = false;
            runGPU = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\nUsage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --cpu-only    Run CPU benchmarks only\n";
            std::cout << "  --gpu-only    Run GPU benchmarks only\n";
            std::cout << "  --compare     Run CPU vs GPU comparison\n";
            std::cout << "  --help, -h    Show this help message\n\n";
            return 0;
        }
    }

    try {
        if (runCompare) {
            runComparison(product);
        } else {
            if (runCPU) {
                runCPUBenchmark(product);
            }
            if (runGPU) {
                runGPUBenchmark(product);
            }
        }

        std::cout << "\n═══════════════════════════════════════════════════════════\n";
        std::cout << "   Pricing completed successfully!\n";
        std::cout << "═══════════════════════════════════════════════════════════\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
