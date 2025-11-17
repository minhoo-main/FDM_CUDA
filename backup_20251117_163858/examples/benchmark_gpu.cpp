#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace ELSPricer;

void runBenchmark(const ELSProduct& product, int N1, int N2, int Nt, const std::string& label) {
    std::cout << "\n" << label << " (" << N1 << "×" << N2 << "×" << Nt << ")\n";
    std::cout << std::string(60, '-') << "\n";

    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_result = priceELS(product, N1, N2, Nt, false);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();

    // GPU timing (with warmup)
    auto gpu_result = priceELSGPU(product, N1, N2, Nt, false);  // Warmup

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_result = priceELSGPU(product, N1, N2, Nt, false);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();

    double speedup = cpu_time / gpu_time;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Price:    " << cpu_result.price << " 원\n";
    std::cout << "GPU Price:    " << gpu_result.price << " 원\n";
    std::cout << "Price diff:   " << std::abs(cpu_result.price - gpu_result.price) << " 원\n";
    std::cout << "\n";
    std::cout << "CPU Time:     " << cpu_time << " s\n";
    std::cout << "GPU Time:     " << gpu_time << " s\n";
    std::cout << "Speedup:      " << speedup << "x\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ELS Pricer - CPU vs GPU Benchmark\n";
    std::cout << "======================================================================\n";

    // Test product with realistic parameters
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

    std::cout << "\nTest Parameters:\n";
    std::cout << "  σ₁ = 15.2%, σ₂ = 40.4%, ρ = 0.61\n";
    std::cout << "  r = 3.477%, q₁ = 1.5%, q₂ = 2.0%\n";
    std::cout << "  Maturity = 3 years\n";
    std::cout << "  Barriers = [85%, 85%, 80%, 80%, 75%, 70%]\n";

    // Small grid
    runBenchmark(product, 100, 100, 200, "Small Grid");

    // Medium grid
    runBenchmark(product, 200, 200, 400, "Medium Grid");

    // Large grid
    runBenchmark(product, 500, 500, 1000, "Large Grid");

    // Very large grid (if enough memory)
    try {
        runBenchmark(product, 1000, 1000, 2000, "Very Large Grid");
    } catch (const std::exception& e) {
        std::cout << "\nVery Large Grid: Skipped (insufficient memory)\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  Benchmark Complete\n";
    std::cout << "======================================================================\n";

    return 0;
}
