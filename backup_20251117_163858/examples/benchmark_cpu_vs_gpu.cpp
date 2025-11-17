#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

using namespace ELSPricer;

struct BenchmarkResult {
    int N1, N2, Nt;
    std::string gridName;
    double cpuPrice;
    double cpuTime;
    double gpuPrice;
    double gpuTime;
    double speedup;
    long long totalPoints;
};

void printHeader() {
    std::cout << "═══════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer - CPU vs GPU Comprehensive Benchmark\n";
    std::cout << "   6 Grid Configurations - Direct Performance Comparison\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════════\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n┌───────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                     CPU vs GPU Benchmark Results                          │\n";
    std::cout << "└───────────────────────────────────────────────────────────────────────────┘\n\n";

    // Header
    std::cout << std::setw(4) << "#"
              << std::setw(16) << "Grid"
              << std::setw(12) << "CPU (s)"
              << std::setw(12) << "GPU (s)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Winner"
              << std::setw(14) << "Total Pts\n";
    std::cout << std::string(82, '=') << "\n";

    // Results
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];

        std::string winner;
        if (r.speedup > 1.0) {
            winner = "GPU ✓";
        } else if (r.speedup < 1.0) {
            winner = "CPU ✓";
        } else {
            winner = "Tie";
        }

        std::cout << std::setw(4) << (i + 1)
                  << std::setw(16) << r.gridName
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.cpuTime
                  << std::setw(12) << r.gpuTime
                  << std::setw(11) << std::setprecision(2) << r.speedup << "×"
                  << std::setw(12) << winner
                  << std::setw(11) << (r.totalPoints / 1e6) << "M\n";
    }

    std::cout << std::string(82, '=') << "\n";
}

void printDetailedAnalysis(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n┌───────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                        Detailed Analysis                                  │\n";
    std::cout << "└───────────────────────────────────────────────────────────────────────────┘\n\n";

    // GPU wins count
    int gpuWins = 0;
    int cpuWins = 0;
    double totalSpeedup = 0.0;
    double maxSpeedup = 0.0;
    std::string maxSpeedupGrid;

    for (const auto& r : results) {
        if (r.speedup > 1.0) {
            gpuWins++;
        } else if (r.speedup < 1.0) {
            cpuWins++;
        }
        totalSpeedup += r.speedup;

        if (r.speedup > maxSpeedup) {
            maxSpeedup = r.speedup;
            maxSpeedupGrid = r.gridName;
        }
    }

    std::cout << "📊 Overall Statistics:\n";
    std::cout << "   GPU Wins: " << gpuWins << " / " << results.size() << "\n";
    std::cout << "   CPU Wins: " << cpuWins << " / " << results.size() << "\n";
    std::cout << "   Average GPU Speedup: " << std::fixed << std::setprecision(2)
              << (totalSpeedup / results.size()) << "×\n";
    std::cout << "   Maximum GPU Speedup: " << maxSpeedup << "× (" << maxSpeedupGrid << ")\n\n";

    // Price consistency
    std::cout << "💰 ELS Price Convergence:\n";
    double minPrice = results[0].cpuPrice;
    double maxPrice = results[0].cpuPrice;
    for (const auto& r : results) {
        minPrice = std::min(minPrice, std::min(r.cpuPrice, r.gpuPrice));
        maxPrice = std::max(maxPrice, std::max(r.cpuPrice, r.gpuPrice));
    }
    std::cout << "   Price Range: " << std::fixed << std::setprecision(4)
              << minPrice << " ~ " << maxPrice << "\n";
    std::cout << "   Price Std Dev: " << std::setprecision(6) << (maxPrice - minPrice) << "\n";
    std::cout << "   Convergence: " << (((maxPrice - minPrice) < 0.01) ? "✓ Excellent" : "⚠ Check") << "\n\n";

    // Crossover analysis
    std::cout << "🔍 Performance Crossover Analysis:\n";
    bool foundCrossover = false;
    for (const auto& r : results) {
        if (r.speedup > 1.0 && !foundCrossover) {
            std::cout << "   GPU becomes faster at: " << r.gridName << " and larger\n";
            std::cout << "   Crossover point: ~" << r.N1 << "×" << r.N2 << " grid\n";
            foundCrossover = true;
            break;
        }
    }

    if (!foundCrossover) {
        std::cout << "   CPU is faster for all tested grid sizes\n";
        std::cout << "   Note: GPU overhead dominates for small grids\n";
    }
}

void exportCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    file << "Grid_N1,Grid_N2,Grid_Nt,CPU_Price,CPU_Time,GPU_Price,GPU_Time,Speedup,Total_Points\n";

    for (const auto& r : results) {
        file << r.N1 << "," << r.N2 << "," << r.Nt << ","
             << std::fixed << std::setprecision(4) << r.cpuPrice << ","
             << std::setprecision(3) << r.cpuTime << ","
             << std::setprecision(4) << r.gpuPrice << ","
             << std::setprecision(3) << r.gpuTime << ","
             << std::setprecision(2) << r.speedup << ","
             << r.totalPoints << "\n";
    }

    file.close();
    std::cout << "\n✓ Results exported to: " << filename << "\n";
}

int main() {
    printHeader();

    auto product = createSampleELS();
    product.printInfo();

    // 6 grid configurations
    struct GridConfig {
        int N1, N2, Nt;
        std::string name;
    };

    std::vector<GridConfig> configs = {
        {100, 100, 200,  "100×100×200"},
        {100, 100, 1000, "100×100×1000"},
        {200, 200, 200,  "200×200×200"},
        {200, 200, 1000, "200×200×1000"},
        {400, 400, 200,  "400×400×200"},
        {400, 400, 1000, "400×400×1000"}
    };

    std::vector<BenchmarkResult> results;

    std::cout << "\n┌───────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    Running Benchmarks...                                  │\n";
    std::cout << "└───────────────────────────────────────────────────────────────────────────┘\n\n";

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];

        std::cout << "[" << (i + 1) << "/6] Testing " << config.name << "...\n";

        BenchmarkResult br;
        br.N1 = config.N1;
        br.N2 = config.N2;
        br.Nt = config.Nt;
        br.gridName = config.name;
        br.totalPoints = (long long)config.N1 * config.N2 * config.Nt;

        try {
            // CPU Benchmark
            std::cout << "      CPU: " << std::flush;
            auto cpuResult = priceELS(product, config.N1, config.N2, config.Nt, false);
            br.cpuPrice = cpuResult.price;
            br.cpuTime = cpuResult.computeTime;
            std::cout << std::fixed << std::setprecision(3) << br.cpuTime << "s  ";

            // GPU Benchmark
            std::cout << "GPU: " << std::flush;
            auto gpuResult = CUDA::priceELSGPU(product, config.N1, config.N2, config.Nt, false);
            br.gpuPrice = gpuResult.price;
            br.gpuTime = gpuResult.computeTime;
            std::cout << br.gpuTime << "s  ";

            // Calculate speedup
            br.speedup = br.cpuTime / br.gpuTime;

            // Winner
            if (br.speedup > 1.0) {
                std::cout << "→ GPU " << std::setprecision(2) << br.speedup << "× faster ✓\n";
            } else if (br.speedup < 1.0) {
                std::cout << "→ CPU " << std::setprecision(2) << (1.0 / br.speedup) << "× faster ⚠\n";
            } else {
                std::cout << "→ Tie\n";
            }

            results.push_back(br);

        } catch (const std::exception& e) {
            std::cout << "✗ FAILED: " << e.what() << "\n";
        }

        std::cout << "\n";
    }

    // Print results
    printResults(results);
    printDetailedAnalysis(results);

    // Export to CSV
    exportCSV(results, "cpu_vs_gpu_results.csv");

    // Python comparison
    std::cout << "\n┌───────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                Python vs C++ vs GPU Comparison                            │\n";
    std::cout << "└───────────────────────────────────────────────────────────────────────────┘\n\n";

    std::cout << std::setw(18) << "Grid"
              << std::setw(14) << "Python CPU"
              << std::setw(12) << "C++ CPU"
              << std::setw(12) << "C++ GPU"
              << std::setw(14) << "Total Gain\n";
    std::cout << std::string(70, '-') << "\n";

    // Known Python benchmarks
    struct { std::string grid; double pythonTime; } pythonData[] = {
        {"100×100×200", 6.99},
        {"200×200×1000", 78.26}
    };

    for (const auto& py : pythonData) {
        for (const auto& cpp : results) {
            if (cpp.gridName == py.grid) {
                double totalGain = py.pythonTime / cpp.gpuTime;
                std::cout << std::setw(18) << cpp.gridName
                          << std::setw(14) << std::fixed << std::setprecision(2) << py.pythonTime << "s"
                          << std::setw(12) << cpp.cpuTime << "s"
                          << std::setw(12) << cpp.gpuTime << "s"
                          << std::setw(13) << std::setprecision(0) << totalGain << "×\n";
            }
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "   Benchmark completed successfully!\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════════\n\n";

    return 0;
}
