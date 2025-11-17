#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

using namespace ELSPricer;

struct BenchmarkResult {
    int N1, N2, Nt;
    double price;
    double time;
    long long totalPoints;
    double pointsPerSec;
};

void printHeader() {
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Pricer - Comprehensive Benchmark\n";
    std::cout << "   CPU Performance Test - 6 Grid Configurations\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    Benchmark Results                                │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n\n";

    std::cout << std::setw(5) << "#"
              << std::setw(20) << "Grid Size"
              << std::setw(15) << "Price"
              << std::setw(12) << "Time (s)"
              << std::setw(15) << "Points/sec"
              << std::setw(12) << "Total Pts\n";
    std::cout << std::string(79, '=') << "\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << std::setw(5) << (i + 1)
                  << std::setw(7) << r.N1 << "×" << std::setw(3) << r.N2 << "×" << std::setw(4) << r.Nt
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.price
                  << std::setw(12) << std::setprecision(3) << r.time
                  << std::setw(15) << std::scientific << std::setprecision(2) << r.pointsPerSec
                  << std::setw(12) << std::fixed << std::setprecision(0) << (r.totalPoints / 1e6) << "M\n";
    }

    std::cout << std::string(79, '=') << "\n";
}

void exportCSV(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    // Header
    file << "Grid_N1,Grid_N2,Grid_Nt,Price,Time_sec,Points_per_sec,Total_Points\n";

    // Data
    for (const auto& r : results) {
        file << r.N1 << "," << r.N2 << "," << r.Nt << ","
             << std::fixed << std::setprecision(4) << r.price << ","
             << std::setprecision(3) << r.time << ","
             << std::scientific << std::setprecision(6) << r.pointsPerSec << ","
             << r.totalPoints << "\n";
    }

    file.close();
    std::cout << "\n✓ Results exported to: " << filename << "\n";
}

int main() {
    printHeader();

    // Create sample ELS product
    auto product = createSampleELS();
    product.printInfo();

    // Define 6 grid configurations
    struct GridConfig {
        int N1, N2, Nt;
        std::string description;
    };

    std::vector<GridConfig> configs = {
        {100, 100, 200,  "Small - Medium density"},
        {100, 100, 1000, "Small - High time resolution"},
        {200, 200, 200,  "Large - Medium density"},
        {200, 200, 1000, "Large - High time resolution"},
        {400, 400, 200,  "Very Large - Medium density"},
        {400, 400, 1000, "Very Large - High time resolution"}
    };

    std::vector<BenchmarkResult> results;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    Running Benchmarks...                            │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n\n";

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];

        std::cout << "[" << (i + 1) << "/6] Testing " << config.N1 << "×" << config.N2
                  << "×" << config.Nt << " (" << config.description << ")...\n";
        std::cout << "      ";
        std::cout.flush();

        try {
            auto result = priceELS(product, config.N1, config.N2, config.Nt, false);

            BenchmarkResult br;
            br.N1 = config.N1;
            br.N2 = config.N2;
            br.Nt = config.Nt;
            br.price = result.price;
            br.time = result.computeTime;
            br.totalPoints = (long long)config.N1 * config.N2 * config.Nt;
            br.pointsPerSec = br.totalPoints / result.computeTime;

            results.push_back(br);

            std::cout << "✓ Completed in " << std::fixed << std::setprecision(3)
                      << result.computeTime << "s (Price: " << std::setprecision(4)
                      << result.price << ")\n";

        } catch (const std::exception& e) {
            std::cout << "✗ FAILED: " << e.what() << "\n";
        }

        std::cout << "\n";
    }

    // Print summary
    printResults(results);

    // Export to CSV
    exportCSV(results, "benchmark_results_cpu.csv");

    // Analysis
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    Performance Analysis                             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n\n";

    if (results.size() >= 2) {
        std::cout << "Grid size scaling:\n";
        std::cout << "  100×100 → 200×200 (4× points): "
                  << std::fixed << std::setprecision(2)
                  << (results[2].time / results[0].time) << "× time increase\n";

        if (results.size() >= 4) {
            std::cout << "  200×200 → 400×400 (4× points): "
                      << (results[4].time / results[2].time) << "× time increase\n";
        }

        std::cout << "\nTime resolution scaling:\n";
        std::cout << "  100×100: 200 → 1000 steps (5× steps): "
                  << (results[1].time / results[0].time) << "× time increase\n";

        if (results.size() >= 4) {
            std::cout << "  200×200: 200 → 1000 steps (5× steps): "
                      << (results[3].time / results[2].time) << "× time increase\n";
        }
    }

    // Python comparison (based on original benchmark)
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                Python vs C++ Comparison                             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "Based on original Python benchmarks:\n\n";
    std::cout << std::setw(20) << "Grid"
              << std::setw(15) << "Python CPU"
              << std::setw(15) << "C++ CPU"
              << std::setw(15) << "Speedup\n";
    std::cout << std::string(65, '-') << "\n";

    // Known Python timings
    struct PythonBenchmark {
        int N1, N2, Nt;
        double pythonTime;
    };

    std::vector<PythonBenchmark> pythonData = {
        {100, 100, 200, 6.99},     // From original benchmark
        {200, 200, 1000, 78.26}    // From original benchmark
    };

    for (const auto& py : pythonData) {
        // Find matching C++ result
        for (const auto& cpp : results) {
            if (cpp.N1 == py.N1 && cpp.N2 == py.N2 && cpp.Nt == py.Nt) {
                double speedup = py.pythonTime / cpp.time;
                std::cout << std::setw(7) << cpp.N1 << "×" << cpp.N2 << "×" << cpp.Nt
                          << std::setw(15) << std::fixed << std::setprecision(2) << py.pythonTime << "s"
                          << std::setw(15) << cpp.time << "s"
                          << std::setw(14) << std::setprecision(1) << speedup << "×\n";
            }
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "   Benchmark completed successfully!\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "💡 Next Steps:\n";
    std::cout << "   - Run GPU version for comparison\n";
    std::cout << "   - Upload benchmark_results_cpu.csv for analysis\n";
    std::cout << "   - Expected GPU speedup: 10-50× over CPU\n\n";

    return 0;
}
