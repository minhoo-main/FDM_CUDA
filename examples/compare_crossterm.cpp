#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "ADISolverCrossTerm.h"
#include "CUDAADISolver.cuh"
#include "CUDAADISolverCrossTerm.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

using namespace ELSPricer;

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Cross-Term Impact Analysis\n";
    std::cout << "   Comparing: Simple ADI vs ADI with Explicit Cross-Term\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    auto product = createSampleELS();

    // Display product info
    std::cout << "Product Information:\n";
    std::cout << "  Type: Step-Down ELS (Worst-of, 2 assets)\n";
    std::cout << "  Maturity: " << product.getMaturity() << " years\n";
    std::cout << "  Correlation (ρ): " << product.getRho() << "\n";
    std::cout << "  σ1 = " << product.getSigma1() << ", σ2 = " << product.getSigma2() << "\n\n";

    // Grid sizes to test
    std::vector<int> grid_sizes = {100, 200, 300, 400};
    int Nt = 1000;

    std::cout << "Testing grid sizes: ";
    for (auto N : grid_sizes) std::cout << N << "×" << N << " ";
    std::cout << "\nTime steps: " << Nt << "\n\n";

    std::cout << std::string(130, '=') << "\n";
    std::cout << std::setw(10) << "Grid"
              << std::setw(15) << "CPU Simple"
              << std::setw(15) << "CPU Cross"
              << std::setw(15) << "GPU Simple"
              << std::setw(15) << "GPU Cross"
              << std::setw(18) << "Price Diff (%)"
              << std::setw(18) << "Time Overhead"
              << std::setw(14) << "Cross Effect\n";
    std::cout << std::string(130, '=') << "\n";

    std::ofstream csv("crossterm_comparison.csv");
    csv << "Grid_Size,CPU_Simple_Time,CPU_Cross_Time,GPU_Simple_Time,GPU_Cross_Time,"
        << "CPU_Simple_Price,CPU_Cross_Price,GPU_Simple_Price,GPU_Cross_Price,"
        << "Price_Diff_Percent,CPU_Overhead_Percent,GPU_Overhead_Percent\n";

    for (int N : grid_sizes) {
        std::cout << std::setw(7) << N << "×" << N << std::flush;

        // 1. CPU Simple ADI (no cross-term)
        auto cpu_simple = priceELS(product, N, N, Nt, false);

        // 2. CPU with Cross-Term
        auto cpu_cross = priceELSCrossTerm(product, N, N, Nt, false);

        // 3. GPU Simple ADI (no cross-term)
        auto gpu_simple = CUDA::priceELSGPU(product, N, N, Nt, false);

        // 4. GPU with Cross-Term
        auto gpu_cross = CUDA::priceELSCrossTermGPU(product, N, N, Nt, false);

        // Calculate differences
        double price_diff = std::abs(cpu_cross.price - cpu_simple.price);
        double price_diff_pct = 100.0 * price_diff / cpu_simple.price;

        double cpu_overhead = 100.0 * (cpu_cross.computeTime - cpu_simple.computeTime) / cpu_simple.computeTime;
        double gpu_overhead = 100.0 * (gpu_cross.computeTime - gpu_simple.computeTime) / gpu_simple.computeTime;

        // Display results
        std::cout << std::setw(15) << std::fixed << std::setprecision(4) << cpu_simple.computeTime << "s"
                  << std::setw(14) << cpu_cross.computeTime << "s"
                  << std::setw(14) << gpu_simple.computeTime << "s"
                  << std::setw(14) << gpu_cross.computeTime << "s"
                  << std::setw(17) << std::setprecision(3) << price_diff_pct << "%"
                  << std::setw(14) << std::setprecision(1) << cpu_overhead << "%"
                  << std::setw(14) << (cpu_overhead < 15 ? "✓ Low" : "⚠ Medium") << "\n";

        // Write to CSV
        csv << N << "×" << N << ","
            << cpu_simple.computeTime << ","
            << cpu_cross.computeTime << ","
            << gpu_simple.computeTime << ","
            << gpu_cross.computeTime << ","
            << cpu_simple.price << ","
            << cpu_cross.price << ","
            << gpu_simple.price << ","
            << gpu_cross.price << ","
            << price_diff_pct << ","
            << cpu_overhead << ","
            << gpu_overhead << "\n";
    }

    csv.close();
    std::cout << std::string(130, '=') << "\n\n";

    // Detailed price comparison
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Detailed Price Comparison (400×400×1000)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    int N_detail = 400;
    auto cpu_simple = priceELS(product, N_detail, N_detail, Nt, false);
    auto cpu_cross = priceELSCrossTerm(product, N_detail, N_detail, Nt, false);
    auto gpu_simple = CUDA::priceELSGPU(product, N_detail, N_detail, Nt, false);
    auto gpu_cross = CUDA::priceELSCrossTermGPU(product, N_detail, N_detail, Nt, false);

    std::cout << std::setw(25) << "Method" << std::setw(15) << "Price" << std::setw(15) << "Time (s)" << "\n";
    std::cout << std::string(55, '-') << "\n";
    std::cout << std::setw(25) << "CPU Simple ADI" << std::setw(15) << std::setprecision(6) << cpu_simple.price
              << std::setw(15) << std::setprecision(4) << cpu_simple.computeTime << "\n";
    std::cout << std::setw(25) << "CPU with Cross-Term" << std::setw(15) << cpu_cross.price
              << std::setw(15) << cpu_cross.computeTime << "\n";
    std::cout << std::setw(25) << "GPU Simple ADI" << std::setw(15) << gpu_simple.price
              << std::setw(15) << gpu_simple.computeTime << "\n";
    std::cout << std::setw(25) << "GPU with Cross-Term" << std::setw(15) << gpu_cross.price
              << std::setw(15) << gpu_cross.computeTime << "\n";
    std::cout << std::string(55, '-') << "\n\n";

    // Analysis
    double avg_price = (cpu_simple.price + cpu_cross.price + gpu_simple.price + gpu_cross.price) / 4.0;
    double max_diff = std::max({
        std::abs(cpu_simple.price - avg_price),
        std::abs(cpu_cross.price - avg_price),
        std::abs(gpu_simple.price - avg_price),
        std::abs(gpu_cross.price - avg_price)
    });
    double max_diff_pct = 100.0 * max_diff / avg_price;

    std::cout << "Analysis:\n";
    std::cout << "  Average Price: " << std::setprecision(6) << avg_price << "\n";
    std::cout << "  Max Deviation: " << std::setprecision(4) << max_diff
              << " (" << std::setprecision(3) << max_diff_pct << "%)\n";
    std::cout << "  Price Range: [" << std::setprecision(6)
              << std::min({cpu_simple.price, cpu_cross.price, gpu_simple.price, gpu_cross.price})
              << ", "
              << std::max({cpu_simple.price, cpu_cross.price, gpu_simple.price, gpu_cross.price})
              << "]\n\n";

    double cpu_overhead = 100.0 * (cpu_cross.computeTime - cpu_simple.computeTime) / cpu_simple.computeTime;
    double gpu_overhead = 100.0 * (gpu_cross.computeTime - gpu_simple.computeTime) / gpu_simple.computeTime;

    std::cout << "Performance Impact:\n";
    std::cout << "  CPU Overhead: +" << std::setprecision(1) << cpu_overhead << "%\n";
    std::cout << "  GPU Overhead: +" << std::setprecision(1) << gpu_overhead << "%\n\n";

    // Recommendations
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Recommendations\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    std::cout << "Based on correlation ρ = " << product.getRho() << ":\n\n";

    if (max_diff_pct < 0.1) {
        std::cout << "✓ Price difference < 0.1%\n";
        std::cout << "  → Cross-term effect is NEGLIGIBLE for this correlation\n";
        std::cout << "  → Recommendation: Use simple ADI (faster, same accuracy)\n\n";
    } else if (max_diff_pct < 0.5) {
        std::cout << "⚠ Price difference " << std::setprecision(2) << max_diff_pct << "%\n";
        std::cout << "  → Cross-term has MODERATE effect\n";
        std::cout << "  → Recommendation: Consider cross-term for high-accuracy pricing\n\n";
    } else {
        std::cout << "⚠⚠ Price difference " << std::setprecision(2) << max_diff_pct << "%\n";
        std::cout << "  → Cross-term has SIGNIFICANT effect\n";
        std::cout << "  → Recommendation: MUST use cross-term correction\n\n";
    }

    std::cout << "Performance Overhead: +" << std::setprecision(1) << cpu_overhead << "% (CPU), +"
              << gpu_overhead << "% (GPU)\n";
    if (cpu_overhead < 15) {
        std::cout << "  → Overhead is ACCEPTABLE for improved accuracy\n";
    } else {
        std::cout << "  → Overhead is MODERATE but may be worthwhile\n";
    }

    std::cout << "\n✓ Results saved to: crossterm_comparison.csv\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";

    return 0;
}
