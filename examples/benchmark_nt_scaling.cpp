#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

using namespace ELSPricer;

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Time Grid Scaling Analysis (100×100 fixed)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    auto product = createSampleELS();

    // Nt values to test (200 간격)
    std::vector<int> nt_values = {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};

    std::cout << std::setw(6) << "Nt"
              << std::setw(14) << "CPU Time(s)"
              << std::setw(14) << "GPU Time(s)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "CPU Price"
              << std::setw(14) << "GPU Price\n";
    std::cout << std::string(74, '=') << "\n";

    std::ofstream csv("nt_scaling_results.csv");
    csv << "Nt,CPU_Time,GPU_Time,Speedup,CPU_Price,GPU_Price,Total_Points\n";

    for (int nt : nt_values) {
        std::cout << std::setw(6) << nt << std::flush;

        // CPU test
        auto cpu_result = priceELS(product, 100, 100, nt, false);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << cpu_result.computeTime << std::flush;

        // GPU test
        auto gpu_result = CUDA::priceELSGPU(product, 100, 100, nt, false);
        std::cout << std::setw(14) << std::setprecision(4) << gpu_result.computeTime << std::flush;

        // Speedup
        float speedup = cpu_result.computeTime / gpu_result.computeTime;
        std::cout << std::setw(12) << std::setprecision(2) << speedup << "×" << std::flush;

        // Prices
        std::cout << std::setw(14) << std::setprecision(4) << cpu_result.price
                  << std::setw(14) << gpu_result.price << "\n";

        // Write to CSV
        csv << nt << ","
            << cpu_result.computeTime << ","
            << gpu_result.computeTime << ","
            << speedup << ","
            << cpu_result.price << ","
            << gpu_result.price << ","
            << (100 * 100 * nt) << "\n";
    }

    csv.close();
    std::cout << std::string(74, '=') << "\n";
    std::cout << "\n✓ Results saved to: nt_scaling_results.csv\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";

    return 0;
}
