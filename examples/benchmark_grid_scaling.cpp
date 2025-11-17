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
    std::cout << "   Spatial Grid Scaling Analysis (Nt=1000 fixed)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    auto product = createSampleELS();

    // Grid sizes to test (N×N grids)
    std::vector<int> grid_sizes = {100, 200, 300, 400, 500, 600, 700};

    std::cout << std::setw(10) << "Grid Size"
              << std::setw(14) << "CPU Time(s)"
              << std::setw(14) << "GPU Time(s)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "CPU Price"
              << std::setw(14) << "GPU Price\n";
    std::cout << std::string(78, '=') << "\n";

    std::ofstream csv("grid_scaling_results.csv");
    csv << "N1,N2,Nt,Grid_Size,Total_Points,CPU_Time,GPU_Time,Speedup,CPU_Price,GPU_Price\n";

    for (int N : grid_sizes) {
        std::cout << std::setw(7) << N << "×" << N << std::flush;

        // CPU test
        auto cpu_result = priceELS(product, N, N, 1000, false);
        std::cout << std::setw(14) << std::fixed << std::setprecision(4) << cpu_result.computeTime << std::flush;

        // GPU test
        auto gpu_result = CUDA::priceELSGPU(product, N, N, 1000, false);
        std::cout << std::setw(14) << std::setprecision(4) << gpu_result.computeTime << std::flush;

        // Speedup
        float speedup = cpu_result.computeTime / gpu_result.computeTime;
        std::cout << std::setw(12) << std::setprecision(2) << speedup << "×" << std::flush;

        // Prices
        std::cout << std::setw(14) << std::setprecision(4) << cpu_result.price
                  << std::setw(14) << gpu_result.price << "\n";

        // Write to CSV
        long long total_points = (long long)N * N * 1000;
        csv << N << ","
            << N << ","
            << 1000 << ","
            << N << "×" << N << ","
            << total_points << ","
            << cpu_result.computeTime << ","
            << gpu_result.computeTime << ","
            << speedup << ","
            << cpu_result.price << ","
            << gpu_result.price << "\n";
    }

    csv.close();
    std::cout << std::string(78, '=') << "\n";
    std::cout << "\n✓ Results saved to: grid_scaling_results.csv\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";

    return 0;
}
