#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

/**
 * 3-Asset FDM Feasibility Analysis
 *
 * This program analyzes the computational and memory requirements
 * for 3-asset FDM pricing and compares with alternatives.
 */

// Memory and computation analysis
struct GridAnalysis {
    int N1, N2, N3, Nt;

    size_t getTotalPoints() const {
        return (size_t)N1 * N2 * N3 * Nt;
    }

    double getMemoryGB() const {
        // Need at least 3 arrays for ADI
        size_t arrays_needed = 3;
        size_t bytes = getTotalPoints() * arrays_needed * sizeof(double);
        return bytes / (1024.0 * 1024.0 * 1024.0);
    }

    double getComputationTime(double ops_per_point = 100,
                              double gflops_speed = 100) const {
        // Rough estimate: 100 operations per point
        double total_ops = getTotalPoints() * ops_per_point;
        return total_ops / (gflops_speed * 1e9);
    }
};

// Monte Carlo comparison
double monteCarloTime(int num_paths, int num_steps,
                      double ops_per_step = 50,
                      double gflops_speed = 100) {
    double total_ops = (double)num_paths * num_steps * ops_per_step;
    return total_ops / (gflops_speed * 1e9);
}

int main() {
    std::cout << "===============================================\n";
    std::cout << "    3-Asset FDM Feasibility Analysis\n";
    std::cout << "===============================================\n\n";

    // Test different grid sizes
    std::vector<GridAnalysis> grids = {
        {30, 30, 30, 100},    // Very small
        {50, 50, 50, 200},    // Small
        {75, 75, 75, 300},    // Medium
        {100, 100, 100, 500}, // Large
        {150, 150, 150, 1000} // Very large
    };

    std::cout << "### 3-Asset FDM Requirements:\n";
    std::cout << std::setw(20) << "Grid Size"
              << std::setw(15) << "Total Points"
              << std::setw(12) << "Memory (GB)"
              << std::setw(15) << "CPU Time (s)"
              << std::setw(12) << "Feasible?\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& grid : grids) {
        std::string grid_str = std::to_string(grid.N1) + "×" +
                              std::to_string(grid.N2) + "×" +
                              std::to_string(grid.N3) + "×" +
                              std::to_string(grid.Nt);

        double memory = grid.getMemoryGB();
        double cpu_time = grid.getComputationTime();
        std::string feasible = (memory < 16 && cpu_time < 60) ? "✓ Yes" : "✗ No";

        if (memory > 32) feasible = "✗ GPU OOM";

        std::cout << std::setw(20) << grid_str
                  << std::setw(15) << std::scientific << std::setprecision(2)
                  << (double)grid.getTotalPoints()
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << memory
                  << std::setw(15) << std::setprecision(1) << cpu_time
                  << std::setw(12) << feasible << "\n";
    }

    std::cout << "\n### Comparison with 2-Asset FDM:\n";
    std::cout << std::setw(20) << "Type"
              << std::setw(20) << "Grid/Paths"
              << std::setw(15) << "Memory (GB)"
              << std::setw(15) << "Time (s)\n";
    std::cout << std::string(70, '-') << "\n";

    // 2-Asset FDM
    size_t points_2d = 200 * 200 * 1000;
    double mem_2d = points_2d * 3 * 8.0 / 1e9;
    double time_2d = points_2d * 100 / (100 * 1e9);

    std::cout << std::setw(20) << "2-Asset FDM"
              << std::setw(20) << "200×200×1000"
              << std::setw(15) << std::fixed << std::setprecision(2) << mem_2d
              << std::setw(15) << time_2d << "\n";

    // 3-Asset FDM
    size_t points_3d = 100 * 100 * 100 * 500;
    double mem_3d = points_3d * 3 * 8.0 / 1e9;
    double time_3d = points_3d * 100 / (100 * 1e9);

    std::cout << std::setw(20) << "3-Asset FDM"
              << std::setw(20) << "100×100×100×500"
              << std::setw(15) << mem_3d
              << std::setw(15) << time_3d << "\n";

    // Monte Carlo
    int mc_paths = 1000000;
    int mc_steps = 1000;
    double mc_time = monteCarloTime(mc_paths, mc_steps);
    double mc_mem = mc_paths * mc_steps * 8.0 / 1e9;

    std::cout << std::setw(20) << "3-Asset Monte Carlo"
              << std::setw(20) << "1M paths × 1K steps"
              << std::setw(15) << mc_mem
              << std::setw(15) << mc_time << "\n";

    std::cout << "\n### Recommendations:\n";
    std::cout << "===============================================\n";
    std::cout << "For 3-Asset Pricing:\n\n";

    std::cout << "1. **Small grids (≤50×50×50)**\n";
    std::cout << "   → FDM feasible, good accuracy\n";
    std::cout << "   → Memory: ~2-5 GB\n";
    std::cout << "   → Time: <10 seconds\n\n";

    std::cout << "2. **Medium grids (75×75×75)**\n";
    std::cout << "   → FDM challenging, use GPU\n";
    std::cout << "   → Memory: ~10-15 GB\n";
    std::cout << "   → Consider sparse grids\n\n";

    std::cout << "3. **Large grids (≥100×100×100)**\n";
    std::cout << "   → Use Monte Carlo instead\n";
    std::cout << "   → FDM memory > 30 GB\n";
    std::cout << "   → MC more efficient\n\n";

    std::cout << "4. **Production Recommendations:**\n";
    std::cout << "   → 2 assets: FDM (200×200×1000)\n";
    std::cout << "   → 3 assets: Monte Carlo (10M paths)\n";
    std::cout << "   → 4+ assets: Always Monte Carlo\n\n";

    std::cout << "5. **Hybrid Approach:**\n";
    std::cout << "   → Important region: FDM (30×30×30)\n";
    std::cout << "   → Boundary: Monte Carlo\n";
    std::cout << "   → Best of both worlds\n\n";

    std::cout << "===============================================\n";
    std::cout << "### Memory Limits:\n";
    std::cout << "- CPU RAM: Usually 16-64 GB\n";
    std::cout << "- GPU (T4): 16 GB\n";
    std::cout << "- GPU (A100): 40-80 GB\n";
    std::cout << "- Colab Free: ~13 GB\n\n";

    std::cout << "### When to use 3-Asset FDM:\n";
    std::cout << "✓ High accuracy needed near barriers\n";
    std::cout << "✓ Small number of scenarios\n";
    std::cout << "✓ Greeks calculation important\n";
    std::cout << "✓ Have sufficient GPU memory\n\n";

    std::cout << "### When to avoid 3-Asset FDM:\n";
    std::cout << "✗ Grid > 100×100×100\n";
    std::cout << "✗ Memory limited environment\n";
    std::cout << "✗ Need fast results\n";
    std::cout << "✗ 4+ assets\n\n";

    return 0;
}