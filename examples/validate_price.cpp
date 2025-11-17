#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include "CUDAADISolver.cuh"
#include <iostream>
#include <iomanip>

using namespace ELSPricer;

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Price Validation Test\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // 당신의 파라미터
    Real principal = 100.0;
    Real maturity = 3.0;  // 3년

    // Observation dates: 6개월 간격
    std::vector<Real> observationDates = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    // Redemption barriers: 85%, 85%, 80%, 80%, 75%, 70%
    std::vector<Real> redemptionBarriers = {0.85, 0.85, 0.80, 0.80, 0.75, 0.70};

    // Coupons: 6개월마다 지급되는 쿠폰 (연 8% 가정)
    // 0.5년: 4%, 1.0년: 8%, 1.5년: 12%, 2.0년: 16%, 2.5년: 20%, 3.0년: 24%
    std::vector<Real> coupons = {4.0, 8.0, 12.0, 16.0, 20.0, 24.0};

    Real kiBarrier = 0.45;  // 낙인 배리어 45%

    Real S1_0 = 100.0;  // 기초자산 1 초기가
    Real S2_0 = 100.0;  // 기초자산 2 초기가

    Real sigma1 = 0.152;  // 15.2% 변동성
    Real sigma2 = 0.404;  // 40.4% 변동성
    Real rho = 0.61;      // 상관계수 0.61

    Real r = 0.03477;     // 무위험 이자율 3.477%
    Real q1 = 0.015;      // 배당률 1.5%
    Real q2 = 0.02;       // 배당률 2%

    // ELS 상품 생성
    ELSProduct product(
        principal, maturity,
        observationDates,
        redemptionBarriers,
        coupons,
        kiBarrier,
        S1_0, S2_0,
        sigma1, sigma2, rho,
        r, q1, q2,
        true  // worst-of
    );

    // 상품 정보 출력
    product.printInfo();

    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   Pricing with Different Grid Sizes\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // 다양한 그리드 크기로 테스트
    std::vector<std::tuple<int, int, int>> grids = {
        {100, 100, 200},
        {100, 100, 500},
        {200, 200, 500},
        {200, 200, 1000}
    };

    std::cout << std::setw(8) << "Grid"
              << std::setw(12) << "CPU Price"
              << std::setw(12) << "GPU Price"
              << std::setw(12) << "CPU Time"
              << std::setw(12) << "GPU Time\n";
    std::cout << std::string(56, '=') << "\n";

    for (const auto& [N1, N2, Nt] : grids) {
        // CPU 가격
        auto cpu_result = priceELS(product, N1, N2, Nt, false);

        // GPU 가격
        auto gpu_result = CUDA::priceELSGPU(product, N1, N2, Nt, false);

        std::string gridLabel = std::to_string(N1) + "×" + std::to_string(N2) + "×" + std::to_string(Nt);
        std::cout << std::setw(8) << gridLabel
                  << std::setw(12) << std::fixed << std::setprecision(4) << cpu_result.price
                  << std::setw(12) << gpu_result.price
                  << std::setw(11) << std::setprecision(3) << cpu_result.computeTime << "s"
                  << std::setw(11) << gpu_result.computeTime << "s\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n✓ Validation Complete!\n";
    std::cout << "  → Check price convergence across different grid sizes\n";
    std::cout << "  → CPU and GPU prices should match within 0.01%\n";

    return 0;
}
