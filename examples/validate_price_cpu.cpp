#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>
#include <iomanip>

using namespace ELSPricer;

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "   ELS Price Validation Test (CPU Only)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // 당신의 파라미터
    float principal = 100.0;
    float maturity = 3.0;  // 3년

    // Observation dates: 6개월 간격
    std::vector<float> observationDates = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    // Redemption barriers: 85%, 85%, 80%, 80%, 75%, 70%
    std::vector<float> redemptionBarriers = {0.85, 0.85, 0.80, 0.80, 0.75, 0.70};

    // Coupons: 6개월마다 지급되는 쿠폰 (연 8% 가정)
    // 0.5년: 4%, 1.0년: 8%, 1.5년: 12%, 2.0년: 16%, 2.5년: 20%, 3.0년: 24%
    std::vector<float> coupons = {4.0, 8.0, 12.0, 16.0, 20.0, 24.0};

    float kiBarrier = 0.45;  // 낙인 배리어 45%

    float S1_0 = 100.0;  // 기초자산 1 초기가
    float S2_0 = 100.0;  // 기초자산 2 초기가

    float sigma1 = 0.152;  // 15.2% 변동성
    float sigma2 = 0.404;  // 40.4% 변동성
    float rho = 0.61;      // 상관계수 0.61

    float r = 0.03477;     // 무위험 이자율 3.477%
    float q1 = 0.015;      // 배당률 1.5%
    float q2 = 0.02;       // 배당률 2%

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
    std::cout << "   Pricing with Different Grid Sizes (CPU Only)\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    // 다양한 그리드 크기로 테스트
    std::vector<std::tuple<int, int, int>> grids = {
        {100, 100, 200},
        {100, 100, 500},
        {200, 200, 500}
    };

    std::cout << std::setw(12) << "Grid"
              << std::setw(14) << "Price"
              << std::setw(14) << "Time\n";
    std::cout << std::string(40, '=') << "\n";

    for (const auto& [N1, N2, Nt] : grids) {
        // CPU 가격
        auto result = priceELS(product, N1, N2, Nt, false);

        std::string gridLabel = std::to_string(N1) + "×" + std::to_string(N2) + "×" + std::to_string(Nt);
        std::cout << std::setw(12) << gridLabel
                  << std::setw(14) << std::fixed << std::setprecision(4) << result.price
                  << std::setw(13) << std::setprecision(3) << result.computeTime << "s\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n✓ Validation Complete!\n";
    std::cout << "  → Check price convergence across different grid sizes\n";
    std::cout << "  → Price should stabilize as grid becomes finer\n\n";

    return 0;
}
