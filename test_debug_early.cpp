#include "Grid2D.h"
#include "ELSProduct.h"
#include "ADISolver.h"
#include <iostream>

using namespace ELSPricer;

int main() {
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

    std::cout << "Testing with 100×100×200 grid (verbose mode):\n\n";

    // Call with verbose=true to trigger debug output
    auto result = priceELS(product, 100, 100, 200, true);

    std::cout << "\n\nFinal Price: " << result.price << "\n";

    return 0;
}
