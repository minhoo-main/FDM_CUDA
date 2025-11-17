#ifndef ELS_PRICER_PRECISION_H
#define ELS_PRICER_PRECISION_H

/**
 * Precision Configuration
 *
 * Change this one line to switch between FP32 and FP64:
 * - FP64 (double): Higher precision, slower (0.25 TFLOPS on T4)
 * - FP32 (float):  Lower precision, 8x faster (8.1 TFLOPS on T4)
 *
 * For ELS pricing, FP32 is typically sufficient (< 0.0001% error)
 */

namespace ELSPricer {

// === CHANGE THIS LINE TO SWITCH PRECISION ===
using Real = double;  // Current: FP64 (high precision)
// using Real = float;   // Alternative: FP32 (8x faster)
// ============================================

} // namespace ELSPricer

#endif // ELS_PRICER_PRECISION_H
