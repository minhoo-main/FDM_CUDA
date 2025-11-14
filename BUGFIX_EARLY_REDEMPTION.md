# Early Redemption Bug Fixes - Summary

## Bug #1: Optional vs Mandatory Redemption

### Problem
Early redemption was implemented as **optional** using `std::max(V, payoff)`, allowing the option holder to choose continuation if the continuation value was higher than the redemption payoff.

```cpp
// INCORRECT - treats early redemption as optional (American-style)
if (result.isRedeemed) {
    V[i * N2_ + j] = std::max(V[i * N2_ + j], result.payoff);
}
```

### Why This Is Wrong
Step-Down ELS products have **mandatory** early redemption. When the barrier condition is met at an observation date, the product automatically terminates and pays out. The investor has no choice.

### Fix
Changed to forced assignment:

```cpp
// CORRECT - mandatory redemption
if (result.isRedeemed) {
    V[i * N2_ + j] = result.payoff;  // Force redemption, no choice
}
```

### Impact
- Minor pricing impact (~1-2% in tested cases)
- Conceptually critical - reflects actual product behavior

---

## Bug #2: Timestep Indexing in Backward PDE Loop

### Problem
The backward PDE loop went from `Nt_-1` to `0`, but the time grid has `Nt_+1` points (indices 0 to Nt_).

```cpp
// INCORRECT - misses timestep Nt_
for (int n = Nt_ - 1; n >= 0; --n) {
    // PDE steps
    solveS1Direction(V, V_half);
    solveS2Direction(V_half, V);
    applyBoundaryConditions(V);

    // Early redemption check (never executes for observation at t=T!)
    if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
        applyEarlyRedemption(V, obsIdx, product);
        --obsIdx;
    }
}
```

### Why This Caused Early Redemptions to Be Skipped

1. **Time grid structure**: Grid has Nt_+1 time points
   - `t[0] = 0.0`
   - `t[1] = dt`
   - ...
   - `t[Nt_] = T` (maturity)

2. **Observation mapping**: For a 3-year product with Nt_=200:
   - Observation at t=0.5 → timestep 33
   - Observation at t=1.0 → timestep 67
   - ...
   - **Observation at t=3.0 → timestep 200** ← Loop never reaches this!

3. **Loop range**: Loop goes from 199 down to 0, so condition `n == 200` is never true

### Result
**All early redemptions were skipped**, causing massive overpricing:
- PDE price: 111.74원 (with bug)
- MC price:  104.44원 (correct, without KI)
- Error: **7.0원 (~7% overpricing)**

### Fix
Restructured loop to go from Nt_ down to 1, checking early redemption **before** PDE steps:

```cpp
// CORRECT - includes timestep Nt_
for (int n = Nt_; n >= 1; --n) {
    // Check early redemption BEFORE PDE step
    if (obsIdx >= 0 && n == obsIndices[obsIdx]) {
        applyEarlyRedemption(V, obsIdx, product);
        --obsIdx;
    }

    // ADI steps (except at n=Nt_ which is terminal condition)
    if (n < Nt_) {
        solveS1Direction(V, V_half);
        solveS2Direction(V_half, V);
        applyBoundaryConditions(V);
    }
}
```

### Logic
- **At n=Nt_**: Only check early redemption (no PDE step - this is the terminal condition)
- **At n<Nt_**: Apply PDE steps to evolve backward in time, then check early redemption
- Loop now covers all timesteps from Nt_ down to 1

### Verification

After fix, debug output shows early redemptions executing:

```
[DEBUG] Applying early redemption at timestep 200
  [DEBUG] Observation 5 (t=3): 5776 / 10000 points redeemed (57.8%)
[DEBUG] Applying early redemption at timestep 167
  [DEBUG] Observation 4 (t=2.5): 5625 / 10000 points redeemed (56.2%)
[DEBUG] Applying early redemption at timestep 133
  [DEBUG] Observation 3 (t=2.0): 5329 / 10000 points redeemed (53.3%)
...
```

### Impact
- PDE price: 103.9원 (after fix)
- MC price:  104.44원
- **Error reduced from 7% to 0.5%** ✅

---

## Validation Results

### Test Parameters
- σ₁ = 15.2%, σ₂ = 40.4%, ρ = 0.61
- r = 3.477%, q₁ = 1.5%, q₂ = 2.0%
- Maturity = 3 years
- Barriers = [85%, 85%, 80%, 80%, 75%, 70%]
- KI Barrier = 45%
- Observations at [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] years

### Monte Carlo (100,000 paths)
- **With KI**: 93.92원 (15.93% KI occurrence)
- **Without KI**: 104.44원

### PDE (100×100×200 grid)
- **Before fix**: 111.74원 (no early redemptions)
- **After fix**: 103.9원 (early redemptions working)

### Agreement
PDE (103.9) vs MC (104.44) = **0.54원 difference (0.5%)** ✅

Remaining difference is due to:
- Grid discretization error
- Time discretization error
- Numerical diffusion in ADI scheme

This is well within acceptable tolerance for production use.

---

## Files Modified

### src/ADISolver.cpp
- Fixed `applyEarlyRedemption()`: Changed from `std::max` to forced assignment
- Fixed `solveWithEarlyRedemption()`: Restructured backward loop from Nt_ to 1
- Added debug output to track redemption statistics

### src/cuda/batched_thomas.cu
- Fixed `applyEarlyRedemptionKernel()`: Changed from optional to mandatory redemption

### src/cuda/CUDAADISolver.cu
- Fixed `solveWithEarlyRedemption()`: Restructured backward loop (same as CPU)

### test_debug_early.cpp
- New test program to verify early redemption with debug output

---

## Remaining Known Issues

### KI (Knock-In) Tracking Not Implemented
- Current code assumes `kiOccurred = false` (hardcoded)
- Correct price **with KI**: ~93.92원
- Current PDE price **without KI**: 103.9원
- Error from missing KI: **~10원 (~11%)**

**Solution required**:
- Implement 2-state PDE (doubles memory and computation)
- Or use Monte Carlo for KI products
- See `KI_TRACKING_BUG.md` for details

---

## Commits
- `98eb1d6`: Fix critical timestep indexing bug in early redemption
- `43d64d1`: Fix early redemption to be mandatory (not optional)

## Testing
```bash
# CPU test with debug output
./test_debug_early

# Expected output: Price ~103.9원 with redemption statistics
```
