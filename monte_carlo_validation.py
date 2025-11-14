#!/usr/bin/env python3
"""
Monte Carlo Simulation for ELS Pricing
- With KI tracking (정확한 가격)
- Without KI tracking (현재 PDE 코드의 가정)
"""

import numpy as np
from typing import Tuple

# 파라미터
S1_0 = 100.0
S2_0 = 100.0
sigma1 = 0.152  # 15.2%
sigma2 = 0.404  # 40.4%
rho = 0.61
r = 0.03477
q1 = 0.015
q2 = 0.02
T = 3.0

ki_barrier = 0.45  # 45%
obs_dates = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
redemption_barriers = np.array([0.85, 0.85, 0.80, 0.80, 0.75, 0.70])
coupons = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
principal = 100.0

n_paths = 200000
n_steps = 756  # 3년 × 252 영업일
dt = T / n_steps
np.random.seed(42)

print("═" * 70)
print("   Monte Carlo ELS Pricing Validation")
print("═" * 70)
print(f"\n시뮬레이션 경로 수: {n_paths:,}")
print(f"시간 스텝: {n_steps} (dt = {dt:.6f})")
print(f"\n파라미터:")
print(f"  σ1 = {sigma1*100:.1f}%, σ2 = {sigma2*100:.1f}%, ρ = {rho:.2f}")
print(f"  r = {r*100:.3f}%, q1 = {q1*100:.1f}%, q2 = {q2*100:.1f}%")
print(f"  KI Barrier = {ki_barrier*100:.0f}%")
print(f"  Redemption Barriers = {redemption_barriers*100}")
print("\n" + "─" * 70)

# Cholesky decomposition for correlation
L = np.array([[1.0, 0.0],
              [rho, np.sqrt(1 - rho**2)]])

def simulate_path() -> Tuple[float, bool, float]:
    """
    Returns: (payoff, ki_occurred, redemption_time)
    """
    S1 = S1_0
    S2 = S2_0
    t = 0.0
    ki_occurred = False
    obs_idx = 0

    for step in range(n_steps):
        t += dt

        # Correlated Brownian motion
        Z = np.random.randn(2)
        dW = L @ Z * np.sqrt(dt)

        # GBM update
        S1 *= np.exp((r - q1 - 0.5 * sigma1**2) * dt + sigma1 * dW[0])
        S2 *= np.exp((r - q2 - 0.5 * sigma2**2) * dt + sigma2 * dW[1])

        # KI check (worst-of)
        worst_perf = min(S1 / S1_0, S2 / S2_0)
        if worst_perf < ki_barrier:
            ki_occurred = True

        # Early redemption check
        if obs_idx < len(obs_dates) and t >= obs_dates[obs_idx]:
            worst_perf_now = min(S1 / S1_0, S2 / S2_0)
            if worst_perf_now >= redemption_barriers[obs_idx]:
                # Early redemption
                payoff = principal + coupons[obs_idx]
                discount = np.exp(-r * obs_dates[obs_idx])
                return payoff * discount, ki_occurred, obs_dates[obs_idx]
            obs_idx += 1

    # Maturity payoff
    final_worst = min(S1 / S1_0, S2 / S2_0)
    if ki_occurred:
        # KI occurred: principal * min(1, worst performance)
        payoff = principal * min(1.0, final_worst)
    else:
        # No KI: principal + final coupon
        payoff = principal + coupons[-1]

    discount = np.exp(-r * T)
    return payoff * discount, ki_occurred, T


# Simulate WITH KI tracking
print("\n🔬 Simulation WITH KI Tracking (정확한 가격)...")
payoffs_with_ki = []
ki_count = 0
early_redemption_count = [0] * len(obs_dates)
maturity_count = 0

for i in range(n_paths):
    if (i + 1) % 50000 == 0:
        print(f"  Progress: {i+1:,} / {n_paths:,}")

    pv, ki_occurred, redemption_time = simulate_path()
    payoffs_with_ki.append(pv)

    if ki_occurred:
        ki_count += 1

    if redemption_time < T:
        # Find which observation date
        for j, obs_t in enumerate(obs_dates):
            if abs(redemption_time - obs_t) < dt:
                early_redemption_count[j] += 1
                break
    else:
        maturity_count += 1

price_with_ki = np.mean(payoffs_with_ki)
std_with_ki = np.std(payoffs_with_ki)
se_with_ki = std_with_ki / np.sqrt(n_paths)

print(f"\n✅ WITH KI Tracking 결과:")
print(f"  가격: {price_with_ki:.4f}원 ± {1.96*se_with_ki:.4f} (95% CI)")
print(f"  낙인 발생 확률: {ki_count/n_paths*100:.2f}%")
print(f"  만기까지 도달: {maturity_count/n_paths*100:.2f}%")
print(f"\n  조기상환 비율:")
for i, (obs_t, count) in enumerate(zip(obs_dates, early_redemption_count)):
    print(f"    {obs_t:.1f}년: {count/n_paths*100:5.2f}%")

print("\n" + "─" * 70)

# Simulate WITHOUT KI tracking (assume ki_occurred = False always)
print("\n🔬 Simulation WITHOUT KI Tracking (현재 PDE 가정)...")
payoffs_no_ki = []

np.random.seed(42)  # Same random seed for fair comparison

for i in range(n_paths):
    if (i + 1) % 50000 == 0:
        print(f"  Progress: {i+1:,} / {n_paths:,}")

    S1 = S1_0
    S2 = S2_0
    t = 0.0
    obs_idx = 0
    redeemed = False
    payoff = 0.0

    for step in range(n_steps):
        t += dt

        Z = np.random.randn(2)
        dW = L @ Z * np.sqrt(dt)

        S1 *= np.exp((r - q1 - 0.5 * sigma1**2) * dt + sigma1 * dW[0])
        S2 *= np.exp((r - q2 - 0.5 * sigma2**2) * dt + sigma2 * dW[1])

        # Early redemption check (no KI tracking!)
        if obs_idx < len(obs_dates) and t >= obs_dates[obs_idx]:
            worst_perf_now = min(S1 / S1_0, S2 / S2_0)
            if worst_perf_now >= redemption_barriers[obs_idx]:
                payoff = (principal + coupons[obs_idx]) * np.exp(-r * obs_dates[obs_idx])
                redeemed = True
                break
            obs_idx += 1

    if not redeemed:
        # Maturity: assume NO KI occurred
        payoff = (principal + coupons[-1]) * np.exp(-r * T)

    payoffs_no_ki.append(payoff)

price_no_ki = np.mean(payoffs_no_ki)
std_no_ki = np.std(payoffs_no_ki)
se_no_ki = std_no_ki / np.sqrt(n_paths)

print(f"\n✅ WITHOUT KI Tracking 결과:")
print(f"  가격: {price_no_ki:.4f}원 ± {1.96*se_no_ki:.4f} (95% CI)")

print("\n" + "═" * 70)
print("   결과 비교")
print("═" * 70)
print(f"\n{'항목':<30} {'WITH KI':<15} {'WITHOUT KI':<15}")
print("─" * 60)
print(f"{'가격':<30} {price_with_ki:>10.4f}원 {price_no_ki:>14.4f}원")
print(f"{'표준편차':<30} {std_with_ki:>10.4f} {std_no_ki:>18.4f}")
print(f"{'95% 신뢰구간 폭':<30} {1.96*se_with_ki:>10.4f} {1.96*se_no_ki:>18.4f}")
print("─" * 60)
print(f"{'가격 차이':<30} {abs(price_with_ki - price_no_ki):>10.4f}원 ({abs(price_with_ki - price_no_ki)/price_with_ki*100:>5.2f}%)")
print(f"{'낙인 발생 확률':<30} {ki_count/n_paths*100:>9.2f}%")

print("\n" + "═" * 70)
print("   PDE vs Monte Carlo 비교")
print("═" * 70)
print(f"\n{'방법':<30} {'가격':<15} {'비고':<30}")
print("─" * 75)
print(f"{'PDE (현재 구현)':<30} {'111.74원':<15} {'KI 추적 없음 (버그)':<30}")
print(f"{'MC WITHOUT KI':<30} {f'{price_no_ki:.2f}원':<15} {'PDE와 비교 기준':<30}")
print(f"{'MC WITH KI (정확)':<30} {f'{price_with_ki:.2f}원':<15} {'올바른 가격':<30}")

print("\n💡 결론:")
if price_with_ki < 100:
    print(f"  ✅ KI 추적 시 가격 {price_with_ki:.2f}원 < 100원 (정상)")
    print(f"  ❌ 현재 PDE 111.74원 > 100원 (KI 버그)")
else:
    print(f"  ⚠️  KI 추적해도 가격 > 100원 (조기상환 확률이 매우 높음)")

print(f"  📊 낙인 영향: {price_no_ki - price_with_ki:.2f}원 ({(price_no_ki - price_with_ki)/price_no_ki*100:.1f}% 하락)")
print("═" * 70 + "\n")
