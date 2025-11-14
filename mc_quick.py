#!/usr/bin/env python3
"""Quick Monte Carlo for ELS - WITH and WITHOUT KI"""
import numpy as np

# Parameters
S1_0, S2_0 = 100.0, 100.0
sigma1, sigma2 = 0.152, 0.404
rho = 0.61
r, q1, q2 = 0.03477, 0.015, 0.02
T = 3.0
ki_barrier = 0.45
obs_dates = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
barriers = np.array([0.85, 0.85, 0.80, 0.80, 0.75, 0.70])
coupons = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
principal = 100.0

n_paths = 50000  # Reduced for speed
n_steps = 252 * 3  # Daily steps
dt = T / n_steps
np.random.seed(42)

print("="*70)
print("  Quick Monte Carlo Validation (50,000 paths)")
print("="*70)

# Cholesky
L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])

# WITH KI tracking
print("\n[1] WITH KI Tracking (정확한 가격)...")
payoffs_ki = []
ki_count = 0

for path in range(n_paths):
    S1, S2 = S1_0, S2_0
    ki_hit = False

    for i, obs_t in enumerate(obs_dates):
        # Simulate until this observation date
        steps_to_obs = int(obs_t / dt)
        prev_steps = int(obs_dates[i-1] / dt) if i > 0 else 0

        for _ in range(prev_steps, steps_to_obs):
            Z = np.random.randn(2)
            dW = L @ Z * np.sqrt(dt)
            S1 *= np.exp((r - q1 - 0.5*sigma1**2)*dt + sigma1*dW[0])
            S2 *= np.exp((r - q2 - 0.5*sigma2**2)*dt + sigma2*dW[1])

            # Check KI
            worst = min(S1/S1_0, S2/S2_0)
            if worst < ki_barrier:
                ki_hit = True

        # Check early redemption
        worst_now = min(S1/S1_0, S2/S2_0)
        if worst_now >= barriers[i]:
            pv = (principal + coupons[i]) * np.exp(-r * obs_t)
            payoffs_ki.append(pv)
            if ki_hit:
                ki_count += 1
            break
    else:
        # Maturity
        final_worst = min(S1/S1_0, S2/S2_0)
        if ki_hit:
            payoff = principal * min(1.0, final_worst)
        else:
            payoff = principal + coupons[-1]
        pv = payoff * np.exp(-r * T)
        payoffs_ki.append(pv)
        if ki_hit:
            ki_count += 1

price_ki = np.mean(payoffs_ki)
std_ki = np.std(payoffs_ki)

print(f"  가격: {price_ki:.4f}원")
print(f"  표준편차: {std_ki:.4f}")
print(f"  낙인 발생 확률: {ki_count/n_paths*100:.2f}%")

# WITHOUT KI tracking
print("\n[2] WITHOUT KI Tracking (현재 PDE 가정)...")
np.random.seed(42)
payoffs_no_ki = []

for path in range(n_paths):
    S1, S2 = S1_0, S2_0

    for i, obs_t in enumerate(obs_dates):
        steps_to_obs = int(obs_t / dt)
        prev_steps = int(obs_dates[i-1] / dt) if i > 0 else 0

        for _ in range(prev_steps, steps_to_obs):
            Z = np.random.randn(2)
            dW = L @ Z * np.sqrt(dt)
            S1 *= np.exp((r - q1 - 0.5*sigma1**2)*dt + sigma1*dW[0])
            S2 *= np.exp((r - q2 - 0.5*sigma2**2)*dt + sigma2*dW[1])

        worst_now = min(S1/S1_0, S2/S2_0)
        if worst_now >= barriers[i]:
            pv = (principal + coupons[i]) * np.exp(-r * obs_t)
            payoffs_no_ki.append(pv)
            break
    else:
        # Maturity: assume NO KI
        payoff = principal + coupons[-1]
        pv = payoff * np.exp(-r * T)
        payoffs_no_ki.append(pv)

price_no_ki = np.mean(payoffs_no_ki)
std_no_ki = np.std(payoffs_no_ki)

print(f"  가격: {price_no_ki:.4f}원")
print(f"  표준편차: {std_no_ki:.4f}")

print("\n" + "="*70)
print("  결과 비교")
print("="*70)
print(f"\n{'항목':<25} {'WITH KI':<15} {'WITHOUT KI':<15}")
print("-"*55)
print(f"{'가격':<25} {price_ki:>10.2f}원 {price_no_ki:>14.2f}원")
print(f"{'낙인 영향':<25} {price_no_ki - price_ki:>10.2f}원 ({(price_no_ki-price_ki)/price_no_ki*100:>5.1f}%)")
print(f"{'PDE (111.74원) 비교':<25} {abs(111.74-price_ki):>9.2f}원 {abs(111.74-price_no_ki):>13.2f}원")

print("\n💡 결론:")
print(f"  ✅ 정확한 가격 (KI 포함): {price_ki:.2f}원")
print(f"  ❌ PDE 현재 가격 (KI 무시): 111.74원")
print(f"  📊 오차: {abs(111.74 - price_ki):.2f}원 ({abs(111.74-price_ki)/price_ki*100:.1f}%)")
print(f"  🎯 PDE는 WITHOUT KI와 비슷: {abs(111.74 - price_no_ki):.2f}원 차이")
print("="*70 + "\n")
