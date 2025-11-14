#!/usr/bin/env python3
"""Debug MC - print detailed statistics"""
import numpy as np

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

n_paths = 100000
n_steps_per_obs = 42  # 0.5년당 42 스텝 (약 주간)
np.random.seed(42)

L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])

print("="*70)
print("  Detailed MC Debug (100,000 paths)")
print("="*70)

# WITHOUT KI
print("\n[WITHOUT KI]")
np.random.seed(42)
payoffs = []
redemption_stats = [0] * (len(obs_dates) + 1)

for path in range(n_paths):
    S1, S2 = S1_0, S2_0

    for obs_idx, obs_t in enumerate(obs_dates):
        dt = obs_t / n_steps_per_obs if obs_idx == 0 else (obs_t - obs_dates[obs_idx-1]) / n_steps_per_obs

        for _ in range(n_steps_per_obs):
            Z = np.random.randn(2)
            dW = L @ Z * np.sqrt(dt)
            S1 *= np.exp((r - q1 - 0.5*sigma1**2)*dt + sigma1*dW[0])
            S2 *= np.exp((r - q2 - 0.5*sigma2**2)*dt + sigma2*dW[1])

        worst = min(S1/S1_0, S2/S2_0)
        if worst >= barriers[obs_idx]:
            payoff = (principal + coupons[obs_idx]) * np.exp(-r * obs_t)
            payoffs.append(payoff)
            redemption_stats[obs_idx] += 1
            break
    else:
        # Maturity
        payoff = (principal + coupons[-1]) * np.exp(-r * T)
        payoffs.append(payoff)
        redemption_stats[-1] += 1

price = np.mean(payoffs)
print(f"  Price: {price:.4f}원")
print(f"\n  Redemption stats:")
for i, (obs_t, count) in enumerate(zip(obs_dates, redemption_stats[:-1])):
    print(f"    {obs_t:.1f}년: {count:6d} ({count/n_paths*100:5.2f}%)")
print(f"    Maturity: {redemption_stats[-1]:6d} ({redemption_stats[-1]/n_paths*100:5.2f}%)")

# Average payoff detail
total_redeemed = sum(redemption_stats[:-1])
avg_redemption_time = sum(obs_dates[i] * redemption_stats[i] for i in range(len(obs_dates))) / total_redeemed if total_redeemed > 0 else 0
print(f"\n  Average redemption time: {avg_redemption_time:.2f}년 (조기상환된 경우만)")
print(f"  조기상환 비율: {total_redeemed/n_paths*100:.2f}%")

# Discount factor analysis
print(f"\n  할인 계수 분석:")
for i, obs_t in enumerate(obs_dates):
    disc = np.exp(-r * obs_t)
    undiscounted = principal + coupons[i]
    discounted = undiscounted * disc
    print(f"    {obs_t:.1f}년: {undiscounted:.2f}원 → {discounted:.2f}원 (할인율 {disc:.4f})")

maturity_disc = np.exp(-r * T)
maturity_payoff = principal + coupons[-1]
print(f"    만기:  {maturity_payoff:.2f}원 → {maturity_payoff * maturity_disc:.2f}원 (할인율 {maturity_disc:.4f})")

print("\n" + "="*70)
