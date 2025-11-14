# 🐛 Critical Bug: KI (Knock-In) Tracking Not Implemented

## 문제 발견

**가격이 100원 이상 나오는 이유**: 낙인(KI) 추적이 구현되지 않음

### 현재 코드 (`src/ADISolver.cpp:279`):

```cpp
// TODO: Track KI properly (for now assume no KI)
bool kiOccurred = false;  // ← 항상 false!

for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
        V_T[i * N2 + j] = product.payoffAtMaturity(S1[i], S2[j], kiOccurred);
        //                                                         ↑ 항상 false
    }
}
```

### Terminal Payoff (`src/ELSProduct.cpp:67`):

```cpp
double ELSProduct::payoffAtMaturity(double S1, double S2, bool kiOccurred) const {
    double perf = performance(S1, S2);  // worst-of

    if (kiOccurred) {
        // KI 발생: 원금 손실 가능
        return principal_ * std::min(1.0, perf);
    } else {
        // KI 없음: 원금 + 마지막 쿠폰
        return principal_ + coupons_.back();  // 100 + 24 = 124
    }
}
```

**결과**:
- 모든 경로에서 `kiOccurred = false`로 가정
- 모든 terminal payoff = 124원
- PDE backward 계산 → 현재 가격 ~112원 (할인 후)
- **완전히 틀린 가격!**

---

## 올바른 구현

### KI 추적 필요:

ELS는 만기까지 **단 한 번이라도** 최악 자산이 낙인 베리어(45%) 이하로 떨어지면:
- 낙인 발생 (KI occurred)
- 만기 시 손실 가능

### 2가지 방법:

#### 방법 1: **2-상태 PDE** (정확, 복잡)

```cpp
// 각 그리드 포인트에서 2개 값 추적
V_no_ki[i][j]  // 낙인 아직 안 일어난 경우
V_ki[i][j]     // 낙인 이미 일어난 경우

// Backward PDE:
for (int n = Nt_ - 1; n >= 0; --n) {
    // No-KI 상태 업데이트
    solve_ADI(V_no_ki);

    // KI 상태 업데이트
    solve_ADI(V_ki);

    // KI 체크: 베리어 이하면 No-KI → KI 전환
    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            double perf = worst(S1[i], S2[j]);
            if (perf < kiBarrier) {
                V_no_ki[i][j] = V_ki[i][j];  // 낙인 발생!
            }
        }
    }
}
```

**비용**:
- 메모리: 2배
- 계산: 2배
- 복잡도: 높음

#### 방법 2: **Monte Carlo 검증** (간단, 정확)

PDE로 KI 추적은 복잡하므로, Monte Carlo로 검증:

```python
# Python으로 간단히 확인
import numpy as np

S1_0, S2_0 = 100, 100
sigma1, sigma2 = 0.152, 0.404
rho = 0.61
r, q1, q2 = 0.03477, 0.015, 0.02
T = 3.0
ki_barrier = 0.45
obs_dates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
barriers = [0.85, 0.85, 0.80, 0.80, 0.75, 0.70]
coupons = [4, 8, 12, 16, 20, 24]

n_paths = 100000
payoffs = []

for _ in range(n_paths):
    # Simulate 2-asset path
    S1, S2 = simulate_correlated_paths(S1_0, S2_0, sigma1, sigma2, rho, r, q1, q2, T)

    # Check KI
    worst_perf = min(S1 / S1_0, S2 / S2_0)
    ki_occurred = worst_perf.min() < ki_barrier

    # Check early redemption
    redeemed = False
    for t, bar, coup in zip(obs_dates, barriers, coupons):
        if min(S1[t], S2[t]) / 100 >= bar:
            payoffs.append(100 + coup)
            redeemed = True
            break

    # Maturity
    if not redeemed:
        if ki_occurred:
            payoffs.append(100 * min(1.0, worst_perf[-1]))
        else:
            payoffs.append(100 + 24)

price = np.mean(payoffs) * np.exp(-r * T)
print(f"MC Price: {price:.2f}")
```

---

## 예상 결과

### 올바른 가격 (KI 추적 시):

```
조기상환 확률이 높지만, 낙인 리스크 존재:
- 자산2 변동성 40.4% → 3년간 45% 이하 확률 높음
- 낙인 발생 시 원금 손실 가능
- 예상 가격: 95-98원 (100원 이하)
```

### 현재 잘못된 가격:

```
낙인 무시 → 항상 124원 받는다고 가정
→ 할인 후 ~112원 (완전히 틀림!)
```

---

## 해결 방법

### 단기 (간단한 수정):

**Upper Bound로 해석**:
```
현재 가격 111.74원 = 낙인이 절대 안 일어난다는 가정
→ 실제 가격의 상한선
→ "최대 111.74원, 실제로는 훨씬 낮음"
```

### 중기 (2-상태 PDE 구현):

1. `V_no_ki`, `V_ki` 분리
2. Backward 시 KI 체크
3. 메모리 2배, 계산 2배

### 장기 (Monte Carlo 추가):

1. Monte Carlo 시뮬레이션 구현
2. PDE와 MC 결과 비교
3. MC가 정확한 KI 추적

---

## 코드 수정 필요 파일

1. ✅ **`src/ADISolver.cpp`**:
   - `priceELS()` 함수에서 `kiOccurred` 추적
   - `solveWithEarlyRedemption()` 수정

2. ✅ **`include/ADISolver.h`**:
   - `std::vector<double> V_ki_` 추가

3. ✅ **`src/cuda/CUDAADISolver.cu`**:
   - GPU 버전도 동일 수정

---

## 결론

**당신이 맞습니다!**

가격이 100원 이상 나온 것은:
- ❌ 경계 조건 문제 아님
- ❌ Worst-of 조건 문제 아님
- ✅ **KI 추적이 구현되지 않은 버그**

올바른 가격은 **95-98원** 정도일 것입니다.

---

**작성일**: 2025-11-14
**우선순위**: Critical (가격 계산 오류)
**해결 난이도**: Medium-High (2-상태 PDE 또는 MC 구현 필요)
