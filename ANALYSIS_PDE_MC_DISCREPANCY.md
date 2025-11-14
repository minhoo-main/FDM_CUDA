# PDE vs Monte Carlo 가격 차이 분석 (KI 무시 조건)

## 📊 현재 결과

```
PDE (100×100×500, KI=false):  111.74원
MC (100k paths, WITHOUT KI):  104.44원
차이:                           7.30원 (6.7%)
```

---

## 🔍 가능한 원인들

### 1. **조기상환 타이밍 문제**

#### PDE 방식:
```cpp
// Backward time stepping
for (int n = Nt_ - 1; n >= 0; --n) {
    solveS1Direction();
    solveS2Direction();
    applyBoundaryConditions();

    // Check if this timestep matches observation date
    if (n == obsIndices[obsIdx]) {
        applyEarlyRedemption();  // observation 시점에서만
    }
}
```

**타임스텝 매칭 문제:**
- `obsIndices`는 가장 가까운 timestep으로 매핑
- Observation date = 0.5년, 1.0년, ...
- dt = 3.0 / 500 = 0.006년
- 0.5년 = timestep 83.33 → 반올림하면 83 또는 84

**정확성 이슈:**
```
실제 observation: t = 0.5년 정확히
PDE timestep:     t = 83 × 0.006 = 0.498년 (2일 차이!)
```

이로 인해:
- 조기상환 체크가 약간 다른 시점에서 발생
- 자산 가격이 미묘하게 다름

---

### 2. **할인율 적용 방식 차이**

#### Monte Carlo:
```python
if worst >= barrier:
    pv = (principal + coupon) * exp(-r * obs_t)
    # 관찰 시점에서 즉시 할인
```

#### PDE:
```cpp
// Backward PDE에서 할인은 자동으로 적용됨
V[i][j] = payoff;  // 조기상환 시점의 payoff
// 이후 backward 진행하면서 자동 할인
```

**차이점:**
- MC: 명시적으로 `exp(-r*t)` 곱함
- PDE: 시간 역행하면서 암묵적 할인 (PDE 계수에 포함)

**검증 필요:**
```cpp
// PDE 계수 (ADISolver.cpp:48)
beta[i] = 1.0 + dt * (a + 0.5 * r);
// 0.5 * r ?? 왜 0.5?
```

**이론적 값:**
```
Implicit scheme: (1 - 0.5*r*dt) * V^{n+1} = (1 + 0.5*r*dt) * V^n
Crank-Nicolson 방식

할인 효과: exp(-r*dt) ≈ 1 - r*dt (1차 근사)
하지만 실제로는: (1 + 0.5*r*dt) / (1 - 0.5*r*dt) ≈ exp(r*dt) ??
```

**아! 부호가 반대일 수 있음!**

---

### 3. **Terminal Payoff 문제**

#### 현재 설정 (KI=false):
```cpp
// Maturity payoff
if (!kiOccurred) {
    return principal + coupons.back();  // 100 + 24 = 124
}
```

#### MC에서 실제 발생:
```
Maturity 도달: 14.41% 경로만
평균 payoff: 124 × exp(-r×3) = 111.72원

조기상환: 85.59% 경로
평균 payoff: ~102원 (0.5년에 많이 상환)

가중평균: 0.8559 × 102 + 0.1441 × 111.72
        = 87.30 + 16.08 = 103.38원
```

**PDE가 111.74원인 이유:**
→ **조기상환이 거의 일어나지 않고 있음!**

---

## 🐛 핵심 문제 발견!

### PDE에서 조기상환이 안 일어나는 이유:

```cpp
void ADISolver::applyEarlyRedemption(...) {
    for (int i = 0; i < N1_; ++i) {
        for (int j = 0; j < N2_; ++j) {
            auto result = product.checkEarlyRedemption(S1[i], S2[j], obsIdx);
            if (result.isRedeemed) {
                V[i * N2_ + j] = result.payoff;  // 예: 104원
            }
        }
    }
}
```

**Backward PDE에서:**
```
시간 T=3년 (만기):
  모든 V[i][j] = 124원 (terminal payoff)

시간 T=2.99년 (backward 1 step):
  V[i][j] ≈ 124 × (1 - r×dt) ≈ 123.99원

...

시간 T=3.0년 관찰 (마지막 observation):
  조건 체크: worst >= 70%?
  만약 YES → V[i][j] = 124원
  이미 124원이므로 변화 없음!

시간 T=2.5년 관찰:
  조건 체크: worst >= 75%?
  만약 YES → V[i][j] = 120원
  하지만 현재 V[i][j] ≈ 123원 (만기에서 역행)
  120 < 123 이므로... 120으로 바뀜! ✓

시간 T=0.5년 관찰 (첫 번째):
  조건 체크: worst >= 85%?
  만약 YES → V[i][j] = 104원
  하지만 현재 V[i][j] ≈ 120원 (나중 observation들의 영향)
  104 < 120 이므로... 104로 바뀜! ✓
```

**아! 작동은 하는데...**

---

## 🎯 진짜 문제: 조기상환 조건이 너무 빡빡함

### MC 결과 (100k paths):
```
0.5년 관찰 (barrier 85%): 66.33% 상환
1.0년 관찰 (barrier 85%):  7.91% 상환
...
만기 도달: 14.41%
```

**첫 번째 관찰에서 66%가 상환!**

### PDE에서는?

**그리드 포인트 (S1_0=100, S2_0=100):**
```
100×100 그리드
S1 range: [0, 300]
S2 range: [0, 300]

중심점: i=50, j=50 → S1=150, S2=150
worst = min(150/100, 150/100) = 150% > 85% ✓ → 상환!

하지만 우리가 관심있는 것은: S1=100, S2=100 근처
```

**보간 문제:**
```
Grid point 찾기 (findS1Index, findS2Index):
S1=100 → 어느 그리드 점?
  Grid: [0, 3.33, 6.67, ..., 100, ..., 300]
  정확히 100이 그리드 점이 아닐 수 있음!

보간으로 가격 계산:
  가까운 4개 점에서 보간
  조기상환이 이미 적용된 값들의 보간
```

---

## 💡 테스트 방법

### 조기상환이 제대로 작동하는지 확인:

```cpp
// Debug: 각 observation에서 몇 개 포인트가 상환되는지 출력
int redeemed_count = 0;
for (int i = 0; i < N1_; ++i) {
    for (int j = 0; j < N2_; ++j) {
        auto result = product.checkEarlyRedemption(S1[i], S2[j], obsIdx);
        if (result.isRedeemed) {
            redeemed_count++;
            V[i * N2_ + j] = result.payoff;
        }
    }
}
std::cout << "Observation " << obsIdx << ": "
          << redeemed_count << " / " << (N1_*N2_)
          << " points redeemed\n";
```

### 예상 결과:
```
Observation 0 (0.5년, 85%): ~40-50% 포인트 상환?
Observation 5 (3.0년, 70%): ~80% 포인트 상환?
```

만약 거의 안 일어나면 → 조기상환 로직에 버그
만약 많이 일어나는데도 가격 111원이면 → 다른 문제

---

## 🔬 추가 검증 필요

### 1. Observation 타이밍 출력:
```cpp
std::cout << "Observation dates: ";
for (double t : observationDates) {
    int idx = findClosestTimeStep(t);
    double actual_t = idx * dt;
    std::cout << t << " (" << actual_t << ") ";
}
```

### 2. 특정 포인트 추적:
```cpp
// (S1=100, S2=100) 근처의 V 값 출력
int i0 = findS1Index(100);
int j0 = findS2Index(100);
std::cout << "At (100, 100): V = " << V[i0*N2 + j0] << "\n";
```

### 3. 조기상환 적용 전후 비교:
```cpp
double V_before = V[i0*N2 + j0];
applyEarlyRedemption();
double V_after = V[i0*N2 + j0];
std::cout << "Before: " << V_before << ", After: " << V_after << "\n";
```

---

## 📝 다음 단계

1. ✅ **디버그 출력 추가**: 조기상환이 몇 번 일어나는지 확인
2. ⬜ **Observation 타이밍 검증**: PDE timestep vs 실제 날짜
3. ⬜ **할인율 검증**: PDE 계수의 0.5*r이 맞는지
4. ⬜ **보간 정확도**: (S1_0, S2_0) 위치의 정확한 값

---

**작성일**: 2025-11-14
**상태**: 분석 중 - 조기상환 작동 여부 확인 필요
