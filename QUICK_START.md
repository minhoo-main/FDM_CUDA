# ELS Pricer - Quick Start

## 📦 Colab용 패키지 파일

**파일명**: `els-pricer-colab.tar.gz` (26KB)

## 🚀 빠른 시작 (Google Colab)

### 1단계: 파일 업로드
```python
# Colab에서 실행
from google.colab import files
uploaded = files.upload()  # els-pricer-colab.tar.gz 선택
```

### 2단계: 압축 해제 및 빌드
```python
!tar -xzf els-pricer-colab.tar.gz
%cd els-pricer-cpp
!bash colab_setup.sh
```

### 3단계: 테스트 실행
```python
# CPU 가격 검증
!./validate_price_cpu
```

**기대 결과**: ~103.9원 (몬테칼로 104.44원과 0.5% 차이)

### 4단계 (GPU 있으면): 성능 비교
```python
# Runtime > Change runtime type > GPU 선택 후
!./benchmark_gpu
```

## 📊 주요 개선사항

### Before (버그 있음)
- 가격: **111.74원**
- 오차: **7.0%** (몬테칼로 대비)
- 조기상환: ❌ 작동 안 함

### After (버그 수정)
- 가격: **103.9원** ✅
- 오차: **0.5%**
- 조기상환: ✅ 정상 작동 (50-58% 상환)

## 🐛 수정된 버그

1. **조기상환 로직**
   - 이전: `V = std::max(V, payoff)` (선택적)
   - 수정: `V = payoff` (강제)

2. **타임스텝 인덱싱** (치명적)
   - 이전: 루프 `Nt_-1 → 0` (마지막 관찰 누락)
   - 수정: 루프 `Nt_ → 1` (모든 관찰 포함)

## 📁 포함 파일

```
els-pricer-cpp/
├── src/                 # CPU 소스코드
│   ├── Grid2D.cpp
│   ├── ELSProduct.cpp
│   ├── ADISolver.cpp    ← 버그 수정됨!
│   └── cuda/            # GPU 소스코드
│       ├── batched_thomas.cu
│       └── CUDAADISolver.cu  ← 버그 수정됨!
├── include/             # 헤더 파일
├── examples/
│   ├── validate_price_cpu.cpp
│   └── benchmark_gpu.cpp
├── colab_setup.sh       # 자동 빌드 스크립트
├── ELS_Pricer_Colab.ipynb
└── 문서들
    ├── COLAB_GUIDE.md
    ├── BUGFIX_SUMMARY.txt
    └── BUGFIX_EARLY_REDEMPTION.md
```

## ⚠️ 알려진 제한사항

**낙인(KI) 추적 미구현**
- 현재 가격: 103.9원 (KI 무시)
- 정확한 가격: ~93.92원 (KI 포함)
- 해결: 2-state PDE 구현 필요 (향후 작업)

KI가 있는 상품은 몬테칼로 시뮬레이션 사용 권장

## 📚 상세 문서

- **COLAB_GUIDE.md**: 전체 사용 가이드
- **BUGFIX_SUMMARY.txt**: 버그 수정 요약
- **BUGFIX_EARLY_REDEMPTION.md**: 버그 상세 분석

## 💡 테스트 파라미터

```cpp
σ₁ = 15.2%, σ₂ = 40.4%, ρ = 0.61
r = 3.477%, q₁ = 1.5%, q₂ = 2.0%
Maturity = 3년
Barriers = [85%, 85%, 80%, 80%, 75%, 70%]
KI = 45%
```

## 🎯 성능

| Grid Size | CPU Time | GPU Time (T4) | Speedup |
|-----------|----------|---------------|---------|
| 100×100×200 | ~0.1s | ~0.02s | ~5x |
| 500×500×1000 | ~10s | ~0.5s | ~20x |
| 1000×1000×2000 | ~100s | ~2s | ~50x |

---

**Ready for production**: Non-KI ELS products ✅
