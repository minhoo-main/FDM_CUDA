# Google Colab 사용 가이드

## 1. 파일 업로드

1. Google Colab 노트북 생성
2. `els-pricer-colab.tar.gz` 파일을 Colab에 업로드

## 2. GPU 런타임 설정 (선택)

**메뉴**: Runtime → Change runtime type → Hardware accelerator → GPU (T4 추천)

## 3. 노트북 실행

업로드한 `ELS_Pricer_Colab.ipynb`를 열고 셀을 순서대로 실행하거나,
아래 코드를 새 노트북에서 실행:

### 기본 설정 및 빌드

```python
# 1. 파일 압축 해제
!tar -xzf els-pricer-colab.tar.gz
%cd els-pricer-cpp

# 2. 빌드 (CPU + GPU 자동 감지)
!bash colab_setup.sh
```

### CPU 검증 테스트

```python
# 수정된 코드로 가격 검증
!./validate_price_cpu
```

**기대 결과**: ~103.9원 (Monte Carlo: 104.44원)

### GPU 벤치마크 (GPU 사용 시)

```python
# CPU vs GPU 성능 비교
!./benchmark_gpu
```

**기대 성능**:
- 100×100×200: ~2-5x speedup
- 500×500×1000: ~10-50x speedup
- 1000×1000×2000: ~50-200x speedup

### 다중 해상도 테스트

```python
# 그리드 수렴성 확인
!./test_final_validation
```

## 4. 버그 수정 확인

```python
# 버그 수정 요약 보기
!cat BUGFIX_SUMMARY.txt
```

```python
# 상세 버그 문서 보기
!cat BUGFIX_EARLY_REDEMPTION.md
```

## 5. 주요 개선사항

### ✅ 수정된 버그들
1. **조기상환 로직**: 선택적 → 강제 상환
2. **타임스텝 인덱싱**: 마지막 관찰시점 포함

### 📊 결과
- **수정 전**: 111.74원 (7% 오차)
- **수정 후**: 103.9원 (0.5% 오차)
- **몬테칼로**: 104.44원 (벤치마크)

### ⚠️ 알려진 제한사항
- **낙인(KI) 추적 미구현**: 현재 `kiOccurred = false` 가정
- 낙인 포함 정확한 가격: ~93.92원
- 해결 방법:
  1. 2-state PDE 구현 (향후 작업)
  2. 또는 Monte Carlo 사용

## 6. 커스텀 테스트

### 파라미터 수정하여 테스트

```python
# validate_price_cpu.cpp를 수정하여 다른 상품 테스트
# 예: 다른 변동성, 배리어, 쿠폰 등

# 재빌드
!g++ -std=c++17 -O3 -Iinclude examples/validate_price_cpu.cpp \
     src/Grid2D.o src/ELSProduct.o src/ADISolver.o -o validate_price_cpu

# 실행
!./validate_price_cpu
```

### Python으로 결과 분석

```python
import subprocess
import re

# 가격 추출
result = subprocess.run(['./validate_price_cpu'],
                       capture_output=True, text=True)
output = result.stdout

# 정규식으로 가격 파싱
price_match = re.search(r'Final price: ([\d.]+)', output)
if price_match:
    price = float(price_match.group(1))
    print(f"Extracted price: {price:.2f}원")
```

## 7. 파일 구조

```
els-pricer-cpp/
├── include/              # 헤더 파일
│   ├── Grid2D.h
│   ├── ELSProduct.h
│   ├── ADISolver.h
│   └── CUDAADISolver.cuh
├── src/                  # CPU 소스
│   ├── Grid2D.cpp
│   ├── ELSProduct.cpp
│   ├── ADISolver.cpp
│   └── cuda/            # GPU 소스
│       ├── batched_thomas.cu
│       └── CUDAADISolver.cu
├── examples/
│   ├── validate_price_cpu.cpp
│   └── benchmark_gpu.cpp
├── colab_setup.sh       # 빌드 스크립트
├── ELS_Pricer_Colab.ipynb
└── 문서들...
```

## 8. 문제 해결

### "CUDA not found" 메시지
- 정상입니다! CPU 버전만 빌드됩니다
- GPU 벤치마크를 원하면: Runtime → Change runtime type → GPU 선택 후 재시작

### 컴파일 오류
```python
# 클린 빌드
!rm -f src/*.o src/cuda/*.o validate_price_cpu benchmark_gpu
!bash colab_setup.sh
```

### 메모리 부족 (큰 그리드)
- 그리드 크기 줄이기: 1000×1000 → 500×500
- 또는 GPU 런타임 업그레이드 (Pro/Pro+)

## 9. 성능 팁

### CPU 최적화
- `-O3 -march=native` 이미 적용됨
- 추가 최적화는 컴파일러 플래그 수정

### GPU 최적화
- T4 GPU 권장 (무료)
- A100 (Pro+): 더 빠른 성능
- 큰 그리드에서 최대 성능 발휘

## 10. 추가 리소스

- 버그 수정 상세: `BUGFIX_EARLY_REDEMPTION.md`
- KI 이슈 설명: `KI_TRACKING_BUG.md`
- 전체 요약: `BUGFIX_SUMMARY.txt`

---

**문의**: GitHub Issues 또는 이메일
