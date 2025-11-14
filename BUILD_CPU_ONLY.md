# CPU-Only Build Guide

이 가이드는 CUDA/GPU 없이 CPU 버전만 빌드하는 방법을 설명합니다.

## 빠른 시작

```bash
cd /home/minhoo/els-pricer-cpp

# 빌드
make -f Makefile.cpu

# 실행
./cpu_pricer    # 벤치마크
./cpu_test      # 테스트
```

## 테스트 결과 (검증 완료 ✓)

### 단위 테스트
```
Testing Grid2D... ✓ PASSED
Testing ELSProduct... ✓ PASSED
Testing CPU ADI Solver...
  Computed price: 113.3317
  ✓ PASSED (price=113.3317, time=0.002s)
Testing Grid Convergence...
  40×40×80:   113.3317
  60×60×120:  113.3310
  80×80×160:  113.3301
  ✓ PASSED
```

### 벤치마크 결과

| Grid | Size | Price | Time (s) | Points/sec |
|------|------|-------|----------|------------|
| Small | 50×50×100 | 113.3308 | 0.004 | 6.04e+07 |
| Medium | 100×100×200 | 113.3300 | 0.034 | 5.81e+07 |
| Large | 150×150×300 | 113.3298 | 0.120 | 5.61e+07 |

### 가격 수렴성

그리드 크기가 증가할수록 가격이 안정적으로 수렴:
- 40×40 → 60×60: 변화 0.0007
- 60×60 → 80×80: 변화 0.0009

## 시스템 요구사항

- **컴파일러**: GCC 9+ 또는 Clang 10+ (C++17 지원)
- **메모리**: 최소 512MB
- **OS**: Linux, macOS, Windows (WSL)

## 빌드 옵션

### 디버그 모드
```bash
make -f Makefile.cpu clean
g++ -std=c++17 -g -O0 -Wall -Wextra -Iinclude \
    src/*.cpp examples/main_cpu.cpp -o cpu_pricer_debug -lm
```

### 릴리스 모드 (기본)
```bash
make -f Makefile.cpu
# -O3 최적화가 자동 적용됨
```

## 성능 분석

### 처리 속도
- **평균**: ~60M points/second
- **150×150×300 그리드**: 6.75M points 처리에 0.12초

### Python 대비 성능
예상 성능 (Python NumPy 대비):
- C++ CPU: 3-5배 빠름
- 작은 그리드에서도 일관된 성능

## 문제 해결

### 컴파일 오류: C++17 not supported
```bash
# GCC 버전 확인
g++ --version

# GCC 9+ 필요. 업그레이드:
# Ubuntu/Debian:
sudo apt install g++-11
export CXX=g++-11
```

### 링크 오류: undefined reference to sqrt
```bash
# -lm 플래그 추가 확인
g++ ... -lm
```

## 다음 단계

1. **더 큰 그리드 테스트**: Makefile.cpu 수정
2. **Python 비교**: 원본 프로젝트와 성능 비교
3. **GPU 버전**: CUDA 설치 후 GPU 버전 빌드 시도

## 파일 구조

생성된 실행 파일:
```
cpu_pricer    # 벤치마크 프로그램
cpu_test      # 테스트 스위트
```

소스 파일:
```
examples/main_cpu.cpp    # CPU 전용 메인
tests/test_cpu.cpp       # CPU 전용 테스트
Makefile.cpu             # CPU 빌드 파일
```
