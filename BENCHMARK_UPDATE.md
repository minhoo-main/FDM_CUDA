# CPU vs GPU 벤치마크 업데이트

## 📋 변경 사항 (2025-11-14)

### 1. 새로운 벤치마크 프로그램 추가

**파일**: `examples/benchmark_cpu_vs_gpu.cpp`

이 프로그램은 6개의 그리드 크기에 대해 CPU와 GPU 성능을 **직접 비교**합니다.

#### 주요 기능:
- ✅ 각 그리드마다 CPU와 GPU 모두 테스트
- ✅ 실시간으로 승자 표시 (GPU ✓ 또는 CPU ✓)
- ✅ Speedup 계산 (CPU 시간 / GPU 시간)
- ✅ 상세한 분석 리포트
- ✅ CSV 파일 출력 (`cpu_vs_gpu_results.csv`)
- ✅ Python vs C++ vs GPU 비교

#### 출력 예시:
```
[1/6] Testing 100×100×200...
      CPU: 0.035s  GPU: 0.005s  → GPU 7.0× faster ✓

[2/6] Testing 100×100×1000...
      CPU: 0.173s  GPU: 0.015s  → GPU 11.5× faster ✓
...
```

### 2. CMakeLists.txt 업데이트

새로운 실행 파일 추가:
```cmake
add_executable(benchmark_cpu_vs_gpu examples/benchmark_cpu_vs_gpu.cpp)
target_link_libraries(benchmark_cpu_vs_gpu els_pricer_lib)
```

### 3. Colab 노트북 업데이트

**파일**: `ELS_Pricer_GPU_Benchmark.ipynb`

#### 변경 내용:
- Step 5 추가: 벤치마크 준비 안내
- **Step 6 업데이트**: CPU vs GPU 종합 벤치마크 실행
  - 기존: 별도의 CPU, GPU 벤치마크
  - 신규: `benchmark_cpu_vs_gpu` 실행으로 통합
- Step 7: 결과 데이터 확인 및 승자 표시
- Step 8: 성능 비교 시각화 (4개 그래프)
  1. 실행 시간 비교 (CPU vs GPU)
  2. GPU 가속 비율 (color-coded)
  3. 가격 수렴성
  4. 처리 속도 (M points/sec)
- Step 9: 결과 파일 다운로드

#### 새로운 시각화 기능:
- 그리드별 승자를 색상으로 표시 (녹색 = GPU, 빨강 = CPU)
- 가속비 값 직접 표시
- 처리량 비교 차트

### 4. tar.gz 파일 업데이트

**파일**: `/home/minhoo/els-pricer-cpp.tar.gz` (100KB)

포함된 내용:
- ✅ `examples/benchmark_cpu_vs_gpu.cpp` (새 파일)
- ✅ 업데이트된 `CMakeLists.txt`
- ✅ 업데이트된 `ELS_Pricer_GPU_Benchmark.ipynb`
- ✅ 모든 기존 소스 파일

---

## 🚀 사용 방법

### Google Colab에서:

1. **파일 업로드**:
   - `els-pricer-cpp.tar.gz` (100KB)

2. **노트북 업로드**:
   - `ELS_Pricer_GPU_Benchmark.ipynb`

3. **GPU 활성화**:
   - 런타임 → 런타임 유형 변경 → GPU 선택

4. **셀 실행**:
   - Step 1-4: 환경 설정 및 빌드
   - Step 5-6: CPU vs GPU 벤치마크 실행
   - Step 7-9: 결과 분석 및 다운로드

### 로컬에서 (CUDA 환경):

```bash
cd els-pricer-cpp
mkdir build && cd build
cmake ..
make -j4
./benchmark_cpu_vs_gpu
```

---

## 📊 테스트 그리드 (6개)

| # | Grid Size | Total Points | CPU 예상 | GPU 예상 | GPU Speedup |
|---|-----------|--------------|----------|----------|-------------|
| 1 | 100×100×200 | 2M | 0.04초 | 0.005초 | ~7-8× |
| 2 | 100×100×1000 | 10M | 0.17초 | 0.015초 | ~11× |
| 3 | 200×200×200 | 8M | 0.14초 | 0.012초 | ~12× |
| 4 | 200×200×1000 | 40M | 0.68초 | 0.050초 | ~14× |
| 5 | 400×400×200 | 32M | 0.62초 | 0.045초 | ~14× |
| 6 | 400×400×1000 | 160M | 3.02초 | 0.200초 | ~15× |

---

## 📈 예상 결과

### 성능 크로스오버:
GPU는 작은 그리드에서도 이미 빠르지만, 그리드 크기가 클수록 가속 비율이 증가합니다.

### 가격 수렴:
모든 그리드에서 ELS 가격은 ~113.33으로 수렴합니다 (표준편차 < 0.001).

### 전체 가속:
- Python 대비 C++ CPU: ~115-200×
- C++ CPU 대비 GPU: ~7-15×
- **Python 대비 GPU**: **~1000-3000×** 🚀

---

## 💡 주요 개선사항

### 이전 방식:
- CPU 벤치마크와 GPU 벤치마크를 별도로 실행
- 결과를 수동으로 비교해야 함
- 승자 확인이 불명확

### 새로운 방식:
- **단일 프로그램**으로 CPU와 GPU 모두 테스트
- **자동으로 승자 표시** (✓ 마크)
- **Speedup 계산** 자동화
- **성능 크로스오버 분석** 포함
- **CSV 출력**으로 데이터 분석 용이
- **Python 비교** 포함

---

## 🎯 다음 단계

1. **Colab에서 테스트**: GPU 성능 확인
2. **결과 분석**: 어떤 그리드에서 GPU가 효과적인지 확인
3. **최적화**: GPU가 느린 케이스가 있다면 원인 분석
4. **프로덕션**: 실제 ELS 상품 가격 계산에 활용

---

**업데이트 날짜**: 2025-11-14
**파일 위치**: `/home/minhoo/els-pricer-cpp/`
