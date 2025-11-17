# FP32 vs FP64 성능 비교 보고서 생성 가이드

## 📊 보고서 내용

이 노트북은 다음을 포함한 종합 성능 분석 보고서를 생성합니다:

1. **4가지 구성 비교**:
   - CPU FP64 (기준)
   - CPU FP32
   - GPU FP64
   - GPU FP32 (최적)

2. **시각화**:
   - 절대 실행 시간 비교
   - GPU Speedup (CPU 대비)
   - FP64→FP32 개선 효과
   - 처리량 비교 (points/sec)
   - 하드웨어 활용률 분석
   - 비용 효율성 분석

3. **통계 분석**:
   - 상세 성능 테이블
   - 가격 정확도 검증
   - 핵심 발견사항

4. **출력 파일**:
   - PNG 이미지 3개 (고해상도)
   - CSV 데이터
   - HTML 보고서

---

## 🚀 사용 방법

### 로컬에서 실행:

```bash
# 1. Jupyter 설치 (없는 경우)
pip install jupyter pandas matplotlib seaborn numpy

# 2. 노트북 실행
cd /path/to/els-pricer-cpp
jupyter notebook FP32_vs_FP64_Performance_Report.ipynb

# 3. 브라우저에서 열기
# Run → Run All Cells

# 4. 결과 확인
# - performance_comparison_fp32_vs_fp64.png
# - hardware_utilization_analysis.png
# - cost_effectiveness_analysis.png
# - performance_comparison_detailed.csv
# - performance_report_fp32_vs_fp64.html ← 브라우저로 열기!
```

---

### Google Colab에서 실행:

```python
# 1. Colab에서 새 노트북 열기

# 2. 첫 번째 셀에서 GitHub에서 가져오기
!git clone https://github.com/minhoo-main/FDM_CUDA.git
%cd FDM_CUDA

# 3. 노트북 열기
# 또는 직접 업로드: FP32_vs_FP64_Performance_Report.ipynb

# 4. Runtime → Run all
```

---

## 📁 생성되는 파일

```
els-pricer-cpp/
├── performance_comparison_fp32_vs_fp64.png    ← 메인 성능 비교
├── hardware_utilization_analysis.png         ← GPU 하드웨어 분석
├── cost_effectiveness_analysis.png           ← 비용 효율성
├── performance_comparison_detailed.csv       ← 상세 데이터
└── performance_report_fp32_vs_fp64.html      ← 종합 보고서 (열기!)
```

---

## 📊 예상 결과 (400×400×1000 기준)

| 구성 | 시간 | 개선율 | 비용 |
|------|------|--------|------|
| CPU FP64 | 3.67초 | 1.0× | $0 |
| CPU FP32 | 3.49초 | 1.05× | $0 |
| GPU FP64 | 2.15초 | 1.7× | $0 |
| **GPU FP32** | **0.29초** | **12.6×** | **$0** ✅ |
| A100 (FP64) | 0.26초 | 14× | $13,000 |

**결론**: GPU FP32가 최적 (무료 + 12배 빠름!)

---

## 🎯 핵심 발견

### CPU: FP64 → FP32 (작은 개선)
- 개선율: **1.05×** (5% 빠름)
- 이유: FP32와 FP64가 **같은 ALU** 사용

### GPU: FP64 → FP32 (극적 개선)
- 개선율: **7.3×** (730% 빠름!)
- 이유: FP32 코어가 **32배 많음** (80개 vs 2560개)

### 병렬화 효율
- FP64: 400개 시스템 → 5 배치 (80개씩)
- FP32: 400개 시스템 → 1 배치 (한 번에!)

---

## 💡 사용 팁

### 데이터 수정하기:

노트북의 **섹션 1**에서 실제 측정 데이터를 수정하세요:

```python
# FP32 실제 측정 데이터
fp32_data = {
    'Grid_N1': [100, 100, 200, 200, 400, 400],
    'Grid_N2': [100, 100, 200, 200, 400, 400],
    'Grid_Nt': [200, 1000, 200, 1000, 200, 1000],
    'CPU_Time_FP32': [0.056, 0.240, 0.182, 0.834, 0.688, 3.492],  # ← 여기 수정
    'GPU_Time_FP32': [0.016, 0.078, 0.031, 0.154, 0.041, 0.294],  # ← 여기 수정
    # ...
}
```

### 추가 그리드 크기 테스트:

더 많은 그리드 크기를 추가하려면 데이터에 행을 추가하세요:

```python
'Grid_N1': [100, 100, 200, 200, 400, 400, 600, 600],  # 600×600 추가
'Grid_N2': [100, 100, 200, 200, 400, 400, 600, 600],
'Grid_Nt': [200, 1000, 200, 1000, 200, 1000, 200, 1000],
# ...
```

---

## ❓ 문제 해결

### 한글이 안 보여요:
```python
# 노트북 첫 번째 코드 셀에서:
plt.rcParams['font.family'] = 'DejaVu Sans'  # 또는 다른 폰트
```

### 그래프가 너무 작아요:
```python
plt.rcParams['figure.figsize'] = (20, 14)  # 크기 조정
```

### CSV가 깨져요:
```python
comparison_table.to_csv('file.csv', index=False, encoding='utf-8-sig')
```

---

## 📧 지원

문제가 있으면 GitHub Issues에 등록하세요:
https://github.com/minhoo-main/FDM_CUDA/issues

---

## 🎉 완료!

노트북을 실행하면 자동으로:
- ✅ 3개의 고해상도 그래프 생성
- ✅ 상세 통계 테이블 출력
- ✅ HTML 보고서 생성
- ✅ CSV 데이터 내보내기

**HTML 보고서를 브라우저로 열어서 결과를 확인하세요!** 🚀
