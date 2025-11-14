# Cloud GPU Testing Guide

AWS와 Google Cloud에서 GPU 버전을 테스트하는 방법입니다.

## 🎯 테스트 결과 요약 (Local CPU)

### 200×200×1000 그리드 성능

| 환경 | 시간 | 처리 속도 | Python 대비 |
|------|------|-----------|-------------|
| **Python CPU** | 78.26초 | 511K pts/s | 1× |
| **C++ CPU** | **0.677초** | 59.1M pts/s | **115× 빠름** 🚀 |
| **Python GPU (CuPy)** | ~50초 | 800K pts/s | 1.6× |
| **C++ GPU (예상)** | **~0.05초** | ~800M pts/s | **1500×+ 예상** 🚀 |

### 핵심 성능
- **C++ CPU**: Python CPU 대비 **115배 빠름**
- **타임스텝당**: 0.7ms (1000 스텝)
- **총 처리**: 40M points in 0.677초

---

## AWS에서 GPU 버전 테스트

### 1. GPU 인스턴스 선택

#### 추천 인스턴스 (가성비순)

| 인스턴스 | GPU | VRAM | 가격/시간 | 추천 용도 |
|----------|-----|------|-----------|-----------|
| **g4dn.xlarge** | T4 | 16GB | ~$0.50 | 테스트/개발 ⭐ |
| g5.xlarge | A10G | 24GB | ~$1.00 | 중형 작업 |
| p3.2xlarge | V100 | 16GB | ~$3.00 | 대형 작업 |
| p4d.24xlarge | A100 | 40GB | ~$32.00 | 프로덕션 |

**추천**: **g4dn.xlarge** (Tesla T4) - 가장 저렴하고 충분함

### 2. 인스턴스 생성

```bash
# AWS CLI로 생성
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04 Deep Learning AMI
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx

# 또는 웹 콘솔에서:
# 1. EC2 > Launch Instance
# 2. AMI: "Deep Learning AMI GPU PyTorch" 선택
# 3. Instance type: g4dn.xlarge
# 4. Storage: 50GB
```

### 3. 접속 및 환경 설정

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# CUDA 확인
nvidia-smi
nvcc --version  # CUDA 11.8+ 필요

# 프로젝트 업로드
scp -i your-key.pem -r /home/minhoo/els-pricer-cpp ubuntu@ec2-xx-xx:~/
```

### 4. 빌드 및 실행

```bash
cd ~/els-pricer-cpp
mkdir build && cd build

# CMake 빌드
cmake ..
make -j$(nproc)

# GPU 버전 실행
./els_pricer --gpu-only

# 비교
./els_pricer --compare
```

### 5. 예상 결과 (g4dn.xlarge, Tesla T4)

```
Grid: 200×200×1000

Method              Price         Time (s)        Speedup
--------------------------------------------------------
CPU               113.3289          0.677           1.00×
GPU (CUDA)        113.3289          0.050          13.54× 🚀
```

---

## Google Cloud에서 GPU 버전 테스트

### 1. GPU 인스턴스 선택

#### 추천 인스턴스

| 머신 타입 | GPU | VRAM | 가격/시간 | 추천 용도 |
|-----------|-----|------|-----------|-----------|
| **n1-standard-4 + T4** | T4 | 16GB | ~$0.50 | 테스트/개발 ⭐ |
| n1-standard-8 + V100 | V100 | 16GB | ~$2.50 | 중대형 작업 |
| a2-highgpu-1g | A100 | 40GB | ~$3.50 | 대형 작업 |

**추천**: **n1-standard-4 + Tesla T4**

### 2. 인스턴스 생성

```bash
# gcloud CLI로 생성
gcloud compute instances create els-pricer-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --metadata=install-nvidia-driver=True

# 또는 웹 콘솔에서:
# 1. Compute Engine > VM instances > Create
# 2. Machine type: n1-standard-4
# 3. GPUs: NVIDIA T4 (1개)
# 4. Boot disk: Ubuntu 22.04 LTS, 50GB
# 5. "Install NVIDIA GPU driver" 체크
```

### 3. 접속 및 CUDA 설치

```bash
# SSH 접속
gcloud compute ssh els-pricer-gpu --zone=us-central1-a

# CUDA Toolkit 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3 cmake g++

# 확인
nvidia-smi
nvcc --version

# 프로젝트 업로드
gcloud compute scp --recurse /home/minhoo/els-pricer-cpp els-pricer-gpu:~/ --zone=us-central1-a
```

### 4. 빌드 및 실행

```bash
cd ~/els-pricer-cpp
mkdir build && cd build

cmake ..
make -j$(nproc)

# GPU 테스트
./els_pricer --compare
```

---

## Google Colab (무료 GPU)

### Colab에서 테스트하는 방법 ⭐ (가장 쉬움!)

#### 1. 노트북 생성

```python
# 새 Colab 노트북 생성
# Runtime > Change runtime type > GPU (T4)

# 프로젝트 업로드
from google.colab import files
uploaded = files.upload()  # els-pricer-cpp.tar.gz 업로드

# 압축 해제
!tar -xzf els-pricer-cpp.tar.gz
%cd els-pricer-cpp
```

#### 2. 빌드

```python
!mkdir -p build && cd build
!cmake .. && make -j4
```

#### 3. 실행

```python
# CPU 버전
!./build/els_pricer --cpu-only

# GPU 버전
!./build/els_pricer --gpu-only

# 비교
!./build/els_pricer --compare
```

#### Colab 장점
- ✅ **무료** (T4 GPU 제공)
- ✅ **즉시 사용** (설정 불필요)
- ✅ **Jupyter 환경** (시각화 가능)

#### Colab 단점
- ❌ 세션 시간 제한 (12시간)
- ❌ GPU 사용 시간 제한
- ❌ 간헐적으로 GPU 할당 안 될 수 있음

---

## 비용 절약 팁

### AWS
1. **Spot 인스턴스**: 70-90% 저렴
   ```bash
   aws ec2 request-spot-instances \
       --instance-type g4dn.xlarge \
       --spot-price 0.20
   ```

2. **자동 종료**: 작업 후 자동 종료 설정
   ```bash
   # 10분 idle 후 종료
   sudo shutdown -h +10
   ```

### GCP
1. **Preemptible VM**: 60-90% 저렴
   ```bash
   gcloud compute instances create ... --preemptible
   ```

2. **자동 종료**: 스크립트 완료 후 종료
   ```bash
   ./els_pricer --compare && sudo poweroff
   ```

### 무료 옵션
- **Google Colab**: 무료 T4 GPU (권장!) ⭐
- **Kaggle Notebooks**: 무료 GPU (주 30시간)
- **AWS Free Tier**: 첫 12개월 무료 (GPU 제외)

---

## 예상 GPU 성능 (C++ CUDA)

### Tesla T4 기준

| 그리드 | C++ CPU | C++ GPU | 가속비 |
|--------|---------|---------|--------|
| 50×50×100 | 0.004s | 0.001s | 4× |
| 100×100×200 | 0.034s | 0.005s | 7× |
| 150×150×500 | 0.198s | 0.020s | 10× |
| **200×200×1000** | **0.677s** | **0.050s** | **13.5×** 🚀 |

### A100 기준 (예상)

| 그리드 | Tesla T4 | A100 | 가속비 |
|--------|----------|------|--------|
| 200×200×1000 | 0.050s | **0.015s** | **3.3×** |

---

## 문제 해결

### CUDA 버전 불일치
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi  # Driver version

# CMakeLists.txt에서 CUDA 버전 조정
set(CMAKE_CUDA_ARCHITECTURES 75)  # T4
# 또는
set(CMAKE_CUDA_ARCHITECTURES 80)  # A100
```

### 메모리 부족
```bash
# 그리드 크기 줄이기
./els_pricer  # 기본 100×100×200으로 테스트

# 또는 큰 VRAM GPU 사용 (V100, A100)
```

### 컴파일 오류
```bash
# 필요한 패키지 설치
sudo apt install -y build-essential cmake cuda-toolkit-12-3

# PATH 설정
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 빠른 시작 스크립트

### AWS 원클릭 테스트
```bash
#!/bin/bash
# aws_gpu_test.sh

# 1. 인스턴스 생성
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g4dn.xlarge \
    --query 'Instances[0].InstanceId' \
    --output text)

# 2. 대기
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# 3. IP 가져오기
IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

# 4. 프로젝트 업로드 및 실행
scp -r els-pricer-cpp ubuntu@$IP:~/
ssh ubuntu@$IP 'cd els-pricer-cpp && mkdir build && cd build && cmake .. && make -j4 && ./els_pricer --compare'

# 5. 종료
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

### GCP 원클릭 테스트
```bash
#!/bin/bash
# gcp_gpu_test.sh

gcloud compute instances create els-gpu-test \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --metadata=install-nvidia-driver=True

# 대기 (드라이버 설치)
sleep 180

# 업로드 및 실행
gcloud compute scp --recurse els-pricer-cpp els-gpu-test:~/ --zone=us-central1-a
gcloud compute ssh els-gpu-test --zone=us-central1-a --command='cd els-pricer-cpp && mkdir build && cd build && cmake .. && make -j4 && ./els_pricer --compare'

# 종료
gcloud compute instances delete els-gpu-test --zone=us-central1-a --quiet
```

---

## 추천 순서

1. **로컬 CPU 테스트** ✅ (완료!)
   - 200×200×1000: 0.677초

2. **Google Colab** ⭐ (가장 추천!)
   - 무료
   - 설정 간단
   - T4 GPU 바로 사용

3. **AWS/GCP Spot/Preemptible**
   - 저렴 (~$0.10-0.20/시간)
   - 프로덕션 테스트

4. **프로덕션 GPU**
   - A100 등 고성능 GPU
   - 대규모 계산

---

## 연락처

프로젝트 경로: `/home/minhoo/els-pricer-cpp`

GPU 테스트 결과를 공유하고 싶으시면 벤치마크 결과를 저장해주세요!
