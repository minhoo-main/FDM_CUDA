# Quick Start Guide

## 5-Minute Setup

### Prerequisites Check

```bash
# Check C++ compiler
g++ --version  # Should be 9.0 or later

# Check CMake
cmake --version  # Should be 3.18 or later

# Check CUDA (optional, for GPU)
nvcc --version  # Should be 11.0 or later
nvidia-smi     # Check GPU availability
```

### Build and Run

```bash
# Navigate to project
cd /home/minhoo/els-pricer-cpp

# Create build directory
mkdir -p build && cd build

# Configure (auto-detects CUDA)
cmake ..

# Build
make -j$(nproc)

# Run benchmark
./els_pricer --compare

# Run tests
./test_els
```

## Expected Output

```
═══════════════════════════════════════════════════════════
   ELS Pricer - C++/CUDA Implementation
   2D FDM ADI Solver with GPU Acceleration
═══════════════════════════════════════════════════════════

✓ GPU Initialized: NVIDIA GeForce RTX 4080

Grid: 100 × 100 × 200

Running CPU solver...
Running GPU solver...

============================================================
Method                  Price         Time (s)        Speedup
------------------------------------------------------------
CPU                   98.1234          2.145           1.00×
GPU (CUDA)            98.1231          0.156          13.75×
============================================================

✓ GPU is 13.8× faster than CPU!
```

## Common Issues

### 1. CUDA Not Found

**Error:**
```
Could NOT find CUDAToolkit
```

**Solution:**
```bash
# Install CUDA toolkit
# Ubuntu:
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

# Then add to PATH:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 2. Build CPU-Only Version

If you don't have a GPU:

```bash
cd build
cmake -DUSE_CUDA=OFF ..
make -j$(nproc)

# Run CPU-only benchmarks
./els_pricer --cpu-only
```

### 3. Architecture Mismatch

**Error:**
```
nvcc fatal: Unsupported gpu architecture 'compute_89'
```

**Solution:**
Edit `CMakeLists.txt` and change:
```cmake
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# For compute capability 8.6 (RTX 30xx):
set(CMAKE_CUDA_ARCHITECTURES 86)

# For compute capability 7.5 (RTX 20xx, T4):
set(CMAKE_CUDA_ARCHITECTURES 75)
```

Then rebuild:
```bash
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

## Next Steps

- **Modify ELS parameters**: Edit `examples/main.cpp`
- **Try different grid sizes**: Change N1, N2, Nt in main.cpp
- **Compare with Python**: Run Python version and compare performance
- **Profile performance**: Use `nvprof ./els_pricer` to analyze GPU kernels

## Performance Tips

### For Best CPU Performance
```bash
# Compile with optimizations
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### For Best GPU Performance
- Use grid sizes that are multiples of 32 (warp size)
- Example: 64×64, 128×128, 192×192 instead of arbitrary sizes
- Larger grids (200×200×500+) show better GPU speedup

## Benchmark Suggested Grid Sizes

```bash
# Quick test (< 1 second)
Grid: 50×50×100

# Standard test (few seconds)
Grid: 100×100×200

# Performance test (10-30 seconds)
Grid: 150×150×300

# Stress test (1-5 minutes)
Grid: 200×200×1000
```

Edit `examples/main.cpp` to change these.

## Comparing with Python

```bash
# C++ version
cd /home/minhoo/els-pricer-cpp/build
time ./els_pricer --compare

# Python version
cd /home/minhoo/els-pricing-gpu-project/source/els-fdm-pricer
time python3 benchmark_gpu.py
```

Expected results:
- C++ 3-5× faster than Python on CPU
- C++ 2-10× faster than Python on GPU (depending on grid size)

## Understanding Results

### Price Interpretation
- **98-105**: Normal range for sample ELS at t=0
- **Price > 100**: Low volatility or high barriers make early redemption likely
- **Price < 95**: High volatility or strict barriers reduce expected payoff

### Speedup Interpretation
- **< 1×**: GPU overhead dominates (small grid)
- **1-5×**: Moderate speedup (medium grid)
- **5-20×**: Good speedup (large grid)
- **> 20×**: Excellent speedup (very large grid)

### When to Use GPU
- Grid size > 100×100×200
- Multiple pricing runs
- Real-time applications

### When to Use CPU
- Grid size < 100×100
- Single pricing calculation
- No GPU available
- Debugging

## Documentation

- Full documentation: `README.md`
- Algorithm details: See original Python project at `/home/minhoo/els-pricing-gpu-project/docs/`
- Code examples: `examples/main.cpp` and `tests/test_pricing.cpp`
