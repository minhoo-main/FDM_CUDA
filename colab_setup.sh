#!/bin/bash
# Colab setup script for ELS Pricer
set -e

echo "========================================="
echo "  ELS Pricer - Colab Setup"
echo "========================================="

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA detected:"
    nvcc --version | grep "release"
    HAS_CUDA=1
else
    echo "✗ CUDA not found - CPU only mode"
    HAS_CUDA=0
fi

echo ""
echo "Building CPU version..."
g++ -std=c++17 -O3 -march=native -Iinclude \
    -c src/Grid2D.cpp -o src/Grid2D.o

g++ -std=c++17 -O3 -march=native -Iinclude \
    -c src/ELSProduct.cpp -o src/ELSProduct.o

g++ -std=c++17 -O3 -march=native -Iinclude \
    -c src/ADISolver.cpp -o src/ADISolver.o

g++ -std=c++17 -O3 -march=native -Iinclude \
    examples/validate_price_cpu.cpp \
    src/Grid2D.o src/ELSProduct.o src/ADISolver.o \
    -o validate_price_cpu

echo "✓ CPU version built successfully"

if [ $HAS_CUDA -eq 1 ]; then
    echo ""
    echo "Building GPU version..."

    # Compile CUDA kernels
    nvcc -O3 -std=c++17 -Iinclude \
        -c src/cuda/batched_thomas.cu -o src/cuda/batched_thomas.o

    nvcc -O3 -std=c++17 -Iinclude \
        -c src/cuda/CUDAADISolver.cu -o src/cuda/CUDAADISolver.o

    # Link GPU benchmark
    g++ -std=c++17 -O3 -march=native -Iinclude \
        examples/benchmark_gpu.cpp \
        src/Grid2D.o src/ELSProduct.o \
        src/cuda/batched_thomas.o src/cuda/CUDAADISolver.o \
        -o benchmark_gpu \
        -L/usr/local/cuda/lib64 -lcudart -lcublas

    echo "✓ GPU version built successfully"
else
    echo "⊗ Skipping GPU build (CUDA not available)"
fi

echo ""
echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "Run tests:"
echo "  ./validate_price_cpu       # CPU validation"
if [ $HAS_CUDA -eq 1 ]; then
    echo "  ./benchmark_gpu            # GPU benchmark"
fi
echo ""
