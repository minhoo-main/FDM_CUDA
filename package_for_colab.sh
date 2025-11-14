#!/bin/bash
# Package ELS Pricer for Colab

echo "Packaging ELS Pricer for Colab..."

# Create temporary directory
rm -rf /tmp/els-pricer-cpp
mkdir -p /tmp/els-pricer-cpp

# Copy directory structure
mkdir -p /tmp/els-pricer-cpp/{include,src/cuda,examples}

# Copy ALL header files (including .cuh)
cp include/*.h /tmp/els-pricer-cpp/include/ 2>/dev/null || true
cp include/*.cuh /tmp/els-pricer-cpp/include/ 2>/dev/null || true

# Copy source files
cp src/Grid2D.cpp /tmp/els-pricer-cpp/src/
cp src/ELSProduct.cpp /tmp/els-pricer-cpp/src/
cp src/ADISolver.cpp /tmp/els-pricer-cpp/src/

# Copy CUDA files
cp src/cuda/*.cu /tmp/els-pricer-cpp/src/cuda/
cp src/cuda/*.cuh /tmp/els-pricer-cpp/src/cuda/ 2>/dev/null || true

# Copy example files
cp examples/validate_price_cpu.cpp /tmp/els-pricer-cpp/examples/
cp examples/benchmark_gpu.cpp /tmp/els-pricer-cpp/examples/

# Copy test files
cp test_final_validation.cpp /tmp/els-pricer-cpp/ 2>/dev/null || true

# Copy documentation
cp README.md /tmp/els-pricer-cpp/ 2>/dev/null || echo "# ELS Pricer" > /tmp/els-pricer-cpp/README.md
cp BUGFIX_SUMMARY.txt /tmp/els-pricer-cpp/ 2>/dev/null || true
cp BUGFIX_EARLY_REDEMPTION.md /tmp/els-pricer-cpp/ 2>/dev/null || true
cp KI_TRACKING_BUG.md /tmp/els-pricer-cpp/ 2>/dev/null || true
cp COLAB_GUIDE.md /tmp/els-pricer-cpp/ 2>/dev/null || true

# Copy setup scripts
cp colab_setup.sh /tmp/els-pricer-cpp/
cp ELS_Pricer_Colab.ipynb /tmp/els-pricer-cpp/

# Create tarball
cd /tmp
tar -czf els-pricer-colab.tar.gz els-pricer-cpp/

# Move to original directory
mv els-pricer-colab.tar.gz /home/minhoo/els-pricer-cpp/

echo ""
echo "✓ Package created: els-pricer-colab.tar.gz"
ls -lh /home/minhoo/els-pricer-cpp/els-pricer-colab.tar.gz

echo ""
echo "Contents:"
tar -tzf /home/minhoo/els-pricer-cpp/els-pricer-colab.tar.gz | grep -E "\.(cpp|cu|cuh|h|md|txt|ipynb|sh)$" | sort
