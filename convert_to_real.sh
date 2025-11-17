#!/bin/bash
# Convert double to Real type across the project

set -e

echo "======================================================================"
echo "Converting project to use Real type (precision.h)"
echo "======================================================================"
echo ""

# Backup first
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r include src examples "$BACKUP_DIR/"
echo "✓ Backup created"
echo ""

# Files to modify
HEADER_FILES=$(find include -name "*.h" -o -name "*.cuh")
SOURCE_FILES=$(find src -name "*.cpp" -o -name "*.cu")
EXAMPLE_FILES=$(find examples -name "*.cpp" 2>/dev/null || true)

ALL_FILES="$HEADER_FILES $SOURCE_FILES $EXAMPLE_FILES"

echo "Files to modify:"
echo "$ALL_FILES" | tr ' ' '\n'
echo ""

read -p "Proceed with conversion? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Adding #include \"precision.h\" to headers..."
echo "----------------------------------------------------------------------"

for file in $HEADER_FILES; do
    # Skip precision.h itself
    if [[ "$file" == *"precision.h"* ]]; then
        continue
    fi

    # Check if already has the include
    if grep -q "#include \"precision.h\"" "$file"; then
        echo "  ✓ $file (already has include)"
        continue
    fi

    # Add include after first #ifndef or at the beginning
    if grep -q "#ifndef" "$file"; then
        # Add after the #define line
        sed -i '/#define.*_H/a #include "precision.h"' "$file"
    else
        # Add at the very beginning
        sed -i '1i #include "precision.h"' "$file"
    fi

    echo "  ✓ $file"
done

echo ""
echo "Step 2: Converting 'double' to 'Real' (keeping function names)..."
echo "----------------------------------------------------------------------"

for file in $ALL_FILES; do
    # Skip if file doesn't exist
    if [[ ! -f "$file" ]]; then
        continue
    fi

    # Complex sed to replace 'double' with 'Real' but preserve:
    # - function names containing 'double' (e.g., computeDouble)
    # - comments
    # Strategy: Only replace 'double' when it's a type (followed by space, *, &, or >)

    sed -i \
        -e 's/\<double\s\+\*/Real */g' \
        -e 's/\<double\s\+&/Real\&/g' \
        -e 's/\<double\s\+\([a-zA-Z_]\)/Real \1/g' \
        -e 's/\<double>/Real>/g' \
        -e 's/\<double,/Real,/g' \
        -e 's/const\s\+double\s\+\*/const Real */g' \
        -e 's/const\s\+double\s\+&/const Real\&/g' \
        -e 's/std::vector<double>/std::vector<Real>/g' \
        -e 's/vector<double>/vector<Real>/g' \
        "$file"

    echo "  ✓ $file"
done

echo ""
echo "Step 3: Fixing CUDA files..."
echo "----------------------------------------------------------------------"

# CUDA files need special handling
CUDA_FILES=$(find src/cuda include -name "*.cu" -o -name "*.cuh" 2>/dev/null || true)

for file in $CUDA_FILES; do
    if [[ ! -f "$file" ]]; then
        continue
    fi

    # Add include to CUDA files if not present
    if ! grep -q "#include \"precision.h\"" "$file"; then
        # Add at the beginning
        sed -i '1i #include "precision.h"' "$file"
    fi

    # Make sure we're using ELSPricer::Real
    sed -i 's/\<Real\>/ELSPricer::Real/g' "$file"

    echo "  ✓ $file"
done

echo ""
echo "Step 4: Verification..."
echo "----------------------------------------------------------------------"

# Count conversions
TOTAL_REAL=$(grep -r "Real " include src examples 2>/dev/null | wc -l)
REMAINING_DOUBLE=$(grep -r "\<double\>" include src examples 2>/dev/null | grep -v "// " | grep -v "/\*" | wc -l || echo "0")

echo "Statistics:"
echo "  Real types found: $TOTAL_REAL"
echo "  Remaining 'double': $REMAINING_DOUBLE"
echo ""

if [ "$REMAINING_DOUBLE" -gt 50 ]; then
    echo "⚠ Warning: Many 'double' instances remain"
    echo "  This might be normal (comments, function names, etc.)"
    echo "  Please review manually."
fi

echo ""
echo "======================================================================"
echo "✓ Conversion complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Build: cd build && cmake .. && make"
echo "  3. Test FP64: (should work as before)"
echo "  4. Switch to FP32: Edit include/precision.h"
echo "  5. Rebuild and test FP32 version"
echo ""
echo "To revert: rm -rf include src examples && mv $BACKUP_DIR/* ."
echo ""
echo "Backup location: $BACKUP_DIR/"
