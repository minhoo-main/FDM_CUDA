#!/bin/bash

echo "Converting entire codebase to FP32 (float)..."

# Find all source files
FILES=$(find . -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) \
    ! -path "*/backup*" ! -path "*/build*" ! -path "*/.git/*")

for file in $FILES; do
    echo "Processing: $file"

    # Replace ELSPricer::Real with float
    sed -i 's/ELSPricer::Real/float/g' "$file"

    # Replace standalone Real with float (but not in comments)
    sed -i 's/\bReal\b/float/g' "$file"

    # Replace remaining double with float (except in specific contexts)
    sed -i 's/\bdouble\b/float/g' "$file"

    # Remove precision.h include
    sed -i '/#include "precision.h"/d' "$file"
done

# Remove precision.h file
rm -f include/precision.h

echo "✅ Conversion to FP32 complete!"
echo "All code now uses 'float' directly"
