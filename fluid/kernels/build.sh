#!/bin/bash
# Build script for Fluid kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "============================================"
echo "Building Fluid Kernels"
echo "  Build type: ${BUILD_TYPE}"
echo "  Build dir: ${BUILD_DIR}"
echo "============================================"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DPYTHON_EXECUTABLE=$(which python3)

# Build
cmake --build . --parallel $(nproc)

# Copy output to ops directory
mkdir -p "${SCRIPT_DIR}/../ops"
cp -v fluid_kernels.so "${SCRIPT_DIR}/../ops/" 2>/dev/null || true

echo "============================================"
echo "Build complete!"
echo "Output: ${SCRIPT_DIR}/../ops/fluid_kernels.so"
echo "============================================"
