#!/bin/bash
set -euo pipefail
# Build COLMAP Docker image from the repository root.
# Usage:
#   ./build.sh                     # auto-detect GPU architecture
#   ./build.sh 89                  # explicit compute capability (e.g. 89 for Ada Lovelace)
#   ./build.sh all-major           # compile for all major architectures
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Determine CUDA architecture: argument > auto-detect > fallback to all-major
if [ $# -ge 1 ]; then
    CUDA_ARCH="$1"
    echo "CUDA_ARCHITECTURES : $CUDA_ARCH (from argument)"
else
    # Auto-detect from the first available GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
        if [ -n "$RAW" ] && [ "$RAW" != "N/A" ]; then
            CUDA_ARCH="$RAW"
            echo "CUDA_ARCHITECTURES : $CUDA_ARCH (auto-detected from GPU)"
        else
            CUDA_ARCH="all-major"
            echo "CUDA_ARCHITECTURES : $CUDA_ARCH (could not parse GPU compute capability)"
        fi
    else
        CUDA_ARCH="all-major"
        echo "CUDA_ARCHITECTURES : $CUDA_ARCH (no GPU detected, building for all)"
    fi
fi

echo "Building COLMAP Docker image from: $REPO_ROOT"
echo ""

docker build "$REPO_ROOT" \
    -f "$REPO_ROOT/docker/Dockerfile" \
    -t colmap:latest \
    --build-arg CUDA_ARCHITECTURES="$CUDA_ARCH"
