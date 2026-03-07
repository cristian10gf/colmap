#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <host_directory>"
    echo "Example: $0 ../dataset/"
    exit 1
fi

if docker image inspect colmap:latest >/dev/null 2>&1; then
    COLMAP_IMAGE="colmap:latest"
else
    echo "Local COLMAP image not found, pulling official image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

HOST_DIR=$(realpath "$1")
if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi
echo "Running COLMAP container with directory: $HOST_DIR"

DOCKER_ARGS=(
    -it --rm
    -v "${HOST_DIR}:/working"
    -w /working
)

echo "Testing for GPU acceleration..."
if docker run --rm --gpus all "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected (--gpus all)."
    DOCKER_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
elif docker run --rm --runtime=nvidia "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected (--runtime=nvidia)."
    DOCKER_ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
else
    echo "GPU not detected. Using CPU mode."
fi

echo "Starting interactive bash shell..."
docker run "${DOCKER_ARGS[@]}" "${COLMAP_IMAGE}" bash
