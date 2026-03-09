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

echo "Detectando GPU NVIDIA..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi no encontrado en el host. Ejecutando en modo CPU."
else
    echo "GPU del host:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do echo "  $line"; done

    if docker run --rm --gpus all "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        echo "GPU Docker   : OK (--gpus all)"
        DOCKER_ARGS+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
    elif docker run --rm --runtime=nvidia "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        echo "GPU Docker   : OK (--runtime=nvidia)"
        DOCKER_ARGS+=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
    else
        echo "ERROR: nvidia-smi funciona en el host pero Docker no puede acceder a la GPU." >&2
        echo "  Verifica que nvidia-container-toolkit este instalado:" >&2
        echo "    sudo apt install nvidia-container-toolkit && sudo systemctl restart docker" >&2
        exit 1
    fi
fi

echo "Starting interactive bash shell..."
docker run "${DOCKER_ARGS[@]}" "${COLMAP_IMAGE}" bash
