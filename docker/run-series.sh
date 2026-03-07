#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Uso: ./run-series.sh <serie> [opciones] [-- argumentos_extra]

Asume la estructura:
  preprocesamiento/data/<serie>/images/

Modos:
  --mode automatic   Ejecuta automatic_reconstructor (por defecto)
  --mode advanced    Reutiliza run-advance.sh para control fino
  --mode shell       Abre una shell dentro del contenedor

Opciones:
  --data-root PATH            Carpeta base de datasets
  --force-context-default     Ejecuta "docker context use default" antes de correr
  --cpu                       Fuerza ejecucion sin GPU
  -h, --help                  Muestra esta ayuda

Ejemplos:
  ./run-series.sh edificio-a
  ./run-series.sh edificio-a --mode advanced -- --max-image-size 4000 --cache-size 16
  ./run-series.sh edificio-a --mode shell
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../../..")
DATA_ROOT="${PROJECT_ROOT}/preprocesamiento/data"
MODE="automatic"
FORCE_CPU=0
FORCE_CONTEXT_DEFAULT=0
SERIES=""
EXTRA_ARGS=()

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --force-context-default)
            FORCE_CONTEXT_DEFAULT=1
            shift
            ;;
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -* )
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [ -z "$SERIES" ]; then
                SERIES="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$SERIES" ]; then
    echo "Error: debes indicar el nombre de la serie."
    echo ""
    usage
    exit 1
fi

case "$MODE" in
    automatic|advanced|shell)
        ;;
    *)
        echo "Error: modo invalido '$MODE'. Usa automatic, advanced o shell."
        exit 1
        ;;
esac

if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: la carpeta de datos '$DATA_ROOT' no existe."
    exit 1
fi

DATA_ROOT=$(realpath "$DATA_ROOT")
SERIES_DIR="${DATA_ROOT}/${SERIES}"
IMAGES_DIR="${SERIES_DIR}/images"

if [ ! -d "$SERIES_DIR" ]; then
    echo "Error: la serie '$SERIES' no existe en '$DATA_ROOT'."
    echo "Crea la carpeta '${SERIES_DIR}/images' y coloca ahi las imagenes."
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Error: no se encontro '${IMAGES_DIR}'."
    echo "La estructura esperada es: preprocesamiento/data/<serie>/images/"
    exit 1
fi

if [ "$FORCE_CONTEXT_DEFAULT" -eq 1 ]; then
    docker context use default
fi

select_colmap_image() {
    if docker image inspect colmap:latest >/dev/null 2>&1; then
        echo "colmap:latest"
    else
        echo "No se encontro la imagen local colmap:latest. Descargando la oficial..." >&2
        docker pull colmap/colmap:latest >/dev/null
        echo "colmap/colmap:latest"
    fi
}

configure_gpu_args() {
    local image="$1"

    GPU_ARGS=()
    USE_GPU=0

    if [ "$FORCE_CPU" -eq 1 ]; then
        return
    fi

    if docker run --rm --gpus all "$image" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS+=(--gpus all)
        GPU_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all)
        GPU_ARGS+=(-e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
        return
    fi

    if docker run --rm --runtime=nvidia "$image" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS+=(--runtime=nvidia)
        GPU_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all)
        GPU_ARGS+=(-e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
    fi
}

run_with_docker_hint() {
    if ! "$@"; then
        echo ""
        echo "La ejecucion en Docker fallo."
        echo "Si usas Docker Desktop, prueba primero: docker context use default"
        exit 1
    fi
}

if [ "$MODE" = "shell" ]; then
    echo "Serie        : $SERIES_DIR"
    echo "Imagenes     : $IMAGES_DIR"
    echo "Modo         : shell"
    echo ""
    echo "Dentro del contenedor puedes correr, por ejemplo:"
    echo "  colmap automatic_reconstructor --image_path ./images --workspace_path ."
    exec "${SCRIPT_DIR}/run.sh" "$SERIES_DIR"
fi

if [ "$MODE" = "advanced" ]; then
    echo "Serie        : $SERIES_DIR"
    echo "Imagenes     : $IMAGES_DIR"
    echo "Modo         : advanced"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        echo "Args extra   : ${EXTRA_ARGS[*]}"
    fi
    exec "${SCRIPT_DIR}/run-advance.sh" "$SERIES_DIR" "${EXTRA_ARGS[@]}"
fi

COLMAP_IMAGE=$(select_colmap_image)
configure_gpu_args "$COLMAP_IMAGE"

NUM_CPUS=$(nproc)
DOCKER_ARGS=(
    --rm
    -v "${SERIES_DIR}:/working"
    -w /working
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
)

if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    DOCKER_ARGS+=("${GPU_ARGS[@]}")
fi

AUTO_ARGS=(
    automatic_reconstructor
    --image_path ./images
    --workspace_path .
    --use_gpu "$USE_GPU"
)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    AUTO_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Serie        : $SERIES_DIR"
echo "Imagenes     : $IMAGES_DIR"
echo "Modo         : automatic"
echo "GPU          : $USE_GPU"
echo "CPUs         : $NUM_CPUS"
echo "Resultados   : $SERIES_DIR"
echo ""

run_with_docker_hint docker run "${DOCKER_ARGS[@]}" "$COLMAP_IMAGE" colmap "${AUTO_ARGS[@]}"
