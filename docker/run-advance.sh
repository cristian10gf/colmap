#!/bin/bash
set -euo pipefail

# Step-by-step COLMAP pipeline with full parameter control.
# This provides maximum GPU/VRAM utilization vs automatic_reconstructor.
#
# Usage: ./run-advance.sh <host_directory> [options]
#   --cache-size GB      : VRAM for PatchMatchStereo in GB (default: 16)
#   --max-image-size N   : Max image dimension (default: 3200)
#   --max-features N     : Max SIFT features per image (default: 8192)
#   --max-matches N      : Max SIFT matches per pair (default: 32768)
#   --matcher TYPE       : sequential (default) | exhaustive | vocab_tree
#   --overlap N          : Sequential matcher overlap (default: 20)
#   --vocab-tree-num-images N : Nº de imágenes candidatas por query en vocab_tree matcher (default: 150).
#                          150 cubre bien interiores con pasillos repetitivos; subir a 200+ para escenas
#                          con muchas zonas similares.
#   --skip-to STAGE      : Resume from stage (extraction|matching|sparse|dense|fusion)
#   --no-dense           : Skip dense reconstruction
#   --mesh               : Enable mesh generation after stereo_fusion (default: disabled)
#   --mesher TYPE        : Mesher type when --mesh is enabled: poisson (default) or delaunay
#   --no-single-camera   : Estimate separate intrinsics per image
#   --camera-model MODEL : Camera model (default: SIMPLE_RADIAL)
#   --cpu                : Force CPU mode (no GPU)
#   --feature-backend TYPE : colmap (default) | roma
#   --matching-backend TYPE: colmap (default) | roma
#   --roma               : Shortcut for --feature-backend roma --matching-backend roma
#   --roma-fallback      : If RoMa backend fails, fallback to native COLMAP backend (default)
#   --no-roma-fallback   : Disable fallback; fail fast if RoMa backend fails
#   --roma-image TAG     : Docker image for RoMa bridge (default: roma-integration:latest)
#   --roma-rebuild       : Rebuild RoMa image before running bridge
#   --roma-setting NAME  : RoMaV2 setting (default: precise)
#   --roma-match-type T  : COLMAP matches_importer type: raw (default) | inliers
#   --roma-max-correspondences N : Requested correspondences sampled by RoMa (default: 5000)
#   --roma-confidence-threshold F: Min RoMa overlap confidence (default: 0.2)
#   --roma-min-correspondences N : Min correspondences to keep a pair (default: 64)
#   --roma-max-pairs N   : Max pair count processed by RoMa (default: 0 = all)
#   --roma-compile       : Enable torch.compile path in RoMaV2
#   --roma-pairs-mode M  : auto (default) | sequential | exhaustive
#   --sift               : Usar SIFT clásico en lugar de ALIKED+LightGlue (más rápido, peor en pasillos/oscuridad)
#   --dsp                : Activar DSP-SIFT (implica --sift; extracción CPU-only, 10-30x más lento)
#   --ba-global-ratio N  : Global BA trigger ratio (default: 1.4; COLMAP default: 1.1 = every 10%)
#                          1.4 = every 40% of new images → ~3x less global BA rounds
#   --ba-max-iter N      : Max iterations per global BA round (default: 30; COLMAP default: 50)
#   --ba-max-refinements N : Global BA refinement loops per trigger (default: 2; COLMAP default: 5)
#                          Each loop = retriangulation + full BA. 2 gives same quality as 5 in
#                          incremental SfM because local BA fills in between triggers. 5 is
#                          ~2.5x slower with negligible quality gain for indoor reconstructions.
#   --ba-gpu-min-images N : Umbral BundleAdjustmentCeres para bundle_adjuster (default: 50)
#   --ba-gpu-max-direct-dense N : Umbral BundleAdjustmentCeres dense GPU para bundle_adjuster (default: 200)
#   --ba-gpu-max-direct-sparse N: Umbral BundleAdjustmentCeres sparse GPU para bundle_adjuster (default: 4000;
#                          auto-sube a 6000 en datasets >=5000 imágenes con GPUs >=20GB VRAM)
#   --refine-intrinsics  : Habilita refinamiento de focal y distorsión en BA incluso con
#                          single_camera=1. Necesario cuando los frames no tienen EXIF
#                          (extraídos de video) y el prior f=1.2×max_dim puede ser incorrecto.
#   --fast-dense         : Fast dense mode: geom_consistency=0, filter=1, samples=7, iters=3, window_step=2
#                          ~4-5x faster than default dense, moderate quality loss
#   --depth-min M        : Fallback min depth [m] for PatchMatchStereo (default: -1 = auto from sparse).
#                          Set explicitly (e.g. 0.1) when some images have no visible sparse points and
#                          patch_match_stereo crashes with "depth_min > 0" check failure.
#   --depth-max M        : Fallback max depth [m] for PatchMatchStereo (default: -1 = auto from sparse).
#                          Set explicitly (e.g. 100.0) together with --depth-min.
#   --resume-dense       : Skip dense cleanup and image_undistorter; go straight to patch_match_stereo.
#                          Use together with --skip-to dense to resume a crashed PatchMatchStereo run
#                          without losing already-computed depth maps.
#   --fusion-use-cache   : Enable StereoFusion streaming cache (use_cache=1). Required when RAM < total
#                          size of all depth+normal maps (~50GB for 3143 images). Prevents OOM.
#   --fusion-cache-size N: RAM (GB) for fusion streaming cache (default: 32). Only used with --fusion-use-cache.
#   --geom-consistency N : PatchMatchStereo geom_consistency (0 or 1, default: 1).
#   --global-mapper      : Usar GlobalMapper (rotation+translation averaging) en lugar del mapper
#                          incremental. Más rápido en datasets grandes y sin deriva acumulada,
#                          pero menos robusto en escenas con poca superposición o sin calibración.
#   --hierarchical-mapper: Usar HierarchicalMapper en lugar del incremental. Divide el dataset en
#                          sub-modelos con overlap, los reconstruye en paralelo y los fusiona.
#                          Recomendado para >1500 imágenes. Si los sub-modelos no fusionan, usa el
#                          mayor como fallback. Incompatible con --global-mapper.
#   --max-consecutive-failures N : Alias histórico para controlar `Mapper.max_reg_trials`
#                          (default: 6; COLMAP default: 3). En COLMAP 3.14 ya no existe
#                          `Mapper.max_num_consecutive_failures`; este valor limita los reintentos
#                          por imagen en el registro incremental.
#   --max-reg-trials N    : Sinónimo explícito de `Mapper.max_reg_trials` (misma semántica).
#   --init-tri-angle DEG : Ángulo mínimo de triangulación para el par inicial (default: 10°; COLMAP: 16°).
#                          10° es un buen balance para pasillos indoor con baseline corta. Valores <4°
#                          producen geometría degenerada. Solo afecta la selección del par de init.
#   --log-level N        : Nivel de verbosidad del log COLMAP (default: 0; 1=info extra; 2=debug completo).
#                          Equivale a glog FLAGS_v; activa mensajes VLOG(N) del código interno.
#                          1 = dos pasadas: fotométrica + geométrica → nube limpia, menos puntos (~6M).
#                          0 = solo pasada fotométrica → más puntos (~15-18M) con más ruido en
#                              superficies uniformes; fusión cae a photometric automáticamente.
#
# Example (full run):
#   ./run-advance.sh ./south-building/
# Example (max quality):
#   ./run-advance.sh ./south-building/ --cache-size 16 --max-image-size 4000 --max-features 16384
# Example (resume from dense):
#   ./run-advance.sh ./south-building/ --skip-to dense
# Example (video frames, sequential matching with loop detection):
#   ./run-advance.sh ./campus-tour/ --matcher sequential --overlap 20

if [ $# -eq 0 ]; then
    echo "Usage: $0 <host_directory> [--cache-size 16] [--max-image-size 3200]"
    echo "       [--max-features 8192] [--max-matches 65536]"
    echo "       [--skip-to extraction|matching|sparse|dense|fusion]"
    echo "       [--no-dense] [--mesh] [--mesher poisson|delaunay]"
    echo "       [--matcher sequential|exhaustive|vocab_tree]"
    echo "       [--feature-backend colmap|roma] [--matching-backend colmap|roma]"
    echo "       [--roma] [--no-roma-fallback] [--roma-image roma-integration:latest]"
    echo "       [--no-single-camera] [--camera-model SIMPLE_RADIAL] [--cpu]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../../../..")"
ROMA_RUN_SH="${PROJECT_ROOT}/preprocesamiento/models/roma/docker/run.sh"

# --- Defaults ---
CACHE_SIZE=32              # GB of VRAM for PatchMatchStereo
MAX_IMAGE_SIZE=3200        # pixels (COLMAP thresholds are tuned for ~3200)
MAX_FEATURES=""         # SIFT features per image
MAX_MATCHES=65536          # matches per image pair
SKIP_TO=""                 # resume from this stage
RUN_DENSE=1
RUN_MESH=0
OVERWRITE=1
MESHER="poisson"
MATCHER="exhaustive"       # sequential | exhaustive | vocab_tree
OVERLAP=20                 # overlap for sequential matcher (10 sufficient for 2 FPS video; was 20)
VOCAB_TREE_NUM_IMAGES=200  # candidatos por query en vocab_tree matcher (150 cubre bien interiores)
SINGLE_CAMERA=1            # all frames share intrinsics (same camera/video)
CAMERA_MODEL="SIMPLE_RADIAL"  # COLMAP camera model (default: SIMPLE_RADIAL, good for most smartphone cameras; use OPENCV for more complex lenses)
FORCE_CPU=0
USE_SIFT=0                   # Usar SIFT clásico en lugar de ALIKED+LightGlue (más rápido, peor en pasillos/oscuridad)
DSP_SIFT=0                 # DSP-SIFT: better features but forces CPU extraction (10-30x slower)
NUM_CPUS_OVERRIDE=""         # override nproc with --cpus N
FEATURE_BACKEND="colmap"   # colmap | roma
MATCHING_BACKEND="colmap"  # colmap | roma
ROMA_FALLBACK=1            # 1=fallback to colmap if RoMa fails
ROMA_IMAGE="roma-integration:latest"
ROMA_REBUILD=0
ROMA_SETTING="precise"
ROMA_MATCH_TYPE="raw"      # raw | inliers
ROMA_MAX_CORRESPONDENCES=5000
ROMA_CONFIDENCE_THRESHOLD=0.2
ROMA_MIN_CORRESPONDENCES=64
ROMA_MAX_PAIRS=0
ROMA_COMPILE=0
ROMA_PAIRS_MODE="auto"     # auto | sequential | exhaustive
ROMA_BRIDGE_RAN=0
ROMA_FEATURES_IMPORTED=0
ROMA_BRIDGE_MATCH_LIST=""

# --- Bundle Adjustment (Mapper) ---
# COLMAP default for ba_global_*_ratio is 1.1 (triggers global BA every 10% new images/points).
# With 1800 images this causes ~16 global BA rounds, each failing with dense Cholesky on CPU.
# Fix: use GPU solver (ba_use_gpu) + reduce frequency (ratio 1.1 = every 10% = ~16 rounds).
BA_GLOBAL_FRAMES_RATIO=1.1   # COLMAP default: 1.1
BA_GLOBAL_POINTS_RATIO=1.1   # COLMAP default: 1.1
BA_GLOBAL_MAX_ITER=50        # COLMAP default: 50 — fewer iterations per global BA round
BA_GLOBAL_MAX_REFINEMENTS=5  # Safety default: avoids global-refinement crash when mapper temporarily drops to <2 registered images.
BA_REFINE_INTRINSICS=1       # 0=fija focal/distorsión en single_camera (default), 1=permite refinamiento
BA_GPU_MIN_NUM_IMAGES=50
BA_GPU_MAX_NUM_IMAGES_DIRECT_DENSE=200
BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE=4000
FAST_DENSE=0                 # 0=quality dense (default) | 1=fast dense (~4-5x faster)
DEPTH_MIN=-1                    # -1 = auto from sparse; set >0 as fallback for images with no sparse points
DEPTH_MAX=-1                 # -1 = auto from sparse; set >0 as fallback
RESUME_DENSE=0               # 1 = skip cleanup+undistorter, go straight to patch_match_stereo
FUSION_USE_CACHE=1          # 1 = stream depth maps in chunks (required when RAM < ~50GB for large datasets)
FUSION_CACHE_SIZE=32         # GB of RAM for fusion streaming cache (only used when FUSION_USE_CACHE=1)
GEOM_CONSISTENCY=1           # 1=dos pasadas (fotométrica+geométrica, nube limpia) | 0=solo fotométrica (más puntos, más ruido)
USE_GLOBAL_MAPPER=0          # 0=incremental mapper (default) | 1=global mapper (rotation+translation averaging)
USE_HIERARCHICAL_MAPPER=0    # 0=incremental (default) | 1=hierarchical mapper (sub-modelos con overlap, luego fusión)
                             #   Mejor para datasets muy grandes (>1500 imágenes); requiere overlap entre zonas.
                             #   Si los sub-modelos no fusionan → se usa el mayor como fallback.
MAPPER_MAX_REG_TRIALS=3            # Reintentos máximos por imagen durante el registro incremental.
                                   # COLMAP default: 3. Aumentar ayuda en transiciones difíciles
                                   # (giros bruscos, escaleras) sin afectar el parseo en COLMAP 3.14.
MAPPER_INIT_MIN_TRI_ANGLE=16      # Ángulo mínimo de triangulación para el par inicial (COLMAP default: 16°).
                                   # 10° es un buen balance para pasillos indoor con baseline corta.
                                   # 8° era demasiado agresivo. Valores <4° producen geometría degenerada.
LOG_LEVEL=2                  # COLMAP log verbosity: 0=normal, 1=info extra, 2=debug completo

# --- Merge quality guard ---
# If merged model quality is much worse than the anchor model, prefer anchor.
MERGE_QUALITY_GUARD=1
MERGE_MAX_REPROJ_ERROR_PX=2.0
MERGE_MAX_REPROJ_RATIO=2.0

HOST_DIR=$(realpath "$1")
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-size)       CACHE_SIZE="$2";      shift 2 ;;
        --max-image-size)   MAX_IMAGE_SIZE="$2";  shift 2 ;;
        --max-features)     MAX_FEATURES="$2";    shift 2 ;;
        --max-matches)      MAX_MATCHES="$2";     shift 2 ;;
        --skip-to)          SKIP_TO="$2";         shift 2 ;;
        --no-dense)         RUN_DENSE=0;          shift   ;;
        --mesh)             RUN_MESH=1;           shift   ;;
        --no-overwrite)     OVERWRITE=0;          shift   ;;
        --mesher)           MESHER="$2";          shift 2 ;;
        --matcher)          MATCHER="$2";         shift 2 ;;
        --overlap)          OVERLAP="$2";         shift 2 ;;
        --no-single-camera) SINGLE_CAMERA=0;      shift   ;;
        --camera-model)     CAMERA_MODEL="$2";    shift 2 ;;
        --cpu)              FORCE_CPU=1;          shift   ;; 
        --ba-max-refinements) BA_GLOBAL_MAX_REFINEMENTS="$2";                         shift 2 ;;
        --ba-gpu-min-images) BA_GPU_MIN_NUM_IMAGES="$2";                              shift 2 ;;
        --ba-gpu-max-direct-dense) BA_GPU_MAX_NUM_IMAGES_DIRECT_DENSE="$2";           shift 2 ;;
        --ba-gpu-max-direct-sparse) BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE="$2";         shift 2 ;;
        --feature-backend)  FEATURE_BACKEND="$2"; shift 2 ;;
        --matching-backend) MATCHING_BACKEND="$2"; shift 2 ;;
        --roma)             FEATURE_BACKEND="roma"; MATCHING_BACKEND="roma"; shift ;;
        --roma-fallback)    ROMA_FALLBACK=1; shift ;;
        --no-roma-fallback) ROMA_FALLBACK=0; shift ;;
        --roma-image)       ROMA_IMAGE="$2"; shift 2 ;;
        --roma-rebuild)     ROMA_REBUILD=1; shift ;;
        --roma-setting)     ROMA_SETTING="$2"; shift 2 ;;
        --roma-match-type)  ROMA_MATCH_TYPE="$2"; shift 2 ;;
        --roma-max-correspondences) ROMA_MAX_CORRESPONDENCES="$2"; shift 2 ;;
        --roma-confidence-threshold) ROMA_CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
        --roma-min-correspondences) ROMA_MIN_CORRESPONDENCES="$2"; shift 2 ;;
        --roma-max-pairs)   ROMA_MAX_PAIRS="$2"; shift 2 ;;
        --roma-compile)     ROMA_COMPILE=1; shift ;;
        --roma-pairs-mode)  ROMA_PAIRS_MODE="$2"; shift 2 ;;
        --cpus)             NUM_CPUS_OVERRIDE="$2"; shift 2 ;;
        --sift)            USE_SIFT=1;           shift   ;;
        --dsp)              DSP_SIFT=1;           shift   ;;
        --ba-global-ratio)  BA_GLOBAL_FRAMES_RATIO="$2"; BA_GLOBAL_POINTS_RATIO="$2"; shift 2 ;;
        --ba-max-iter)      BA_GLOBAL_MAX_ITER="$2";                                  shift 2 ;;
        --refine-intrinsics) BA_REFINE_INTRINSICS=1;                                  shift   ;;
        --fast-dense)       FAST_DENSE=1;                                              shift   ;;
        --depth-min)        DEPTH_MIN="$2";                                            shift 2 ;;
        --depth-max)        DEPTH_MAX="$2";                                            shift 2 ;;
        --resume-dense)     RESUME_DENSE=1;                                            shift   ;;
        --fusion-use-cache) FUSION_USE_CACHE=1;                                        shift   ;;
        --fusion-cache-size) FUSION_CACHE_SIZE="$2";                                   shift 2 ;;
        --geom-consistency) GEOM_CONSISTENCY="$2";                                     shift 2 ;;
        --vocab-tree-num-images) VOCAB_TREE_NUM_IMAGES="$2";                           shift 2 ;;
        --global-mapper)         USE_GLOBAL_MAPPER=1;                                 shift   ;;
        --hierarchical-mapper)   USE_HIERARCHICAL_MAPPER=1;                          shift   ;;
        --max-consecutive-failures|--max-reg-trials) MAPPER_MAX_REG_TRIALS="$2";      shift 2 ;;
        --init-tri-angle)        MAPPER_INIT_MIN_TRI_ANGLE="$2";                     shift 2 ;;
        --log-level)             LOG_LEVEL="$2";                                      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolver MAX_FEATURES si no fue especificado por el usuario.
# ALIKED features son más discriminativos → 4096 es suficiente.
# SIFT necesita más volumen → 16384 compensa descriptores menos discriminativos.
if [ -z "$MAX_FEATURES" ]; then
    if [ "$USE_SIFT" -eq 1 ]; then
        MAX_FEATURES=16384
    else
        MAX_FEATURES=4096
    fi
fi

case "$FEATURE_BACKEND" in
    colmap|roma) ;;
    *)
        echo "Error: --feature-backend must be colmap or roma (got '$FEATURE_BACKEND')."
        exit 1
        ;;
esac

case "$MATCHING_BACKEND" in
    colmap|roma) ;;
    *)
        echo "Error: --matching-backend must be colmap or roma (got '$MATCHING_BACKEND')."
        exit 1
        ;;
esac

if [ "$FEATURE_BACKEND" != "$MATCHING_BACKEND" ]; then
    echo "Error: use the same backend for extraction and matching."
    echo "  Supported combos:"
    echo "    --feature-backend colmap --matching-backend colmap"
    echo "    --feature-backend roma --matching-backend roma"
    exit 1
fi

case "$ROMA_MATCH_TYPE" in
    raw|inliers) ;;
    *)
        echo "Error: --roma-match-type must be raw or inliers (got '$ROMA_MATCH_TYPE')."
        exit 1
        ;;
esac

case "$ROMA_PAIRS_MODE" in
    auto|sequential|exhaustive) ;;
    *)
        echo "Error: --roma-pairs-mode must be auto, sequential or exhaustive (got '$ROMA_PAIRS_MODE')."
        exit 1
        ;;
esac

if [ "$FEATURE_BACKEND" = "roma" ] || [ "$MATCHING_BACKEND" = "roma" ]; then
    if [ ! -x "$ROMA_RUN_SH" ]; then
        echo "Error: RoMa bridge runner not found or not executable: $ROMA_RUN_SH"
        exit 1
    fi
fi

if [ "$RUN_DENSE" -eq 0 ] && [ "$RUN_MESH" -eq 1 ]; then
    echo "WARN: --mesh ignored because --no-dense was set."
    RUN_MESH=0
fi

if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi
if [ ! -d "$HOST_DIR/images" ]; then
    echo "Error: '$HOST_DIR/images' not found. Images must be in an 'images/' subfolder."
    exit 1
fi

# --- Dataset + memory estimation (based on COLMAP FAQ and current flags) ---
# Source: https://colmap.github.io/faq.html#feature-matching-fails-due-to-illegal-memory-access
# SIFT matching peak formula (bytes): 4*n^2 + 4*n*256, where n=FeatureMatching.max_num_matches
bytes_to_gib() {
    awk -v b="$1" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
}

IMAGE_COUNT=$(find "$HOST_DIR/images" -type f | wc -l | tr -d ' ')
# Avoid SIGPIPE under `set -o pipefail` by not piping `find` into `head`.
FIRST_IMAGE=$(find "$HOST_DIR/images" -type f -print -quit)
IMAGE_WIDTH=0
IMAGE_HEIGHT=0
IMAGE_RESOLUTION="unknown"

if [ -n "$FIRST_IMAGE" ]; then
    IMAGE_META=$(file "$FIRST_IMAGE" 2>/dev/null || true)
    IMAGE_RESOLUTION=$(echo "$IMAGE_META" | grep -oE '[0-9]+x[0-9]+' | tail -n 1 || true)
    if [ -n "$IMAGE_RESOLUTION" ]; then
        IMAGE_WIDTH=${IMAGE_RESOLUTION%x*}
        IMAGE_HEIGHT=${IMAGE_RESOLUTION#*x}
    fi
fi

PIXELS_PER_IMAGE=$((IMAGE_WIDTH * IMAGE_HEIGHT))
RAW_IMAGE_SET_BYTES=$((IMAGE_COUNT * PIXELS_PER_IMAGE * 3))
SIFT_MATCH_VRAM_BYTES=$((4 * MAX_MATCHES * MAX_MATCHES + 4 * MAX_MATCHES * 256))

# Dense maps memory approximation at PatchMatchStereo.max_image_size=1200:
# depth (float32) + normal xyz (3*float32) ~= 16 bytes/pixel.
DENSE_SCALE=$(awk -v w="$IMAGE_WIDTH" -v h="$IMAGE_HEIGHT" 'BEGIN { m = (w > h ? w : h); if (m > 0) { s = 1200.0 / m; if (s > 1) s = 1; printf "%.6f", s } else { printf "0" } }')
DENSE_PIXELS_PER_IMAGE=$(awk -v p="$PIXELS_PER_IMAGE" -v s="$DENSE_SCALE" 'BEGIN { printf "%.0f", p * s * s }')
DEPTH_NORMAL_SET_BYTES=$(awk -v n="$IMAGE_COUNT" -v p="$DENSE_PIXELS_PER_IMAGE" 'BEGIN { printf "%.0f", n * p * 16 }')

# Direct dense BA upper bound (camera Schur only, values-only, no factorization overhead):
# Matrix size (6N x 6N), doubles => 8*(6N)^2 bytes.
BA_DENSE_VALUES_BYTES=$(awk -v n="$BA_GPU_MAX_NUM_IMAGES_DIRECT_DENSE" 'BEGIN { printf "%.0f", 8 * (6 * n) * (6 * n) }')
BA_DENSE_VALUES_DATASET_BYTES=$(awk -v n="$IMAGE_COUNT" 'BEGIN { printf "%.0f", 8 * (6 * n) * (6 * n) }')

# Sparse BA rough budget (heuristic): values-only sparse blocks * fill-in factor.
# Assume avg covisibility ~30, 6x6 blocks (288 bytes), fill-in factor 20x for safety.
BA_SPARSE_EST_BYTES=$(awk -v n="$BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE" 'BEGIN {
    avg_k = 30.0;
    block_bytes = 288.0;
    nnz_blocks = n * (1.0 + avg_k / 2.0);
    values_bytes = nnz_blocks * block_bytes;
    printf "%.0f", values_bytes * 20.0;
}')

# --- Docker Image Selection ---
if docker image inspect colmap:latest >/dev/null 2>&1; then
    COLMAP_IMAGE="colmap:latest"
else
    echo "Pulling official COLMAP image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

if [ -n "$NUM_CPUS_OVERRIDE" ]; then
    NUM_CPUS="$NUM_CPUS_OVERRIDE"
else
    NUM_CPUS=$(nproc)
fi

# --- GPU Detection ---
GPU_ARGS=()
USE_GPU=0
GPU_TOTAL_MIB=0

if [ "$FORCE_CPU" -eq 1 ]; then
    echo "GPU          : desactivada (--cpu)"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi no encontrado en el host. Ejecutando en modo CPU." >&2
else
    echo "GPU del host:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do echo "  $line"; done
    GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

    if docker run --rm --gpus all "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
        USE_GPU=1
        echo "GPU Docker   : OK (--gpus all)"
    elif docker run --rm --runtime=nvidia "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
        USE_GPU=1
        echo "GPU Docker   : OK (--runtime=nvidia)"
    else
        echo "ERROR: nvidia-smi funciona en el host pero Docker no puede acceder a la GPU." >&2
        echo "  Verifica que nvidia-container-toolkit este instalado:" >&2
        echo "    sudo apt install nvidia-container-toolkit && sudo systemctl restart docker" >&2
        exit 1
    fi
fi

# Adaptive BA GPU thresholds for very large reconstructions.
# For 5000x1080p with 20GB VRAM, keeping direct sparse up to ~6000 images is usually safe.
AUTO_BA_TUNED=0
if [ "$USE_GPU" -eq 1 ] && [ "$IMAGE_COUNT" -ge 5000 ] && [ "$GPU_TOTAL_MIB" -ge 20000 ]; then
    if [ "$BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE" -lt 6000 ]; then
        BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE=6000
        AUTO_BA_TUNED=1
    fi
fi

echo "======================================================="
echo " COLMAP Advanced Pipeline"
echo "  Dataset          : $HOST_DIR"
echo "  Images           : ${IMAGE_COUNT} (${IMAGE_RESOLUTION})"
echo "  PatchMatch VRAM  : ${CACHE_SIZE}GB"
echo "  Max image size   : ${MAX_IMAGE_SIZE}px"
echo "  Max features     : ${MAX_FEATURES}"
echo "  Max matches      : ${MAX_MATCHES}"
echo "  Match VRAM peak* : $(bytes_to_gib "$SIFT_MATCH_VRAM_BYTES") GiB (SIFT formula FAQ, n=${MAX_MATCHES})"
echo "  Raw image set    : $(bytes_to_gib "$RAW_IMAGE_SET_BYTES") GiB (RGB 8-bit, sin pirámides)"
echo "  Depth+normal set : $(bytes_to_gib "$DEPTH_NORMAL_SET_BYTES") GiB (@1200px, 16 B/px)"
echo "  BA dense @dataset: $(bytes_to_gib "$BA_DENSE_VALUES_DATASET_BYTES") GiB (valores-only @N=${IMAGE_COUNT})"
echo "  BA dense upper*  : $(bytes_to_gib "$BA_DENSE_VALUES_BYTES") GiB (valores-only @N=${BA_GPU_MAX_NUM_IMAGES_DIRECT_DENSE})"
echo "  BA sparse est.*  : $(bytes_to_gib "$BA_SPARSE_EST_BYTES") GiB (heurístico con fill-in)"
echo "  CPUs             : $NUM_CPUS"
echo "  GPU              : $USE_GPU"
if [ "$USE_GPU" -eq 1 ]; then
    echo "  GPU VRAM total   : ${GPU_TOTAL_MIB} MiB"
fi
echo "  Feature backend  : $FEATURE_BACKEND"
echo "  Matching backend : $MATCHING_BACKEND"
if [ "$FEATURE_BACKEND" = "roma" ] || [ "$MATCHING_BACKEND" = "roma" ]; then
    echo "  RoMa fallback    : $ROMA_FALLBACK"
    echo "  RoMa image       : $ROMA_IMAGE"
    echo "  RoMa setting     : $ROMA_SETTING"
    echo "  RoMa match type  : $ROMA_MATCH_TYPE"
fi
echo "  Matcher          : $MATCHER"
echo "  Camera model     : $CAMERA_MODEL"
echo "  Single camera    : $SINGLE_CAMERA"
if [ "$USE_SIFT" -eq 1 ]; then
    if [ "$DSP_SIFT" -eq 1 ]; then
      echo "  Feature type     : DSP-SIFT (CPU-only, affine shape + domain size pooling)"
    else
      echo "  Feature type     : SIFT + SIFT_BRUTEFORCE"
    fi
  else
    echo "  Feature type     : ALIKED_N32 + ALIKED_LIGHTGLUE"
  fi
[ "$MATCHER" = "sequential" ] && echo "  Overlap          : $OVERLAP"
  echo "  BA global ratio  : ${BA_GLOBAL_FRAMES_RATIO} (COLMAP default: 1.1)"
  echo "  BA max iters     : ${BA_GLOBAL_MAX_ITER} (COLMAP default: 50)"
    echo "  BA refinements   : ${BA_GLOBAL_MAX_REFINEMENTS} (higher = más retriangulación + más riesgo de inestabilidad)"
  if [ "$USE_GLOBAL_MAPPER" -eq 1 ]; then
    echo "  Mapper           : GlobalMapper (rotation+translation averaging)"
  elif [ "$USE_HIERARCHICAL_MAPPER" -eq 1 ]; then
    echo "  Mapper           : HierarchicalMapper (sub-modelos con overlap, fusión iterativa)"
  else
    echo "  Mapper           : incremental (default)"
  fi
    echo "  BA GPU thresholds: min=${BA_GPU_MIN_NUM_IMAGES} dense<=${BA_GPU_MAX_NUM_IMAGES_DIRECT_DENSE} sparse<=${BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE}"
    [ "$AUTO_BA_TUNED" -eq 1 ] && echo "  BA auto-tune     : sparse threshold subido a ${BA_GPU_MAX_NUM_IMAGES_DIRECT_SPARSE} (dataset grande + VRAM suficiente)"
  echo "  Init tri angle   : ${MAPPER_INIT_MIN_TRI_ANGLE}° (indoor recomendado: 8-12°)"
    echo "  Max reg trials   : ${MAPPER_MAX_REG_TRIALS} (COLMAP default: 3)"
  echo "  Log level        : ${LOG_LEVEL} (0=normal, 1=info extra, 2=debug)"
    [ "$RUN_MESH" -eq 1 ] && echo "  Mesh generation  : ENABLED (${MESHER})"
    [ "$RUN_MESH" -eq 0 ] && echo "  Mesh generation  : DISABLED (--mesh to enable)"
  [ -n "$SKIP_TO" ] && echo "  Resuming from    : $SKIP_TO"
  [ "$RUN_DENSE" -eq 0 ] && echo "  Dense            : SKIPPED"
  [ "$RUN_DENSE" -eq 1 ] && [ "$FAST_DENSE" -eq 1 ] && echo "  Dense mode       : FAST (geom_consistency=0 filter=1 samples=7 iters=3 step=2)"
  [ "$RUN_DENSE" -eq 1 ] && [ "$FAST_DENSE" -eq 0 ] && echo "  Dense mode       : QUALITY (geom_consistency=1 samples=15 iters=5)"
  [ "$RUN_DENSE" -eq 1 ] && [ "$DEPTH_MIN" != "-1" ] && echo "  Depth bounds     : min=${DEPTH_MIN}m max=${DEPTH_MAX}m (fallback for images with no sparse points)"
  [ "$RUN_DENSE" -eq 1 ] && [ "$RESUME_DENSE" -eq 1 ] && echo "  Resume dense     : YES (skipping cleanup + undistorter)"
[ "$OVERWRITE" -eq 1 ] && echo "  Overwrite        : YES (--no-overwrite to skip)"
echo "======================================================="

# --- Overwrite: clean all previous outputs ---
if [ "$OVERWRITE" -eq 1 ] && [ -z "$SKIP_TO" ]; then
    echo ""
    echo "Cleaning previous outputs..."
    docker run --rm \
        -v "${HOST_DIR}:/working" \
        -w /working \
        "${COLMAP_IMAGE}" \
        bash -c "rm -rf /working/database.db /working/database.db-shm /working/database.db-wal /working/sparse /working/dense && echo 'Removed: database.db  database.db-shm  database.db-wal  sparse/  dense/'"
fi

# --- Docker base arguments (reused for every stage) ---
DOCKER_BASE=(
    --rm
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=32g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOST_DIR}:/working"
    -w /working
)

if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    DOCKER_BASE+=("${GPU_ARGS[@]}")
fi

run_colmap() {
    local subcmd="$1"; shift
    docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap "$subcmd" \
        --log_level "${LOG_LEVEL}" \
        "$@"
}

get_model_reproj_error() {
    local model_path="$1"
    docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap model_analyzer \
        --path "$model_path" 2>/dev/null \
        | awk '
            /Mean reprojection error:/ {
                # POSIX-awk compatible parser (mawk does not support match(..., ..., array)).
                line = $0
                sub(/^.*Mean reprojection error:[[:space:]]*/, "", line)
                split(line, parts, /[[:space:]]+/)
                if (parts[1] != "") {
                    print parts[1]
                    exit
                }
            }
        '
}

resolve_roma_pairs_mode() {
    if [ "$ROMA_PAIRS_MODE" != "auto" ]; then
        echo "$ROMA_PAIRS_MODE"
        return
    fi

    if [ "$MATCHER" = "sequential" ]; then
        echo "sequential"
    else
        # vocab_tree has no direct RoMa pairing equivalent; exhaustive gives global coverage.
        echo "exhaustive"
    fi
}

run_roma_bridge() {
    if [ "$ROMA_BRIDGE_RAN" -eq 1 ]; then
        return 0
    fi

    local pairs_mode
    pairs_mode="$(resolve_roma_pairs_mode)"

    local bridge_args=()
    bridge_args+=(--pairs-mode "$pairs_mode")
    bridge_args+=(--setting "$ROMA_SETTING")
    bridge_args+=(--match-type "$ROMA_MATCH_TYPE")
    bridge_args+=(--max-correspondences "$ROMA_MAX_CORRESPONDENCES")
    bridge_args+=(--confidence-threshold "$ROMA_CONFIDENCE_THRESHOLD")
    bridge_args+=(--min-correspondences "$ROMA_MIN_CORRESPONDENCES")

    if [ "$pairs_mode" = "sequential" ]; then
        bridge_args+=(--sequential-overlap "$OVERLAP")
    fi
    if [ "$ROMA_MAX_PAIRS" -gt 0 ]; then
        bridge_args+=(--max-pairs "$ROMA_MAX_PAIRS")
    fi
    if [ "$ROMA_COMPILE" -eq 1 ]; then
        bridge_args+=(--compile)
    else
        bridge_args+=(--no-compile)
    fi

    local roma_args=("$HOST_DIR" --image "$ROMA_IMAGE")
    if [ "$ROMA_REBUILD" -eq 1 ]; then
        roma_args+=(--rebuild)
    fi
    if [ "$FORCE_CPU" -eq 1 ]; then
        roma_args+=(--cpu)
    fi
    roma_args+=(--)
    roma_args+=("${bridge_args[@]}")

    echo "   RoMa bridge      : running (${pairs_mode}, setting=${ROMA_SETTING})"
    if ! "$ROMA_RUN_SH" "${roma_args[@]}"; then
        return 1
    fi

    local match_list_host
    if [ "$ROMA_MATCH_TYPE" = "inliers" ]; then
        match_list_host="${HOST_DIR}/roma/matches_inliers.txt"
        ROMA_BRIDGE_MATCH_LIST="./roma/matches_inliers.txt"
    else
        match_list_host="${HOST_DIR}/roma/matches_raw.txt"
        ROMA_BRIDGE_MATCH_LIST="./roma/matches_raw.txt"
    fi

    if [ ! -d "${HOST_DIR}/roma/features" ]; then
        echo "ERROR: RoMa bridge finished but roma/features is missing." >&2
        return 1
    fi
    if [ ! -s "$match_list_host" ]; then
        echo "ERROR: RoMa bridge finished but match list is missing or empty: $match_list_host" >&2
        return 1
    fi

    ROMA_BRIDGE_RAN=1
    return 0
}

import_roma_features() {
    if [ "$ROMA_FEATURES_IMPORTED" -eq 1 ]; then
        return 0
    fi

    if [ ! -f "${HOST_DIR}/database.db" ]; then
        run_colmap database_creator --database_path ./database.db
    fi

    run_colmap feature_importer \
        --database_path ./database.db \
        --image_path ./images \
        --import_path ./roma/features

    ROMA_FEATURES_IMPORTED=1
    return 0
}

run_colmap_feature_extraction() {
    local EXTRACT_ARGS=(
        --database_path ./database.db
        --image_path ./images
        --ImageReader.camera_model "$CAMERA_MODEL"
        --ImageReader.single_camera "$SINGLE_CAMERA"
        --FeatureExtraction.use_gpu "$USE_GPU"
        --FeatureExtraction.max_image_size "${MAX_IMAGE_SIZE}"
    )

    if [ "$USE_SIFT" -eq 1 ]; then
        # ── SIFT clásico ──────────────────────────────────────────────────
        echo "   Extractor: SIFT (--sift)"
        EXTRACT_ARGS+=(
            --SiftExtraction.max_num_features "${MAX_FEATURES}"
            # first_octave=-1: upsampling 2× antes de la pirámide SIFT.
            # Detecta features más pequeños (marcos de puertas, señales).
            # Usa ~2× más VRAM que first_octave=0.
            --SiftExtraction.first_octave -1
            --SiftExtraction.num_octaves 4
            --SiftExtraction.octave_resolution 4
            # peak_threshold=0.0055: filtra features de bajo contraste en
            # paredes lisas (default COLMAP: 0.0067).
            --SiftExtraction.peak_threshold 0.004
            --SiftExtraction.edge_threshold 10

            
            --SiftExtraction.max_num_orientations 2
            --SiftExtraction.upright 0
        )

        # DSP-SIFT: domain_size_pooling + estimate_affine_shape.
        # CPU-only, ~4-5× más lento. Mejora en texturas repetitivas.
        if [ "${DSP_SIFT:-0}" -eq 1 ]; then
            EXTRACT_ARGS+=(
                --SiftExtraction.domain_size_pooling 1
                --SiftExtraction.estimate_affine_shape 1

                # Parámetros DSP-SIFT recomendados (ajustables según el dataset):
                --SiftExtraction.dsp_num_scales 10
                --SiftExtraction.dsp_min_scale 0.166667
                --SiftExtraction.dsp_max_scale 4
            )
            echo "   DSP-SIFT enabled (CPU-only, ~4-5x slower)"
        fi

        if [ "$USE_GPU" -eq 1 ]; then
            EXTRACT_ARGS+=(--FeatureExtraction.gpu_index 0,0,0,0)
        fi
    else
        # ── ALIKED N32 + LightGlue (default) ──────────────────────────────
        # ALIKED: learned keypoint + descriptor extractor vía ONNX Runtime GPU.
        # Mejor que SIFT en: poca textura, poca luz, cambios bruscos de viewpoint.
        # Los modelos .onnx se descargan automáticamente en la primera ejecución
        # (requiere DOWNLOAD_ENABLED=ON en la compilación — ya habilitado).
        #
        # NOTA: ALIKED no es completamente rotation-invariant como SIFT.
        # COLMAP 4.x lee orientación EXIF y rota las imágenes automáticamente
        # durante extracción. Para video sin EXIF, las imágenes deben estar
        # aproximadamente en orientación vertical (landscape normal).
        echo "   Extractor: ALIKED_N32 + LightGlue (default)"
        EXTRACT_ARGS+=(
            --FeatureExtraction.type ALIKED_N16ROT
            # max_num_features: ALIKED default es 2048, subimos a 4096 para indoor
            # donde más features ayudan con texturas repetitivas y transiciones.
            # No subir a 8192+ — ALIKED features son más discriminativos que SIFT
            # y 4096 cubre mejor que 16384 SIFT features.
            --AlikedExtraction.max_num_features "${MAX_FEATURES}"
        )
        if [ "$USE_GPU" -eq 1 ]; then
            EXTRACT_ARGS+=(--FeatureExtraction.gpu_index 0)
        fi
    fi

    # Tolerate extractor failures (e.g., one corrupt image or transient OOM) and
    # continue if the database still contains a usable amount of keypoints.
    set +e
    run_colmap feature_extractor "${EXTRACT_ARGS[@]}"
    EXTRACT_EXIT=$?
    set -e

    if [ ! -f "${HOST_DIR}/database.db" ]; then
        echo "ERROR: feature_extractor no generó database.db; no se puede continuar." >&2
        exit 1
    fi

    KEYPOINT_ROWS=0
    KEYPOINT_IMAGES_NONZERO=0
    if command -v sqlite3 >/dev/null 2>&1; then
        KEYPOINT_ROWS=$(sqlite3 "${HOST_DIR}/database.db" "SELECT COUNT(*) FROM keypoints;" 2>/dev/null || echo "0")
        KEYPOINT_IMAGES_NONZERO=$(sqlite3 "${HOST_DIR}/database.db" "SELECT COUNT(*) FROM keypoints WHERE rows > 0;" 2>/dev/null || echo "0")
    else
        echo "WARN: sqlite3 no disponible; no se pudo validar keypoints de extracción." >&2
    fi

    if [ "${KEYPOINT_IMAGES_NONZERO:-0}" -le 1 ]; then
        echo "ERROR: extracción insuficiente (${KEYPOINT_IMAGES_NONZERO} imágenes con keypoints > 0)." >&2
        echo "  No se puede continuar a matching/sparse de forma fiable." >&2
        exit 1
    fi

    if [ "$EXTRACT_EXIT" -ne 0 ]; then
        echo "WARN: feature_extractor terminó con código ${EXTRACT_EXIT}." >&2
        echo "  Se continúa porque hay ${KEYPOINT_IMAGES_NONZERO} imágenes con keypoints (filas keypoints=${KEYPOINT_ROWS})."
    else
        echo "Feature extraction done (${KEYPOINT_IMAGES_NONZERO} imágenes con keypoints, filas keypoints=${KEYPOINT_ROWS})."
    fi
}

run_roma_feature_extraction() {
    if ! run_roma_bridge; then
        return 1
    fi
    import_roma_features
}

run_colmap_feature_matching() {
    local MATCH_BASE_ARGS=(
        --database_path ./database.db
        --FeatureMatching.use_gpu "$USE_GPU"
        --FeatureMatching.max_num_matches "${MAX_MATCHES}"
        # min_num_inliers: pares cross-piso/pasillo frecuentemente pasan la verificación geométrica
        # con 15-25 inliers. 35 exige más evidencia para aceptar un par.
        --TwoViewGeometry.min_num_inliers 20

        # Nuevos parametros -----------------------------------
        # RANSAC geométrico: aplica a todos los feature types (SIFT, ALIKED, LightGlue).
        --TwoViewGeometry.confidence 0.999
        #--TwoViewGeometry.max_num_trials 10000
        #--TwoViewGeometry.max_error 2.5
        #--TwoViewGeometry.min_inlier_ratio 0.25
    )

    if [ "$USE_SIFT" -eq 1 ]; then
        # ── SIFT matching params ──────────────────────────────────────────
        MATCH_BASE_ARGS+=(
            --FeatureMatching.guided_matching 1
            # Lowe's ratio test: 0.70 = restrictivo. En escenas repetitivas el 2° mejor
            # match viene de un pasillo/piso diferente (ratio ≈ 0.9-1.0) → rechazado.
            --SiftMatching.max_ratio 0.8
            --SiftMatching.max_distance 0.6
            --SiftMatching.cross_check 1
        )
    else
        # ── ALIKED+LightGlue matching params ──────────────────────────────
        # LightGlue es un matcher basado en attention (no brute force).
        # No usa ratio test ni cross_check — la red neuronal produce matches
        # directamente con score de confianza.
        MATCH_BASE_ARGS+=(
            --FeatureMatching.type ALIKED_LIGHTGLUE
            --FeatureMatching.guided_matching 0
        )
    fi

    if [ "$USE_GPU" -eq 1 ]; then
        MATCH_BASE_ARGS+=(--FeatureMatching.gpu_index 0,0)
    fi

    if [ "$MATCHER" = "sequential" ]; then
        run_colmap sequential_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            --SequentialMatching.overlap "${OVERLAP}" \
            --SequentialMatching.loop_detection 1 \
            --SequentialMatching.loop_detection_num_images 200
    elif [ "$MATCHER" = "vocab_tree" ]; then
        # El vocab tree se descarga automáticamente según el tipo de feature:
        #   ALIKED N32 → vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin
        #   SIFT       → vocab_tree_faiss_flickr100K_words256K.bin
        # COLMAP usa GetVocabTreeUriForFeatureType() cuando vocab_tree_path está vacío
        # (requiere DOWNLOAD_ENABLED=ON, ya compilado en la imagen).
        # Si existe un vocab tree local en la carpeta de la serie, se usa ese en su lugar.
        local vocab_tree_args=(
            --VocabTreeMatching.num_images "${VOCAB_TREE_NUM_IMAGES}"
        )
        if [ -f "${HOST_DIR}/vocab_tree.bin" ]; then
            echo "   Vocab tree       : local (${HOST_DIR}/vocab_tree.bin)"
            vocab_tree_args+=(--VocabTreeMatching.vocab_tree_path /working/vocab_tree.bin)
        else
            echo "   Vocab tree       : auto-descarga según tipo de feature"
        fi
        echo "   Candidatos/query : ${VOCAB_TREE_NUM_IMAGES} imágenes"
        run_colmap vocab_tree_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            "${vocab_tree_args[@]}"
    else
        run_colmap exhaustive_matcher "${MATCH_BASE_ARGS[@]}"
    fi
}

run_roma_feature_matching() {
    if ! run_roma_bridge; then
        return 1
    fi

    # Ensure keypoint indices in the DB match the RoMa-generated match lists.
    import_roma_features

    run_colmap matches_importer \
        --database_path ./database.db \
        --match_list_path "$ROMA_BRIDGE_MATCH_LIST" \
        --match_type "$ROMA_MATCH_TYPE"
}

# Helper to skip stages when --skip-to is set
REACHED_STAGE=0
should_run() {
    local stage="$1"
    if [ -z "$SKIP_TO" ]; then
        return 0
    fi
    if [ "$stage" = "$SKIP_TO" ]; then
        REACHED_STAGE=1
    fi
    if [ "$REACHED_STAGE" -eq 1 ]; then
        return 0
    fi
    echo "  Skipping stage: $stage"
    return 1
}

mkdir -p "${HOST_DIR}/sparse"
[ "$RUN_DENSE" -eq 1 ] && mkdir -p "${HOST_DIR}/dense/0"

# =============================================
# STAGE 1: Feature Extraction
# =============================================
if should_run "extraction"; then
    echo ""
    echo "[1/5] Feature Extraction..."

    if [ "$FEATURE_BACKEND" = "roma" ]; then
        echo "   Backend          : RoMaV2 bridge"
        if run_roma_feature_extraction; then
            echo "   RoMa features imported into database.db"
        elif [ "$ROMA_FALLBACK" -eq 1 ]; then
            echo "WARN: RoMa extraction backend failed, falling back to native COLMAP extractor." >&2
            run_colmap_feature_extraction
        else
            echo "ERROR: RoMa extraction backend failed and fallback is disabled (--no-roma-fallback)." >&2
            exit 1
        fi
    else
        run_colmap_feature_extraction
    fi

    echo "Feature extraction done."
fi

# =============================================
# STAGE 2: Feature Matching
# =============================================
if should_run "matching"; then
    echo ""
    echo "[2/5] Feature Matching (${MATCHER})..."

    if [ "$MATCHING_BACKEND" = "roma" ]; then
        echo "   Backend          : RoMaV2 bridge"
        if run_roma_feature_matching; then
            echo "   RoMa matches imported into database.db"
        elif [ "$ROMA_FALLBACK" -eq 1 ]; then
            echo "WARN: RoMa matching backend failed, falling back to native COLMAP matcher." >&2
            echo "WARN: Re-running COLMAP feature extraction to guarantee matcher compatibility." >&2
            run_colmap_feature_extraction
            run_colmap_feature_matching
        else
            echo "ERROR: RoMa matching backend failed and fallback is disabled (--no-roma-fallback)." >&2
            exit 1
        fi
    else
        run_colmap_feature_matching
    fi

    echo "Feature matching done."
fi

# =============================================
# STAGE 3: Sparse Reconstruction (SfM)
# =============================================
if should_run "sparse"; then
    echo ""
    echo "[3/5] Sparse Reconstruction (SfM)..."
    echo "   BA global ratio  : ${BA_GLOBAL_FRAMES_RATIO}  |  BA max iters: ${BA_GLOBAL_MAX_ITER}"
    [ "$USE_GPU" -eq 1 ] && echo "   BA GPU solver    : ON (--Mapper.ba_use_gpu 1)"

    # --- Pre-flight: verify GPU BA (cuDSS/CUDA_SPARSE) is actually available ---
    # If Ceres was built without cuDSS, ba_use_gpu silently falls back to CPU.
    # That causes 10-18 min per global BA round instead of <30s on GPU.
    if [ "$USE_GPU" -eq 1 ]; then
        CUDSS_CHECK_RAW=$(docker run --rm "${GPU_ARGS[@]}" "${COLMAP_IMAGE}" \
            bash -c 'ldconfig -p 2>/dev/null | grep -c libcudss || true' 2>/dev/null || true)
        # Docker runtime can print extra lines; keep only the last numeric token.
        CUDSS_CHECK=$(printf '%s\n' "${CUDSS_CHECK_RAW}" | grep -Eo '[0-9]+' | tail -n1)
        [ -z "${CUDSS_CHECK}" ] && CUDSS_CHECK=0
        if [ "${CUDSS_CHECK:-0}" -eq 0 ]; then
            echo ""
            echo "ERROR: libcudss not found in the runtime container." >&2
            echo "  GPU bundle adjustment will NOT work (CPU fallback = hours of BA)." >&2
            echo "  Rebuild the Docker image: cd docker && ./build.sh" >&2
            echo "  The fix is in the Dockerfile (cuDSS runtime path and Ceres cmake)." >&2
            exit 1
        fi
        echo "   cuDSS runtime    : OK (libcudss found)"
    fi

    MAPPER_ARGS=(
        --database_path ./database.db
        --image_path ./images
        --output_path ./sparse
        # --- BA frequency ---
        --Mapper.ba_global_frames_ratio "${BA_GLOBAL_FRAMES_RATIO}"
        --Mapper.ba_global_points_ratio "${BA_GLOBAL_POINTS_RATIO}"
        --Mapper.ba_global_max_num_iterations "${BA_GLOBAL_MAX_ITER}"
        --Mapper.ba_global_max_refinements "${BA_GLOBAL_MAX_REFINEMENTS}"
        # --- Prune redundant 3D points before each BA round (reduces problem size) ---
        --Mapper.ba_global_ignore_redundant_points3D 0
        # --- Use all CPU threads for triangulation/registration ---
        --Mapper.num_threads -1

        # Adicionales   --------------------
        # --- BA local: más imágenes e iteraciones para mejor consistencia local ---
        # ba_local_num_images=10: incluye más contexto en BA local (default: 6, antes: 8).
        --Mapper.ba_local_num_images 6
        # ba_local_max_num_iterations=40: más iteraciones por BA local (default: 25).
        # Con GPU solver el coste extra es marginal (~ms) y mejora convergencia.
        --Mapper.ba_local_max_num_iterations 30
        # ba_local_max_refinements=3: loops de retriangulación+BA local (default: 2).
        # Cada loop refina observaciones. 3 da mejor calidad en transiciones difíciles.
        --Mapper.ba_local_max_refinements 3
        # --- Perfil mapper relajado para escenas indoor con baja textura ---
        # Menos estricto que antes para priorizar conectividad y reducir fragmentación.
        --Mapper.min_num_matches 15
        --Mapper.abs_pose_min_num_inliers 35
        --Mapper.abs_pose_min_inlier_ratio 0.25
        --Mapper.abs_pose_max_error 10
        # --- Inicialización robusta ---
        --Mapper.init_min_num_inliers 100
        --Mapper.init_min_tri_angle "${MAPPER_INIT_MIN_TRI_ANGLE}"
        # --- Robustez en transiciones difíciles (giros, escaleras) ---
        --Mapper.max_reg_trials "${MAPPER_MAX_REG_TRIALS}"
        # --- Post-triangulación: filtrado más estricto de observaciones ---
        --Mapper.filter_max_reproj_error 3.5
        --Mapper.filter_min_tri_angle 2.0
    )

    # GPU Bundle Adjustment: CUDA_SPARSE solver via cuDSS (sparse Cholesky on GPU).
    # Requires Ceres 2.3+ built with USE_CUDA=ON + cuDSS found (confirmed pre-flight above).
    # Eliminates the "dense Cholesky factorization" failures seen with CPU fallback.
    #
    # Nota sobre BundleAdjustmentCeres.*:
    # En esta build (COLMAP 4.1.0.dev0), mapper/hierarchical_mapper/global_mapper
    # no exponen flags BundleAdjustmentCeres.* en CLI; solo bundle_adjuster los acepta.
    # Por eso aquí usamos --Mapper.ba_use_gpu y dejamos los umbrales como referencia
    # para una pasada explícita de bundle_adjuster.
    if [ "$USE_GPU" -eq 1 ]; then
        MAPPER_ARGS+=(
            --Mapper.ba_use_gpu 1
            --Mapper.ba_gpu_index 0
            --Mapper.ba_refine_principal_point 0
            --Mapper.ba_min_num_residuals_for_cpu_multi_threading 50000
        )
    fi

    # Con single_camera=1, todos los frames comparten los mismos intrínsecos.
    # Si los frames NO tienen EXIF (extraídos de video), el prior de focal es
    # f=1.2×max_dim (estimación). En ese caso usar --refine-intrinsics para
    # permitir que COLMAP auto-calibre la focal durante el BA.
    # Si hay EXIF o se proveen --ImageReader.camera_params, se puede fijar (más rápido).
    if [ "$SINGLE_CAMERA" -eq 1 ]; then
        if [ "$BA_REFINE_INTRINSICS" -eq 0 ]; then
            MAPPER_ARGS+=(
                --Mapper.ba_refine_focal_length 0
                --Mapper.ba_refine_extra_params 0
            )
            echo "   Intrinsics       : fijas en BA (single camera, sin --refine-intrinsics)"
        else
            echo "   Intrinsics       : refinables en BA (--refine-intrinsics activo)"
        fi
    else
        echo "   Intrinsics       : por imagen (--no-single-camera, COLMAP refina cada cámara)"
    fi

    if [ "$USE_GLOBAL_MAPPER" -eq 1 ]; then
        GLOBAL_MAPPER_ARGS=(
            --database_path ./database.db
            --image_path ./images
            --output_path ./sparse
            --GlobalMapper.num_threads -1
        )
        if [ "$USE_GPU" -eq 1 ]; then
            GLOBAL_MAPPER_ARGS+=(
                --GlobalMapper.ba_ceres_use_gpu 1
            )
        fi
        echo "   Mapper           : GlobalMapper (rotation+translation averaging)"
        run_colmap global_mapper "${GLOBAL_MAPPER_ARGS[@]}"
    elif [ "$USE_HIERARCHICAL_MAPPER" -eq 1 ]; then
        # HierarchicalMapper: divide el dataset en sub-modelos con overlap, los reconstruye
        # y los fusiona. Recomendado para >1500 imágenes.
        # NAMESPACE: usa --Mapper.* (mismo que el incremental). Aunque la clase interna fue
        # renombrada a IncrementalPipelineOptions en COLMAP 3.11+, la CLI sigue exponiendo
        # Mapper.* para ambos mappers. IncrementalPipeline.* NO es un prefijo CLI válido.
        HIERARCHICAL_MAPPER_ARGS=(
            --database_path ./database.db
            --image_path ./images
            --output_path ./sparse
            --Mapper.ba_global_frames_ratio "${BA_GLOBAL_FRAMES_RATIO}"
            --Mapper.ba_global_points_ratio "${BA_GLOBAL_POINTS_RATIO}"
            --Mapper.ba_global_max_num_iterations "${BA_GLOBAL_MAX_ITER}"
            --Mapper.ba_global_max_refinements "${BA_GLOBAL_MAX_REFINEMENTS}"
            --Mapper.ba_global_ignore_redundant_points3D 1
            --Mapper.num_threads -1
            # --- BA local (mismos parámetros que incremental) ---
            --Mapper.ba_local_num_images 10
            --Mapper.ba_local_max_num_iterations 40
            --Mapper.ba_local_max_refinements 3
            # --- Perfil mapper relajado para mejorar conectividad ---
            --Mapper.min_num_matches 15
            --Mapper.abs_pose_min_num_inliers 25
            --Mapper.abs_pose_min_inlier_ratio 0.25
            --Mapper.abs_pose_max_error 12
            --Mapper.init_min_num_inliers 100
            --Mapper.init_min_tri_angle "${MAPPER_INIT_MIN_TRI_ANGLE}"
            --Mapper.max_reg_trials "${MAPPER_MAX_REG_TRIALS}"
            # --- Post-triangulación ---
            --Mapper.filter_max_reproj_error 4.0
            --Mapper.filter_min_tri_angle 1.5
        )
        if [ "$USE_GPU" -eq 1 ]; then
            HIERARCHICAL_MAPPER_ARGS+=(
                --Mapper.ba_use_gpu 1
                --Mapper.ba_refine_principal_point 1
            )
        fi
        echo "   Mapper           : HierarchicalMapper (sub-modelos con overlap)"
        run_colmap hierarchical_mapper "${HIERARCHICAL_MAPPER_ARGS[@]}"
    else
        run_colmap mapper "${MAPPER_ARGS[@]}"
    fi
    echo "Sparse reconstruction done."
fi

# Merge all sparse sub-models into one (if multiple were generated by mapper).
# model_merger aligns sub-models via shared image observations in the database.
# Iterative: start from the largest model (anchor), merge smaller ones in descending order.
BEST_SPARSE="${HOST_DIR}/sparse/0"
if [ -d "${HOST_DIR}/sparse" ]; then
    # Sub-models are numbered directories (0, 1, 2, ...)
    NUM_MODELS=$(find "${HOST_DIR}/sparse" -maxdepth 1 -mindepth 1 -type d | grep -cE '/[0-9]+$' || true)

    if [ "${NUM_MODELS:-0}" -gt 1 ]; then
        echo "   ${NUM_MODELS} sub-modelos sparse — merging con model_merger..."

        # Sort sub-models by size of points3D.bin descending (largest = anchor first)
        SORTED_MODELS=$(find "${HOST_DIR}/sparse" -maxdepth 2 -name 'points3D.bin' -print0 2>/dev/null \
            | xargs -0 ls -s 2>/dev/null | sort -rn | awk '{print $2}' | xargs -I{} dirname {} \
            | grep -E '/[0-9]+$')

        ANCHOR=$(echo "$SORTED_MODELS" | head -1)
        ANCHOR_NAME=$(basename "$ANCHOR")
        echo "     Anchor: sparse/${ANCHOR_NAME} (modelo más grande)"

        # Bootstrap merged/ from anchor
        rm -rf "${HOST_DIR}/sparse/merged" "${HOST_DIR}/sparse/merged_tmp"
        mkdir -p "${HOST_DIR}/sparse/merged"
        cp -r "${ANCHOR}/." "${HOST_DIR}/sparse/merged/"

        # Merge remaining sub-models iteratively.
        # IMPORTANT: model_merger writes to output_path even on failure (overwrites with input_path2).
        # Use a temp dir for each attempt; only replace merged/ if the merge succeeded.
        MERGE_COUNT=0
        MERGE_FAIL=0
        while IFS= read -r MODEL; do
            [ "$MODEL" = "$ANCHOR" ] && continue
            MODEL_NAME=$(basename "$MODEL")
            echo "     Merging: sparse/${MODEL_NAME} → sparse/merged"

            rm -rf "${HOST_DIR}/sparse/merged_tmp"
            mkdir -p "${HOST_DIR}/sparse/merged_tmp"

            MERGE_LOG=$(docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap model_merger \
                --input_path1 ./sparse/merged \
                --input_path2 "./sparse/${MODEL_NAME}" \
                --output_path ./sparse/merged_tmp 2>&1)
            echo "$MERGE_LOG" | grep -E "Reconstruction [12]|Merging|Merge (succeeded|failed)"

            if echo "$MERGE_LOG" | grep -q "Merge succeeded"; then
                rm -rf "${HOST_DIR}/sparse/merged"
                mv "${HOST_DIR}/sparse/merged_tmp" "${HOST_DIR}/sparse/merged"
                MERGE_COUNT=$((MERGE_COUNT + 1))
                echo "       => Aceptado"
            else
                rm -rf "${HOST_DIR}/sparse/merged_tmp"
                MERGE_FAIL=$((MERGE_FAIL + 1))
                echo "       => Sin solapamiento, descartado"
            fi
        done <<< "$SORTED_MODELS"

        rm -rf "${HOST_DIR}/sparse/merged_tmp"

        if [ "$MERGE_COUNT" -eq 0 ]; then
            # No merge succeeded — fall back to the single largest sub-model
            echo "   Ningún merge exitoso — usando anchor (${ANCHOR_NAME}) como modelo único"
            BEST_SPARSE="${HOST_DIR}/sparse/${ANCHOR_NAME}"
        else
            echo "   Merge completo: ${MERGE_COUNT} fusionados, ${MERGE_FAIL} descartados → sparse/merged"
            BEST_SPARSE="${HOST_DIR}/sparse/merged"

            # Recomendado por la FAQ de COLMAP: aplicar un BA global tras model_merger
            # para mejorar consistencia geométrica antes de aceptar/descartar el merge.
            echo "   Merge refine     : ejecutando bundle_adjuster global sobre sparse/merged"
            rm -rf "${HOST_DIR}/sparse/merged_refined"
            BA_MERGE_ARGS=(
                --input_path ./sparse/merged
                --output_path ./sparse/merged_refined
            )
            if [ "$USE_GPU" -eq 1 ]; then
                BA_MERGE_ARGS+=(
                    --BundleAdjustmentCeres.use_gpu 1
                    --BundleAdjustmentCeres.gpu_index 0
                )
            fi
            if run_colmap bundle_adjuster "${BA_MERGE_ARGS[@]}"; then
                rm -rf "${HOST_DIR}/sparse/merged"
                mv "${HOST_DIR}/sparse/merged_refined" "${HOST_DIR}/sparse/merged"
                echo "   Merge refine     : OK (sparse/merged actualizado)"
            else
                echo "WARN: bundle_adjuster post-merge falló; se evalúa calidad con sparse/merged sin refinar" >&2
                rm -rf "${HOST_DIR}/sparse/merged_refined"
            fi

            if [ "$MERGE_QUALITY_GUARD" -eq 1 ]; then
                ANCHOR_REPROJ=$(get_model_reproj_error "./sparse/${ANCHOR_NAME}" || true)
                MERGED_REPROJ=$(get_model_reproj_error "./sparse/merged" || true)

                if [ -n "$ANCHOR_REPROJ" ] && [ -n "$MERGED_REPROJ" ]; then
                    echo "   Merge quality    : anchor=${ANCHOR_REPROJ}px merged=${MERGED_REPROJ}px"
                    MERGE_IS_BAD=$(awk -v a="$ANCHOR_REPROJ" -v m="$MERGED_REPROJ" \
                        -v max="$MERGE_MAX_REPROJ_ERROR_PX" -v ratio="$MERGE_MAX_REPROJ_RATIO" '
                        BEGIN {
                            a_ref = (a > 0.2 ? a : 0.2)
                            if (m > max || m > a_ref * ratio) {
                                print 1
                            } else {
                                print 0
                            }
                        }
                    ')

                    if [ "$MERGE_IS_BAD" -eq 1 ]; then
                        echo "   Merge quality    : degradado; usando anchor sparse/${ANCHOR_NAME}"
                        BEST_SPARSE="${HOST_DIR}/sparse/${ANCHOR_NAME}"
                    else
                        echo "   Merge quality    : aceptado"
                    fi
                else
                    echo "WARN: no se pudo evaluar calidad del merge (model_analyzer); se conserva sparse/merged" >&2
                fi
            fi
        fi
    else
        # Single model: pick the one with most 3D points
        CANDIDATE=$(find "${HOST_DIR}/sparse" -maxdepth 2 -name 'points3D.bin' -print0 2>/dev/null \
            | xargs -0 ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
        if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
            BEST_SPARSE="$(dirname "$CANDIDATE")"
        fi
    fi
fi
BEST_SPARSE_REL="./sparse/$(basename "$BEST_SPARSE")"
echo "   Sparse model: $BEST_SPARSE_REL"

# =============================================
# STAGE 4: Dense -- PatchMatch Stereo
# =============================================
if [ "$RUN_DENSE" -eq 1 ] && should_run "dense"; then
    echo ""
    echo "[4/5] Dense Reconstruction (PatchMatchStereo)..."
    echo "   VRAM cache: ${CACHE_SIZE}GB  |  Max image dense: 1200px (SfM: ${MAX_IMAGE_SIZE}px)"
    echo "   Using sparse model: $BEST_SPARSE_REL"

    if [ "$RESUME_DENSE" -eq 1 ]; then
        echo "   Resume mode: skipping cleanup and image_undistorter."
        echo "   Existing depth maps in dense/0/stereo/depth_maps/ will be reused."
    else
        # Re-running dense with existing outputs can make image_undistorter fail
        # (stale files / mixed ownership from previous Docker runs).
        # Keep this cleanup scoped to dense workspace only.
        docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" \
            bash -lc "rm -rf /working/dense/0/images /working/dense/0/sparse /working/dense/0/stereo /working/dense/0/normal_maps /working/dense/0/depth_maps /working/dense/0/consistency_graphs"

        run_colmap image_undistorter \
            --image_path ./images \
            --input_path "$BEST_SPARSE_REL" \
            --output_path ./dense/0 \
            --output_type COLMAP \
            --max_image_size "${MAX_IMAGE_SIZE}"
    fi

    # GEOM_CONSISTENCY controla si patch_match_stereo genera consistency_graphs/.
    # stereo_fusion --input_type geometric requiere esos grafos (write_consistency_graph=1).

    if [ "$FAST_DENSE" -eq 1 ]; then
        # Fast mode: fewer samples/iterations + doubled window step for ~4-5x speedup.
        DENSE_ARGS=(
            --workspace_path ./dense/0
            --workspace_format COLMAP
            --PatchMatchStereo.max_image_size 1200
            --PatchMatchStereo.cache_size "${CACHE_SIZE}"
            --PatchMatchStereo.num_samples 7
            --PatchMatchStereo.num_iterations 3
            --PatchMatchStereo.geom_consistency "${GEOM_CONSISTENCY}"
            --PatchMatchStereo.write_consistency_graph "${GEOM_CONSISTENCY}"
            --PatchMatchStereo.filter 1
            --PatchMatchStereo.window_radius 3
        )
    else
        # Quality mode (default): full geometric consistency + consistency graphs for fusion.
        DENSE_ARGS=(
            --workspace_path ./dense/0
            --workspace_format COLMAP
            --PatchMatchStereo.max_image_size ${MAX_IMAGE_SIZE}
            --PatchMatchStereo.cache_size "${CACHE_SIZE}"
            --PatchMatchStereo.num_samples 15
            --PatchMatchStereo.num_iterations 5
            --PatchMatchStereo.geom_consistency "${GEOM_CONSISTENCY}"
            --PatchMatchStereo.write_consistency_graph "${GEOM_CONSISTENCY}"
            --PatchMatchStereo.filter 1
            --PatchMatchStereo.filter_min_ncc 0.07
            --PatchMatchStereo.window_radius 5
            --PatchMatchStereo.num_threads 20
            --PatchMatchStereo.ncc_sigma 0.60000002384185791
        )
    fi

    if [ "$USE_GPU" -eq 1 ]; then
        # RTX 4000 Ada: 48 SMs, ceil(1200/32)=38 bloques/imagen → óptimo ~5 threads simultáneos.
        # VRAM: 5 × 221 MB datos + 200 MB contexto = ~1.3 GB (de 20 GB disponibles).
        DENSE_ARGS+=(--PatchMatchStereo.gpu_index -1)
    fi

    # Fallback depth bounds: required when some images have no visible sparse points.
    # Without these, PatchMatchStereo crashes with "depth_min > 0 && depth_max > 0" check failure.
    # -1 (default) means auto-estimate from sparse model per image.
    if [ "$DEPTH_MIN" != "-1" ]; then
        DENSE_ARGS+=(--PatchMatchStereo.depth_min "${DEPTH_MIN}")
    fi
    if [ "$DEPTH_MAX" != "-1" ]; then
        DENSE_ARGS+=(--PatchMatchStereo.depth_max "${DEPTH_MAX}")
    fi

    # patch_match_stereo can exit non-zero when some images have no sparse neighbors
    # (depth_min/max estimation fails for isolated images). The depth maps for the rest
    # are still valid. Disable set -e around this command so fusion always runs.
    set +e
    run_colmap patch_match_stereo "${DENSE_ARGS[@]}"
    DENSE_EXIT=$?
    set -e
    if [ "$DENSE_EXIT" -ne 0 ]; then
        echo "WARN: patch_match_stereo salió con código ${DENSE_EXIT} (depth maps parciales, continuando a fusion)"
    else
        echo "PatchMatchStereo done."
    fi
fi

# =============================================
# STAGE 5: Stereo Fusion + Meshing
# =============================================
if [ "$RUN_DENSE" -eq 1 ] && should_run "fusion"; then
    echo ""
    echo "[5/5] Stereo Fusion..."

    # Use geometric fusion only when consistency_graphs/ were generated (geom_consistency=1).
    # If geom_consistency=0 was used, consistency_graphs/ is empty and geometric fusion
    # produces almost no points (~2000); photometric fusion uses depth maps directly.
    if [ "${GEOM_CONSISTENCY}" -eq 1 ]; then
        FUSION_INPUT_TYPE="geometric"
    else
        FUSION_INPUT_TYPE="photometric"
    fi
    echo "   Fusion input     : ${FUSION_INPUT_TYPE}"

    FUSION_ARGS=(
        --workspace_path ./dense/0
        --workspace_format COLMAP
        --input_type "${FUSION_INPUT_TYPE}"
        --output_path ./dense/0/fused.ply
        --StereoFusion.num_threads 12
      #  --StereoFusion.max_image_size ${MAX_IMAGE_SIZE}
    )
    if [ "$FUSION_USE_CACHE" -eq 1 ]; then
        FUSION_ARGS+=(
            --StereoFusion.use_cache 1
            --StereoFusion.cache_size "${FUSION_CACHE_SIZE}"
        )
        echo "   Fusion cache     : ON (${FUSION_CACHE_SIZE}GB RAM streaming)"
    fi

    run_colmap stereo_fusion "${FUSION_ARGS[@]}"

    if [ "$RUN_MESH" -eq 1 ]; then
        echo "   Meshing enabled (${MESHER})..."
        if [ "$MESHER" = "poisson" ]; then
            run_colmap poisson_mesher \
                --input_path ./dense/0/fused.ply \
                --output_path ./dense/0/meshed-poisson.ply
        else
            run_colmap delaunay_mesher \
                --input_path ./dense/0 \
                --input_type dense \
                --output_path ./dense/0/meshed-delaunay.ply
        fi
        echo "Fusion and meshing done."
    else
        echo "Fusion done. Meshing skipped (--mesh to enable)."
    fi
fi

echo ""
echo "======================================================="
echo "Pipeline complete!"
echo "   Sparse model : $BEST_SPARSE/"
[ "$RUN_DENSE" -eq 1 ] && echo "   Dense cloud  : $HOST_DIR/dense/0/fused.ply"
[ "$RUN_DENSE" -eq 1 ] && [ "$RUN_MESH" -eq 1 ] && echo "   Mesh          : $HOST_DIR/dense/0/meshed-${MESHER}.ply"
echo "======================================================="
