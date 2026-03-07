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
#   --skip-to STAGE      : Resume from stage (extraction|matching|sparse|dense|fusion)
#   --no-dense           : Skip dense reconstruction
#   --mesher TYPE        : poisson (default) or delaunay
#   --no-single-camera   : Estimate separate intrinsics per image
#   --camera-model MODEL : Camera model (default: SIMPLE_RADIAL)
#   --cpu                : Force CPU mode (no GPU)
#   --dsp                : Enable DSP-SIFT (better features, but CPU-only extraction, 10-30x slower)
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
    echo "       [--max-features 8192] [--max-matches 32768]"
    echo "       [--skip-to extraction|matching|sparse|dense|fusion]"
    echo "       [--no-dense] [--mesher poisson|delaunay]"
    echo "       [--matcher sequential|exhaustive|vocab_tree]"
    echo "       [--no-single-camera] [--camera-model SIMPLE_RADIAL] [--cpu]"
    exit 1
fi

# --- Defaults ---
CACHE_SIZE=16              # GB of VRAM for PatchMatchStereo
MAX_IMAGE_SIZE=3200        # pixels (COLMAP thresholds are tuned for ~3200)
MAX_FEATURES=8192          # SIFT features per image
MAX_MATCHES=32768          # matches per image pair
SKIP_TO=""                 # resume from this stage
RUN_DENSE=1
OVERWRITE=1
MESHER="poisson"
MATCHER="sequential"       # sequential (best for video) | exhaustive | vocab_tree
OVERLAP=20                 # overlap for sequential matcher
SINGLE_CAMERA=1            # all frames share intrinsics (same camera/video)
CAMERA_MODEL="SIMPLE_RADIAL"
FORCE_CPU=0
DSP_SIFT=0                 # DSP-SIFT: better features but forces CPU extraction (10-30x slower)

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
        --no-overwrite)     OVERWRITE=0;          shift   ;;
        --mesher)           MESHER="$2";          shift 2 ;;
        --matcher)          MATCHER="$2";         shift 2 ;;
        --overlap)          OVERLAP="$2";         shift 2 ;;
        --no-single-camera) SINGLE_CAMERA=0;      shift   ;;
        --camera-model)     CAMERA_MODEL="$2";    shift 2 ;;
        --cpu)              FORCE_CPU=1;          shift   ;;
        --dsp)              DSP_SIFT=1;           shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi
if [ ! -d "$HOST_DIR/images" ]; then
    echo "Error: '$HOST_DIR/images' not found. Images must be in an 'images/' subfolder."
    exit 1
fi

# --- Docker Image Selection ---
if docker image inspect colmap:latest >/dev/null 2>&1; then
    COLMAP_IMAGE="colmap:latest"
else
    echo "Pulling official COLMAP image..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

NUM_CPUS=$(nproc)

# --- GPU Detection ---
GPU_ARGS=()
USE_GPU=0

if [ "$FORCE_CPU" -eq 0 ]; then
    if docker run --rm --gpus all "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
    elif docker run --rm --runtime=nvidia "$COLMAP_IMAGE" nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
        USE_GPU=1
    fi
fi

echo "======================================================="
echo " COLMAP Advanced Pipeline"
echo "  Dataset          : $HOST_DIR"
echo "  PatchMatch VRAM  : ${CACHE_SIZE}GB"
echo "  Max image size   : ${MAX_IMAGE_SIZE}px"
echo "  Max features     : ${MAX_FEATURES}"
echo "  Max matches      : ${MAX_MATCHES}"
echo "  CPUs             : $NUM_CPUS"
echo "  GPU              : $USE_GPU"
echo "  Matcher          : $MATCHER"
echo "  Camera model     : $CAMERA_MODEL"
echo "  Single camera    : $SINGLE_CAMERA"
echo "  DSP-SIFT         : $DSP_SIFT"
[ "$MATCHER" = "sequential" ] && echo "  Overlap          : $OVERLAP"
[ -n "$SKIP_TO" ] && echo "  Resuming from    : $SKIP_TO"
[ "$RUN_DENSE" -eq 0 ] && echo "  Dense            : SKIPPED"
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
        bash -c "rm -rf /working/database.db /working/sparse /working/dense && echo 'Removed: database.db  sparse/  dense/'"
fi

# --- Docker base arguments (reused for every stage) ---
DOCKER_BASE=(
    --rm
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOST_DIR}:/working"
    -w /working
)

if [ ${#GPU_ARGS[@]} -gt 0 ]; then
    DOCKER_BASE+=("${GPU_ARGS[@]}")
fi

run_colmap() {
    docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap "$@"
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

    EXTRACT_ARGS=(
        --database_path ./database.db
        --image_path ./images
        --ImageReader.camera_model "$CAMERA_MODEL"
        --ImageReader.single_camera "$SINGLE_CAMERA"
        --SiftExtraction.use_gpu "$USE_GPU"
        --SiftExtraction.max_image_size "${MAX_IMAGE_SIZE}"
        --SiftExtraction.max_num_features "${MAX_FEATURES}"
        --SiftExtraction.first_octave -1
    )

    # domain_size_pooling and estimate_affine_shape produce better features
    # but FORCE CPU-only extraction (10-30x slower). Only enable via flag.
    if [ "${DSP_SIFT:-0}" -eq 1 ]; then
        EXTRACT_ARGS+=(
            --SiftExtraction.domain_size_pooling 1
            --SiftExtraction.estimate_affine_shape 1
        )
        echo "   DSP-SIFT enabled (CPU extraction, slower but better features)"
    fi

    if [ "$USE_GPU" -eq 1 ]; then
        EXTRACT_ARGS+=(--SiftExtraction.gpu_index 0)
    fi

    run_colmap feature_extractor "${EXTRACT_ARGS[@]}"
    echo "Feature extraction done."
fi

# =============================================
# STAGE 2: Feature Matching
# =============================================
if should_run "matching"; then
    echo ""
    echo "[2/5] Feature Matching (${MATCHER})..."
    MATCH_BASE_ARGS=(
        --database_path ./database.db
        --SiftMatching.use_gpu "$USE_GPU"
        --SiftMatching.max_ratio 0.8
        --SiftMatching.max_num_matches "${MAX_MATCHES}"
    )

    if [ "$USE_GPU" -eq 1 ]; then
        MATCH_BASE_ARGS+=(--SiftMatching.gpu_index 0)
    fi

    if [ "$MATCHER" = "sequential" ]; then
        run_colmap sequential_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            --SequentialMatching.overlap "${OVERLAP}" \
            --SequentialMatching.loop_detection 1 \
            --SequentialMatching.loop_detection_num_images 50
    elif [ "$MATCHER" = "vocab_tree" ]; then
        VOCAB_TREE="/working/vocab_tree.bin"
        if [ ! -f "${HOST_DIR}/vocab_tree.bin" ]; then
            echo "vocab_tree_matcher requires a vocabulary tree file."
            echo "Download one from https://demuc.de/colmap/ and place it at:"
            echo "  ${HOST_DIR}/vocab_tree.bin"
            echo ""
            echo "Recommended sizes:"
            echo "  < 1000 images : vocab_tree_flickr100K_words32K.bin"
            echo "  1K-10K images : vocab_tree_flickr100K_words256K.bin"
            echo "  > 10K images  : vocab_tree_flickr100K_words1M.bin"
            exit 1
        fi
        run_colmap vocab_tree_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
            --VocabTreeMatching.num_images 100
    else
        run_colmap exhaustive_matcher "${MATCH_BASE_ARGS[@]}"
    fi
    echo "Feature matching done."
fi

# =============================================
# STAGE 3: Sparse Reconstruction (SfM)
# =============================================
if should_run "sparse"; then
    echo ""
    echo "[3/5] Sparse Reconstruction (SfM)..."
    run_colmap mapper \
        --database_path ./database.db \
        --image_path ./images \
        --output_path ./sparse
    echo "Sparse reconstruction done."
fi

# Pick the best sparse sub-model (largest points3D.bin = most 3D points)
BEST_SPARSE="${HOST_DIR}/sparse/0"
if [ -d "${HOST_DIR}/sparse" ]; then
    CANDIDATE=$(find "${HOST_DIR}/sparse" -maxdepth 2 -name 'points3D.bin' -print0 2>/dev/null \
        | xargs -0 ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
        BEST_SPARSE="$(dirname "$CANDIDATE")"
    fi
fi
BEST_SPARSE_REL="./sparse/$(basename "$BEST_SPARSE")"
echo "   Best sparse model: $BEST_SPARSE_REL"

# =============================================
# STAGE 4: Dense -- PatchMatch Stereo
# =============================================
if [ "$RUN_DENSE" -eq 1 ] && should_run "dense"; then
    echo ""
    echo "[4/5] Dense Reconstruction (PatchMatchStereo)..."
    echo "   VRAM cache: ${CACHE_SIZE}GB  |  Max image: ${MAX_IMAGE_SIZE}px"
    echo "   Using sparse model: $BEST_SPARSE_REL"

    run_colmap image_undistorter \
        --image_path ./images \
        --input_path "$BEST_SPARSE_REL" \
        --output_path ./dense/0 \
        --output_type COLMAP

    DENSE_ARGS=(
        --workspace_path ./dense/0
        --workspace_format COLMAP
        --PatchMatchStereo.max_image_size "${MAX_IMAGE_SIZE}"
        --PatchMatchStereo.cache_size "${CACHE_SIZE}"
        --PatchMatchStereo.num_samples 15
        --PatchMatchStereo.num_iterations 5
        --PatchMatchStereo.geom_consistency 1
    )

    if [ "$USE_GPU" -eq 1 ]; then
        DENSE_ARGS+=(--PatchMatchStereo.gpu_index 0)
    fi

    run_colmap patch_match_stereo "${DENSE_ARGS[@]}"
    echo "PatchMatchStereo done."
fi

# =============================================
# STAGE 5: Stereo Fusion + Meshing
# =============================================
if [ "$RUN_DENSE" -eq 1 ] && should_run "fusion"; then
    echo ""
    echo "[5/5] Stereo Fusion + Meshing (${MESHER})..."

    run_colmap stereo_fusion \
        --workspace_path ./dense/0 \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path ./dense/0/fused.ply

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
fi

echo ""
echo "======================================================="
echo "Pipeline complete!"
echo "   Sparse model : $BEST_SPARSE/"
[ "$RUN_DENSE" -eq 1 ] && echo "   Dense cloud  : $HOST_DIR/dense/0/fused.ply"
[ "$RUN_DENSE" -eq 1 ] && echo "   Mesh          : $HOST_DIR/dense/0/meshed-${MESHER}.ply"
echo "======================================================="
