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
#   --mesh               : Enable mesh generation after stereo_fusion (default: disabled)
#   --mesher TYPE        : Mesher type when --mesh is enabled: poisson (default) or delaunay
#   --no-single-camera   : Estimate separate intrinsics per image
#   --camera-model MODEL : Camera model (default: SIMPLE_RADIAL)
#   --cpu                : Force CPU mode (no GPU)
#   --dsp                : Enable DSP-SIFT (better features, but CPU-only extraction, 10-30x slower)
#   --ba-global-ratio N  : Global BA trigger ratio (default: 1.4; COLMAP default: 1.1 = every 10%)
#                          1.4 = every 40% of new images → ~3x less global BA rounds
#   --ba-max-iter N      : Max iterations per global BA round (default: 30; COLMAP default: 50)
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
    echo "       [--no-single-camera] [--camera-model SIMPLE_RADIAL] [--cpu]"
    exit 1
fi

# --- Defaults ---
CACHE_SIZE=16              # GB of VRAM for PatchMatchStereo
MAX_IMAGE_SIZE=3200        # pixels (COLMAP thresholds are tuned for ~3200)
MAX_FEATURES=8192          # SIFT features per image
MAX_MATCHES=65536          # matches per image pair
SKIP_TO=""                 # resume from this stage
RUN_DENSE=1
RUN_MESH=0
OVERWRITE=1
MESHER="poisson"
MATCHER="exhaustive"       # sequential (best for video) | exhaustive | vocab_tree
OVERLAP=20                 # overlap for sequential matcher (10 sufficient for 2 FPS video; was 20)
SINGLE_CAMERA=1            # all frames share intrinsics (same camera/video)
CAMERA_MODEL="SIMPLE_RADIAL"  # COLMAP camera model (default: SIMPLE_RADIAL, good for most smartphone cameras; use OPENCV for more complex lenses)
FORCE_CPU=0
DSP_SIFT=0                 # DSP-SIFT: better features but forces CPU extraction (10-30x slower)
NUM_CPUS_OVERRIDE=""         # override nproc with --cpus N

# --- Bundle Adjustment (Mapper) ---
# COLMAP default for ba_global_*_ratio is 1.1 (triggers global BA every 10% new images/points).
# With 1800 images this causes ~16 global BA rounds, each failing with dense Cholesky on CPU.
# Fix: use GPU solver (ba_use_gpu) + reduce frequency (ratio 1.1 = every 10% = ~16 rounds).
BA_GLOBAL_FRAMES_RATIO=1.1   # COLMAP default: 1.1
BA_GLOBAL_POINTS_RATIO=1.1   # COLMAP default: 1.1
BA_GLOBAL_MAX_ITER=50        # COLMAP default: 50 — fewer iterations per global BA round
BA_REFINE_INTRINSICS=1       # 0=fija focal/distorsión en single_camera (default), 1=permite refinamiento
FAST_DENSE=0                 # 0=quality dense (default) | 1=fast dense (~4-5x faster)
DEPTH_MIN=0.1                    # -1 = auto from sparse; set >0 as fallback for images with no sparse points
DEPTH_MAX=100.0                 # -1 = auto from sparse; set >0 as fallback
RESUME_DENSE=0               # 1 = skip cleanup+undistorter, go straight to patch_match_stereo
FUSION_USE_CACHE=1          # 1 = stream depth maps in chunks (required when RAM < ~50GB for large datasets)
FUSION_CACHE_SIZE=32         # GB of RAM for fusion streaming cache (only used when FUSION_USE_CACHE=1)

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
        --cpus)             NUM_CPUS_OVERRIDE="$2"; shift 2 ;;
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
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

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

if [ "$FORCE_CPU" -eq 1 ]; then
    echo "GPU          : desactivada (--cpu)"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi no encontrado en el host. Ejecutando en modo CPU." >&2
else
    echo "GPU del host:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do echo "  $line"; done

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
  echo "  BA global ratio  : ${BA_GLOBAL_FRAMES_RATIO} (COLMAP default: 1.1)"
  echo "  BA max iters     : ${BA_GLOBAL_MAX_ITER} (COLMAP default: 50)"
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
        bash -c "rm -rf /working/database.db /working/sparse /working/dense && echo 'Removed: database.db  sparse/  dense/'"
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
        --FeatureExtraction.use_gpu "$USE_GPU"
        --FeatureExtraction.max_image_size "${MAX_IMAGE_SIZE}"
        --SiftExtraction.max_num_features "${MAX_FEATURES}"
        --SiftExtraction.first_octave -1
        --SiftExtraction.num_octaves 4
        --SiftExtraction.octave_resolution 4
        --SiftExtraction.peak_threshold 0.004
        --SiftExtraction.edge_threshold 10
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
        EXTRACT_ARGS+=(--FeatureExtraction.gpu_index 0,0,0)
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
        --FeatureMatching.use_gpu "$USE_GPU"
        --SiftMatching.max_ratio 0.8
        --FeatureMatching.max_num_matches "${MAX_MATCHES}"
        --SiftMatching.cross_check 1
        #--SiftMatching.guided_matching 1
        --TwoViewGeometry.min_num_inliers 20
        #--ExhaustiveMatching.block_size 150
    )

    if [ "$USE_GPU" -eq 1 ]; then
        MATCH_BASE_ARGS+=(--FeatureMatching.gpu_index 0,0)
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
    echo "   BA global ratio  : ${BA_GLOBAL_FRAMES_RATIO}  |  BA max iters: ${BA_GLOBAL_MAX_ITER}"
    [ "$USE_GPU" -eq 1 ] && echo "   BA GPU solver    : ON (--Mapper.ba_use_gpu 1)"

    MAPPER_ARGS=(
        --database_path ./database.db
        --image_path ./images
        --output_path ./sparse
        # --- BA frequency: reduce from default 1.1 to avoid repeated Cholesky failures ---
        --Mapper.ba_global_frames_ratio "${BA_GLOBAL_FRAMES_RATIO}"
        --Mapper.ba_global_points_ratio "${BA_GLOBAL_POINTS_RATIO}"
        --Mapper.ba_global_max_num_iterations "${BA_GLOBAL_MAX_ITER}"
        # --- Prune redundant 3D points before each BA round (reduces problem size) ---
        --Mapper.ba_global_ignore_redundant_points3D 1
        # --- Use all CPU threads for triangulation/registration ---
        --Mapper.num_threads -1
        --Mapper.ba_local_num_images 8
    )

    # GPU Bundle Adjustment: activates CUDA Ceres solver (cuSolver/cuDSS).
    # Avoids the dense CPU Cholesky path that fails with >500 images.
    # Requires Ceres built with USE_CUDA=ON (already done in local Dockerfile).
    if [ "$USE_GPU" -eq 1 ]; then
        MAPPER_ARGS+=(
            --Mapper.ba_use_gpu 1
            --Mapper.ba_refine_principal_point 1
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

    run_colmap mapper "${MAPPER_ARGS[@]}"
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
    echo "   VRAM cache: ${CACHE_SIZE}GB  |  Max image: ${MAX_IMAGE_SIZE}px"
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
            --output_type COLMAP
    fi

    # GEOM_CONSISTENCY tracks whether patch_match_stereo generated consistency_graphs/.
    # stereo_fusion --input_type geometric requires those graphs.
    # write_consistency_graph=1 enables their generation (required for geometric fusion).
    GEOM_CONSISTENCY=1

    if [ "$FAST_DENSE" -eq 1 ]; then
        # Fast mode: fewer samples/iterations + doubled window step for ~4-5x speedup.
        DENSE_ARGS=(
            --workspace_path ./dense/0
            --workspace_format COLMAP
            --PatchMatchStereo.max_image_size "${MAX_IMAGE_SIZE}"
            --PatchMatchStereo.cache_size "${CACHE_SIZE}"
            --PatchMatchStereo.num_samples 7
            --PatchMatchStereo.num_iterations 3
            --PatchMatchStereo.geom_consistency 1
            --PatchMatchStereo.write_consistency_graph 1
            --PatchMatchStereo.filter 1
            --PatchMatchStereo.window_step 2
        )
    else
        # Quality mode (default): full geometric consistency + consistency graphs for fusion.
        DENSE_ARGS=(
            --workspace_path ./dense/0
            --workspace_format COLMAP
            --PatchMatchStereo.max_image_size "${MAX_IMAGE_SIZE}"
            --PatchMatchStereo.cache_size "${CACHE_SIZE}"
            --PatchMatchStereo.num_samples 15
            --PatchMatchStereo.num_iterations 5
            --PatchMatchStereo.geom_consistency 1
            --PatchMatchStereo.write_consistency_graph 1
        )
    fi

    if [ "$USE_GPU" -eq 1 ]; then
        DENSE_ARGS+=(--PatchMatchStereo.gpu_index 0,0,0)  # -1 = use ALL available GPUs
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
    if [ "${GEOM_CONSISTENCY:-1}" -eq 1 ]; then
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
