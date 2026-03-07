#!/bin/bash
set -e

# Step-by-step COLMAP pipeline with full parameter control.
# This provides maximum GPU/VRAM utilization vs automatic_reconstructor.
#
# Usage: ./run-advanced.sh <host_directory> [options]
#   --cache-size GB   : VRAM for PatchMatchStereo in GB (default: 16)
#   --max-image-size N: Max image dimension for dense (default: 3200)
#   --max-features N  : Max SIFT features per image (default: 8192)
#   --max-matches N   : Max SIFT matches per pair (default: 65536)
#   --matcher TYPE    : exhaustive (default) | sequential | vocab_tree
#   --overlap N       : Sequential matcher overlap (default: 20, only for sequential)
#   --skip-to STAGE   : Resume from stage (extraction|matching|sparse|dense|fusion)
#   --no-dense        : Skip dense reconstruction
#   --mesher TYPE     : poisson (default) or delaunay
#
# Example (full run):
#   ./run-advanced.sh ./south-building/
# Example (max quality):
#   ./run-advanced.sh ./south-building/ --cache-size 16 --max-image-size 4000 --max-features 16384
# Example (resume from dense):
#   ./run-advanced.sh ./south-building/ --skip-to dense

if [ $# -eq 0 ]; then
    echo "Usage: $0 <host_directory> [--cache-size 16] [--max-image-size 3200]"
    echo "       [--max-features 8192] [--max-matches 65536]"
    echo "       [--skip-to extraction|matching|sparse|dense|fusion]"
    echo "       [--no-dense] [--mesher poisson|delaunay]"
    exit 1
fi

# --- Defaults ---
CACHE_SIZE=16          # GB of VRAM for PatchMatchStereo \u2014 most impactful param
MAX_IMAGE_SIZE=3200    # pixels; default in COLMAP is 2000
MAX_FEATURES=8192      # SIFT features per image
MAX_MATCHES=65536      # matches per image pair (default: 32768)
SKIP_TO=""             # resume from this stage
RUN_DENSE=1
OVERWRITE=1            # overwrite existing data by default
MESHER="poisson"
MATCHER="exhaustive"   # exhaustive | sequential | vocab_tree
OVERLAP=20             # overlap for sequential matcher

HOST_DIR=$(realpath "$1")
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-size)     CACHE_SIZE="$2";     shift 2 ;;
        --max-image-size) MAX_IMAGE_SIZE="$2"; shift 2 ;;
        --max-features)   MAX_FEATURES="$2";   shift 2 ;;
        --max-matches)    MAX_MATCHES="$2";    shift 2 ;;
        --skip-to)        SKIP_TO="$2";        shift 2 ;;
        --no-dense)       RUN_DENSE=0;         shift   ;;
        --no-overwrite)   OVERWRITE=0;         shift   ;;
        --mesher)         MESHER="$2";         shift 2 ;;
        --matcher)        MATCHER="$2";        shift 2 ;;
        --overlap)        OVERLAP="$2";        shift 2 ;;
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

echo "======================================================="
echo " COLMAP Advanced Pipeline"
echo "  Dataset        : $HOST_DIR"
echo "  PatchMatch VRAM: ${CACHE_SIZE}GB"
echo "  Max image size : ${MAX_IMAGE_SIZE}px"
echo "  Max features   : ${MAX_FEATURES}"
echo "  Max matches    : ${MAX_MATCHES}"
echo "  CPUs           : $NUM_CPUS"
echo "  Matcher        : $MATCHER"
[ "$MATCHER" = "sequential" ] && echo "  Overlap        : $OVERLAP"
[ -n "$SKIP_TO" ] && echo "  Resuming from  : $SKIP_TO"
[ "$RUN_DENSE" -eq 0 ] && echo "  Dense          : SKIPPED"
[ "$OVERWRITE" -eq 1 ] && echo "  Overwrite      : YES (--no-overwrite to skip)"
echo "======================================================="

# --- Overwrite: clean all previous outputs ---
# Use docker to remove files since they may be owned by root (created inside containers)
if [ "$OVERWRITE" -eq 1 ] && [ -z "$SKIP_TO" ]; then
    echo ""
    echo "\U0001f5d1  Cleaning previous outputs..."
    docker run --rm \
        -v "${HOST_DIR}:/working" \
        -w /working \
        "${COLMAP_IMAGE}" \
        bash -c "rm -rf /working/database.db /working/sparse /working/dense && echo 'Removed: database.db  sparse/  dense/'"
fi

# --- Docker base arguments (reused for every stage) ---
DOCKER_BASE=(
    --rm
    --runtime=nvidia
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    --cpus="${NUM_CPUS}"
    --ipc=host
    --shm-size=16g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOST_DIR}:/working"
    -w /working
)

# Helper to run a colmap command inside docker
run_colmap() {
    docker run "${DOCKER_BASE[@]}" "${COLMAP_IMAGE}" colmap "$@"
}

# Helper to skip stages when --skip-to is set
REACHED_STAGE=0
should_run() {
    local stage="$1"
    if [ -z "$SKIP_TO" ]; then
        return 0  # no skip, run everything
    fi
    if [ "$stage" = "$SKIP_TO" ]; then
        REACHED_STAGE=1
    fi
    if [ "$REACHED_STAGE" -eq 1 ]; then
        return 0
    fi
    echo "\u23ed  Skipping stage: $stage"
    return 1
}

mkdir -p "${HOST_DIR}/sparse"
[ "$RUN_DENSE" -eq 1 ] && mkdir -p "${HOST_DIR}/dense/0"

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# STAGE 1: Feature Extraction
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if should_run "extraction"; then
    echo ""
    echo "\u25b6 [1/5] Feature Extraction..."
    run_colmap feature_extractor \
        --database_path ./database.db \
        --image_path ./images \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --FeatureExtraction.use_gpu 1 \
        --FeatureExtraction.gpu_index 0 \
        --FeatureExtraction.num_threads -1 \
        --SiftExtraction.max_image_size "${MAX_IMAGE_SIZE}" \
        --SiftExtraction.max_num_features "${MAX_FEATURES}" \
        --SiftExtraction.first_octave -1
    echo "\u2705 Feature extraction done."
fi

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# STAGE 2: Feature Matching
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if should_run "matching"; then
    echo ""
    echo "\u25b6 [2/5] Feature Matching (${MATCHER})..."
    MATCH_BASE_ARGS=(
        --database_path ./database.db
        --FeatureMatching.use_gpu 1
        --FeatureMatching.gpu_index 0
        --FeatureMatching.num_threads -1
        --FeatureMatching.max_num_matches "${MAX_MATCHES}"
        --SiftMatching.max_ratio 0.8
        --SiftMatching.cross_check 1
    )
    if [ "$MATCHER" = "sequential" ]; then
        run_colmap sequential_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            --SequentialMatching.overlap "${OVERLAP}" \
            --SequentialMatching.loop_detection 1 \
            --SequentialMatching.loop_detection_num_images 50
    elif [ "$MATCHER" = "vocab_tree" ]; then
        run_colmap vocab_tree_matcher \
            "${MATCH_BASE_ARGS[@]}" \
            --VocabTreeMatching.num_images 100
    else
        run_colmap exhaustive_matcher "${MATCH_BASE_ARGS[@]}"
    fi
    echo "\u2705 Feature matching done."
fi

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# STAGE 3: Sparse Reconstruction (SfM)
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if should_run "sparse"; then
    echo ""
    echo "\u25b6 [3/5] Sparse Reconstruction (SfM)..."
    run_colmap mapper \
        --database_path ./database.db \
        --image_path ./images \
        --output_path ./sparse
    echo "\u2705 Sparse reconstruction done."
fi

# Pick the best sparse sub-model (largest points3D.bin = most 3D points)
BEST_SPARSE=$(find "${HOST_DIR}/sparse" -maxdepth 2 -name 'points3D.bin' 2>/dev/null \
    | xargs ls -s 2>/dev/null | sort -n | tail -1 | awk '{print $2}' \
    | xargs dirname 2>/dev/null)
# Fallback to sparse/0 if detection fails
if [ -z "$BEST_SPARSE" ] || [ ! -d "$BEST_SPARSE" ]; then
    BEST_SPARSE="${HOST_DIR}/sparse/0"
fi
BEST_SPARSE_REL="./sparse/$(basename "$BEST_SPARSE")"
echo "   Best sparse model: $BEST_SPARSE_REL"

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# STAGE 4: Dense \u2014 PatchMatch Stereo
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if [ "$RUN_DENSE" -eq 1 ] && should_run "dense"; then
    echo ""
    echo "\u25b6 [4/5] Dense Reconstruction (PatchMatchStereo)..."
    echo "   VRAM cache: ${CACHE_SIZE}GB  |  Max image: ${MAX_IMAGE_SIZE}px"
    echo "   Using sparse model: $BEST_SPARSE_REL"

    # Undistort images first
    run_colmap image_undistorter \
        --image_path ./images \
        --input_path "$BEST_SPARSE_REL" \
        --output_path ./dense/0 \
        --output_type COLMAP

    # PatchMatch stereo \u2014 this is where cache_size matters most
    run_colmap patch_match_stereo \
        --workspace_path ./dense/0 \
        --workspace_format COLMAP \
        --PatchMatchStereo.gpu_index 0 \
        --PatchMatchStereo.max_image_size "${MAX_IMAGE_SIZE}" \
        --PatchMatchStereo.cache_size "${CACHE_SIZE}" \
        --PatchMatchStereo.num_samples 15 \
        --PatchMatchStereo.num_iterations 5 \
        --PatchMatchStereo.geom_consistency 1
    echo "\u2705 PatchMatchStereo done."
fi

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# STAGE 5: Stereo Fusion + Meshing
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if [ "$RUN_DENSE" -eq 1 ] && should_run "fusion"; then
    echo ""
    echo "\u25b6 [5/5] Stereo Fusion + Meshing (${MESHER})..."

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
    echo "\u2705 Fusion and meshing done."
fi

echo ""
echo "======================================================="
echo "\u2705 Pipeline complete!"
echo "   Sparse model : $BEST_SPARSE/"
[ "$RUN_DENSE" -eq 1 ] && echo "   Dense cloud  : $HOST_DIR/dense/0/fused.ply"
[ "$RUN_DENSE" -eq 1 ] && echo "   Mesh          : $HOST_DIR/dense/0/meshed-${MESHER}.ply"
echo "======================================================="
