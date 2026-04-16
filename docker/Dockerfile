# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl
ARG FETCHCONTENT_FULLY_DISCONNECTED=OFF
ENV CCACHE_DIR=/colmap/build/.ccache
ENV CCACHE_BASEDIR=/colmap

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
# NOTE: libceres-dev is NOT installed from apt because Ubuntu's package lacks
# CUDA/cuSOLVER support. Ceres is built from source below with -DUSE_CUDA=ON.
RUN apt-get update && \
    apt-get install -y \
        ccache cmake ninja-build build-essential wget git \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev openimageio-tools \
        libmetis-dev libgoogle-glog-dev libgflags-dev \
        libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 libqt6svg6-dev \
        libcgal-dev libsuitesparse-dev \
        libcurl4-openssl-dev libssl-dev \
        libmkl-full-dev \
        libabsl-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
RUN mkdir -p /usr/include/opencv4

# ── cuDSS 0.6.0 ────────────────────────────────────────────────────────────
# Requerido por Ceres HEAD para bundle adjustment en GPU (CUDA_SPARSE solver).
# Ceres 2.2.0 del apt NO tiene soporte cuDSS → BA cae a CPU silenciosamente.
# cuDSS provee factorización de Cholesky dispersa en GPU, eliminando los
# "Linear solver failure: Unable to perform dense Cholesky factorization"
# que ocurren cuando BA corre en CPU con >500 imágenes.
ARG CUDSS_VERSION=0.6.0.5
RUN wget -q "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz" \
    && tar xf libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz \
    && cp -r libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/ \
    && cp -r libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/ \
    && ldconfig \
    && rm -rf libcudss-linux-x86_64-*

# ── Ceres HEAD (post-2.2 con cuDSS) ─────────────────────────────────────────
# Commit específico validado con cuDSS 0.6 + CUDA 12.9.
#
# IMPORTANTE — problema de versión:
#   El commit b92ade3e tiene el código cuDSS/CUDA_SPARSE pero CMakeLists.txt
#   todavía declara version 2.2.x. COLMAP's bundle_adjustment_ceres.cc guarda
#   el path CUDA_SPARSE bajo #if CERES_VERSION_MINOR >= 3 (correcto: el enum
#   ceres::CUDA_SPARSE no existe en Ceres 2.2.0 real, cambiar a >=2 causaría
#   errores de compilación). El sed siguiente hace que este commit reporte 2.3.0
#   para que el guard pase sin cambiar el código de COLMAP.
#
# Verificar en el output de cmake: "-- Found cuDSS" debe aparecer.
# Si imprime "cuDSS not found" → el BA caerá a CPU a pesar de ba_use_gpu=1.
ARG CERES_COMMIT=b92ade3e116cce9ba53a7639daed93e11aebda0d
RUN git clone https://github.com/ceres-solver/ceres-solver /tmp/ceres-solver \
    && cd /tmp/ceres-solver \
    && git checkout ${CERES_COMMIT} \
    && git submodule update --init third_party/abseil-cpp \
    # Patch: hacer que este commit reporte version 2.3.0 para que el guard de
    # COLMAP (CERES_VERSION_MINOR >= 3) permita usar CUDA_SPARSE.
    && sed -i 's/set(CERES_VERSION_MINOR 2)/set(CERES_VERSION_MINOR 3)/' CMakeLists.txt \
    && mkdir build && cd build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DUSE_CUDA=ON \
        # Ceres usa find_package(cudss CONFIG). El archivo del redist cuDSS
        # instala cmake configs en lib/cmake/cudss/ → tras cp a lib64/:
        #   /usr/local/cuda/lib64/cmake/cudss/cudss-config.cmake
        # cudss_DIR debe apuntar al directorio que contiene cudss-config.cmake.
        -DCMAKE_PREFIX_PATH="/colmap-install;/usr/local/cuda" \
        -Dcudss_DIR=/usr/local/cuda/lib64/cmake/cudss \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
    # Verificación: "#define CERES_NO_CUDSS" (sin comentario al inicio) → cuDSS disabled.
    # Cuando cuDSS está habilitado, CreateCeresConfig.cmake escribe "// #define CERES_NO_CUDSS".
    # Cuando está disabled, escribe "#define CERES_NO_CUDSS" (sin //).
    # grep -P "^\s*#define" distingue la línea activa del comentario.
    && ! grep -qP "^\s*#define CERES_NO_CUDSS" include/ceres/internal/config.h \
        || { echo "ERROR: Ceres compiled with CERES_NO_CUDSS. GPU bundle adjustment will NOT work."; exit 1; } \
    && ninja install \
    && rm -rf /tmp/ceres-solver

# ── COLMAP con Global Mapper + ALIKED + LightGlue ───────────────────────────
# ONNX Runtime 1.24.1 GPU es descargado automáticamente por cmake vía
# FETCH_ONNX=ON (valor por defecto) durante la fase de configure.
# Requiere CUDA >= 12 para inferencia GPU en ALIKED/LightGlue.
#
# Uso en runtime (los modelos .onnx se descargan la primera vez):
#   colmap feature_extractor \
#     --FeatureExtraction.type=ALIKED_N32 \
#     --AlikedExtraction.max_num_features=2048
#   colmap vocab_tree_matcher \
#     --FeatureMatching.type=ALIKED_LIGHTGLUE
#
# Tipos de extractor: SIFT (default), ALIKED_N16ROT, ALIKED_N32
# Tipos de matcher:   SIFT_BRUTEFORCE (default), SIFT_LIGHTGLUE,
#                     ALIKED_BRUTEFORCE, ALIKED_LIGHTGLUE
COPY . /colmap

RUN cd /colmap \
    && mkdir -p build/.ccache \
    && cd build \
    && cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        # COLMAP también hace find_package(cudss CONFIG) para enlazar contra Ceres con cuDSS.
        # Necesita el mismo CMAKE_PREFIX_PATH y cudss_DIR que usamos para Ceres.
        -DCMAKE_PREFIX_PATH="/colmap-install;/usr/local/cuda" \
        -Dcudss_DIR=/usr/local/cuda/lib64/cmake/cudss \
        -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} \
        -DBLA_VENDOR=Intel10_64lp \
        # FETCH_ONNX=ON (default): cmake descarga ORT 1.24.1 GPU automáticamente.
        # ONNX_ENABLED=ON habilita ALIKED + LightGlue nativos.
        # DOWNLOAD_ENABLED=ON compila las URIs de los modelos .onnx para
        #   descarga automática en el primer uso (requiere curl + openssl).
        -DONNX_ENABLED=ON \
        -DDOWNLOAD_ENABLED=ON \
    && ninja install

# Copiar SuiteSpare libs al directorio de instalación para el runtime.
RUN for lib in cholmod amd camd ccolamd colamd suitesparseconfig spqr; do \
        cp -a /usr/lib/x86_64-linux-gnu/lib${lib}*.so* /colmap-install/lib/ 2>/dev/null || true; \
    done

#
# Stage to export build caches for CI round-tripping via actions/cache.
#
FROM scratch AS cache-export
COPY --from=builder /colmap/build/.ccache/ /.ccache/
COPY --from=builder /colmap/build/_deps/ /_deps/

#
# Docker runtime stage.
#
# Base: cudnn-runtime (no devel) para tener cuDNN disponible en PatchMatch.
# PatchMatchStereo usa kernels CUDA que se benefician de cuDNN en runtime.
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# NOTE: libceres is NOT installed from apt — our custom build with CUDA support
# is copied from the builder stage (included in /colmap-install/).
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 libomp5 libopengl0 libmetis5 \
        libopenimageio2.4t64 libgcc-s1 libgl1 libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt6core6 libqt6gui6 libqt6widgets6 libqt6openglwidgets6 libqt6svg6 \
        libcurl4 libssl3t64 \
        libmkl-locale libmkl-intel-lp64 libmkl-intel-thread libmkl-core \
        python3 python3-pip python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar COLMAP compilado (incluye Ceres con cuDSS + ONNX libs + SuiteSpare)
COPY --from=builder /colmap-install/ /usr/local/

# Copiar cuDSS runtime al directorio CUDA donde el toolkit las espera.
# /usr/local/cuda/lib64/ está en ldconfig vía /etc/ld.so.conf.d/cuda*.conf
# de la imagen base nvidia/cuda. NO usar /usr/local/lib/ — no garantizado.
COPY --from=builder /usr/local/cuda/lib64/libcudss* /usr/local/cuda/lib64/

# Las libs de ONNX Runtime 1.24.1 GPU son instaladas por cmake en /colmap-install/lib/
# (libonnxruntime.so, libonnxruntime_providers_shared.so, libonnxruntime_providers_cuda.so)
# y ya quedan incluidas en el COPY anterior de /colmap-install/.
RUN ldconfig

WORKDIR /workspace
