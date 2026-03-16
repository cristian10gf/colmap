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
# NOTE: libabsl-dev is NOT installed because Ubuntu 24.04 ships abseil 20230802,
# pero Ceres master requiere >= 20240116. Se usa el submodule bundled de Ceres.
RUN apt-get update && \
    apt-get install -y \
        ccache \
        cmake \
        ninja-build \
        build-essential \
        wget \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libopenimageio-dev \
        openimageio-tools \
        libmetis-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qt6-base-dev \
        libqt6opengl6-dev \
        libqt6openglwidgets6 \
        libcgal-dev \
        libsuitesparse-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libmkl-full-dev \
        git

# Fix issue in Ubuntu's openimageio CMake config.
# We don't depend on any of openimageio's OpenCV functionality,
# but it still requires the OpenCV include directory to exist.
RUN mkdir -p /usr/include/opencv4

# cuDSS dev+runtime: requerido para GPU sparse BA (SPARSE_SCHUR en GPU).
# El paquete apt instala:
#   cmake config → /usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss/
#   headers      → /usr/include/libcudss/12/
#
# BUG del packaging Ubuntu: cudss-static-targets.cmake referencia libcudss_static.a
# que no viene incluido en el paquete (solo está la shared lib).
# cudss-config.cmake usa `if(EXISTS ...)` antes de cada include, así que
# simplemente eliminar los archivos de static targets es suficiente — el config
# los saltará automáticamente sin necesidad de parchear ningún cmake file.
RUN apt-get install -y libcudss0-dev-cuda-12 && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss/cudss-static-targets.cmake \
          /usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss/cudss-static-targets-release.cmake

# Build Ceres Solver master (2.3.0) from source with CUDA + cuDSS support.
# Ceres master declara versión 2.3.0 y tiene cuDSS merged desde mayo 2024.
# Habilita tanto Dense GPU BA (DENSE_SCHUR) como Sparse GPU BA (SPARSE_SCHUR).
#
# IMPORTANTE: se usa --recurse-submodules para obtener abseil 20240116+ (bundled
# en third_party/abseil-cpp). Ubuntu 24.04 solo tiene abseil 20230802, demasiado
# antiguo — Ceres master requiere >= 20240116.
#
# cudss_DIR apunta al cmake config del paquete apt (ruta versionada no estándar).
# cudss_INCLUDE_DIR también se pasa explícitamente por el mismo motivo.
RUN git clone --depth 1 --branch master --recurse-submodules \
        https://github.com/ceres-solver/ceres-solver.git /tmp/ceres-solver && \
    cd /tmp/ceres-solver && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DUSE_CUDA=ON \
        -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
        -Dcudss_INCLUDE_DIR=/usr/include/libcudss/12 \
        -DCMAKE_PREFIX_PATH=/colmap-install \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF && \
    ninja install && \
    rm -rf /tmp/ceres-solver

# Copy source into the image.
COPY . /colmap

# Build and install COLMAP, pointing CMake to our custom Ceres with CUDA + cuDSS.
# cudss_DIR y cudss_INCLUDE_DIR se pasan explícitamente porque CeresConfig.cmake
# contiene find_dependency(cudss) que re-ejecuta el cmake config de cuDSS.
# El config de cuDSS usa rutas relativas para encontrar headers — en Ubuntu 24.04
# los headers están en /usr/include/libcudss/12/ (ruta no estándar), así que
# hay que forzar la ruta explícitamente.
RUN cd /colmap && \
    mkdir -p build/.ccache && \
    cd build && \
    cmake .. \
        -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DCMAKE_PREFIX_PATH="/colmap-install;/usr/lib/x86_64-linux-gnu/libcudss/12" \
        -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
        -Dcudss_INCLUDE_DIR=/usr/include/libcudss/12 \
        -DFETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED} \
        -DBLA_VENDOR=Intel10_64lp && \
    ninja install

# Collect SuiteSparse runtime libraries needed by Ceres/COLMAP into the install tree.
RUN for lib in cholmod amd camd ccolamd colamd suitesparseconfig spqr cxsparse btf klu umfpack; do \
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
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
# NOTE: libceres is NOT installed from apt — our custom build with CUDA support
# is copied from the builder stage (included in /colmap-install/).
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 \
        libc6 \
        libomp5 \
        libopengl0 \
        libmetis5 \
        libopenimageio2.4t64 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v6t64 \
        libqt6core6 \
        libqt6gui6 \
        libqt6widgets6 \
        libqt6openglwidgets6 \
        libcurl4 \
        libssl3t64 \
        libmkl-locale \
        libmkl-intel-lp64 \
        libmkl-intel-thread \
        libmkl-core \
        libcudss0-cuda-12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files from /colmap-install/ in the builder stage to /usr/local/ in
# the runtime stage. This includes COLMAP binaries/libs, our custom Ceres with
# CUDA/cuSOLVER support, and the SuiteSparse runtime libraries.
COPY --from=builder /colmap-install/ /usr/local/

# Register newly copied shared libraries.
# cuDSS se instala en una ruta versionada no estándar en Ubuntu — hay que añadirla
# explícitamente a ldconfig para que el dynamic linker encuentre libcudss.so.0.
RUN echo /usr/lib/x86_64-linux-gnu/libcudss/12 > /etc/ld.so.conf.d/cudss.conf && \
    ldconfig
