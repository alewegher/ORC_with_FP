# Extends the course image (andreadelprete/orc24:v1) with:
#   - GPU-enabled PyTorch (CUDA 12.1 wheels; needs nvidia-container-toolkit + --gpus on the host)
#   - l4casadi built from the vendored submodule (../l4casadi)
#
# Build from the REPO ROOT (not this folder), so the l4casadi submodule is in context:
#   docker build -f docker/Dockerfile -t orc24-gpu:latest .

FROM andreadelprete/orc24:v1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.1 toolkit (nvcc + dev headers/libs) — l4casadi's CMake build needs the
# actual toolkit, not just the CUDA runtime libs bundled in the torch wheel.
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# GPU build of torch. cu121 wheels work with any driver >= 530 (RTX 40-series driver 580 included).
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY l4casadi/requirements_build.txt /tmp/l4casadi_requirements_build.txt
RUN pip3 install --no-cache-dir -U packaging \
    && pip3 install --no-cache-dir -r /tmp/l4casadi_requirements_build.txt

# Build l4casadi's C++ extension against the torch just installed above.
COPY l4casadi /tmp/l4casadi
RUN pip3 install --no-cache-dir /tmp/l4casadi --no-build-isolation && rm -rf /tmp/l4casadi

USER student
WORKDIR /home/student
