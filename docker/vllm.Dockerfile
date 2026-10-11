# Derived from mixa3607/ML-gfx906 vllm-v2/vllm.Dockerfile, adjusted for the
# KIntegrated/vllm-gfx906-mobydick 0.30.0 line per its README/running.md:
#  - stock triton v3.8.0 built AMD-only with the install-RPATH workaround
#  - PYTORCH_ROCM_ARCH=gfx906 explicit (no GPU during docker build)
#  - cmake/pkg-config/PKG_CONFIG_PATH for the HIP extension build
#  - VLLM_VERSION_OVERRIDE (shallow clone has no usable tag for setuptools_scm)
#  - pip constraints so requirements never replace the gfx906 torch/triton
#  - HIP blocking-sync .pth shim + VLLM_GFX906_HIP_LIB_PATH for TP>=2
# Build seq: rocm_base => build_base => build_triton => build_fa => build_vllm => final

ARG BASE_PYTORCH_IMAGE="docker.io/mixa3607/pytorch-gfx906:v2.13.0-rocm-7.14"
ARG MAX_JOBS="6"
ARG EXTRA_REQUIREMENTS="empty.txt"

ARG VLLM_REPO="https://github.com/KIntegrated/vllm-gfx906-mobydick.git"
ARG VLLM_BRANCH="main"
ARG VLLM_COMMIT=""
ARG VLLM_PATCH="empty.patch"
ARG VLLM_VERSION_OVERRIDE="0.30.0"

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG TRITON_BRANCH="v3.8.0"
ARG TRITON_COMMIT=""
ARG TRITON_PATCH="empty.patch"

ARG FA_REPO="https://github.com/ai-infos/flash-attention-gfx906.git"
ARG FA_BRANCH="main"
ARG FA_COMMIT=""
ARG FA_PATCH="empty.patch"

############# Base image #############
FROM ${BASE_PYTORCH_IMAGE} AS rocm_base

ENV PYTORCH_ROCM_ARCH=gfx906
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib
ENV VLLM_TARGET_DEVICE=rocm
ENV FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python3 -c "from importlib.metadata import version as v; print('\\n'.join(f'{p}=={v(p)}' for p in ('torch','torchvision','torchaudio')))" > /etc/pip-constraints.txt && \
    cat /etc/pip-constraints.txt

RUN pip3 install --upgrade --ignore-installed '/opt/rocm/share/amd_smi' pyjwt && \
    pip3 cache purge && \
    apt-get update && apt-get install -y curl git wget jq aria2 pkg-config && \
    rm -rf /var/lib/apt/lists/*

############# Clone repos #############
FROM rocm_base AS files_triton
ARG TRITON_REPO
ARG TRITON_BRANCH
ARG TRITON_COMMIT
ARG TRITON_PATCH
WORKDIR /app/triton
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 --branch ${TRITON_BRANCH} ${TRITON_REPO} .
RUN if [ "$TRITON_COMMIT" != "" ]; then git checkout "$TRITON_COMMIT"; fi
COPY ./patch/${TRITON_PATCH} ./${TRITON_PATCH}
RUN git apply ./${TRITON_PATCH} --allow-empty

FROM rocm_base AS files_fa
ARG FA_REPO
ARG FA_BRANCH
ARG FA_COMMIT
ARG FA_PATCH
WORKDIR /app/flash-attention
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 --branch ${FA_BRANCH} ${FA_REPO} .
RUN if [ "$FA_COMMIT" != "" ]; then git checkout "$FA_COMMIT"; fi
COPY ./patch/${FA_PATCH} ./${FA_PATCH}
RUN git apply ./${FA_PATCH} --allow-empty

FROM rocm_base AS files_vllm
ARG VLLM_REPO
ARG VLLM_BRANCH
ARG VLLM_COMMIT
ARG VLLM_PATCH
WORKDIR /app/vllm
RUN git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 --branch ${VLLM_BRANCH} ${VLLM_REPO} .
RUN if [ "$VLLM_COMMIT" != "" ]; then test "$(git rev-parse HEAD | cut -c1-${#VLLM_COMMIT})" = "$VLLM_COMMIT"; fi
COPY ./patch/${VLLM_PATCH} ./${VLLM_PATCH}
RUN git apply ./${VLLM_PATCH} --allow-empty

FROM rocm_base AS files_extra
ARG EXTRA_REQUIREMENTS
WORKDIR /app/extra-requirements
COPY ./requirements/${EXTRA_REQUIREMENTS} /app/extra-requirements/requirements.txt

############# Build base #############
FROM rocm_base AS build_base
RUN pip3 install build

############# Build triton #############
FROM build_base AS build_triton
COPY --from=files_triton /app/triton /app/triton
WORKDIR /app/triton
RUN pip3 install -r python/requirements.txt
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    TRITON_CODEGEN_BACKENDS="amd" \
    TRITON_APPEND_CMAKE_ARGS="-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" \
    pip3 wheel -v --no-deps --no-build-isolation --wheel-dir /dist . && \
    rm -rf /root/.triton/llvm /app/triton/build
RUN pip3 install /dist/triton-*.whl && \
    python3 -c "import triton; print('triton', triton.__version__)"

############# Build FA #############
FROM build_triton AS build_fa
COPY --from=files_fa /app/flash-attention /app/flash-attention
WORKDIR /app/flash-attention
RUN pip3 install ninja packaging wheel pybind11 psutil
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    pip3 wheel -v --no-deps --no-build-isolation --wheel-dir /dist .
RUN pip3 install --no-deps /dist/flash_attn-*.whl
RUN ls /dist

############# Build vllm #############
FROM build_fa AS build_vllm
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
COPY --from=files_vllm /app/vllm /app/vllm
WORKDIR /app/vllm
RUN echo "triton==$(python3 -c 'import triton; print(triton.__version__.split("+")[0])')" >> /etc/pip-constraints.txt && \
    cat /etc/pip-constraints.txt
RUN pip3 install -r requirements/build/rocm.txt && pip3 install -r requirements/rocm.txt
ARG MAX_JOBS
ARG VLLM_VERSION_OVERRIDE
ENV PKG_CONFIG_PATH=/opt/rocm/core-7.14/lib/rocm_sysdeps/lib/pkgconfig
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE} \
    pip3 wheel -v --no-deps --no-build-isolation --wheel-dir /dist . 2>&1 | tee /dist/vllm-build.log | tail -200 && \
    ls /dist/vllm-*.whl
RUN python3 -c "import zipfile,glob; n=zipfile.ZipFile(glob.glob('/dist/vllm-*.whl')[0]).namelist(); so=[x for x in n if x.endswith('.so')]; print('\n'.join(so)); assert any('_gfx906_fa_C' in x for x in so), 'gfx906 FA ext missing'; assert any('_rocm_C' in x for x in so), '_rocm_C missing'"

############# Install all #############
FROM rocm_base AS final
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
WORKDIR /app/vllm
RUN --mount=type=bind,from=build_vllm,src=/app/vllm/requirements/,target=/app/vllm/requirements \
    --mount=type=bind,from=build_vllm,src=/etc/pip-constraints.txt,target=/tmp/c.txt \
    --mount=type=bind,from=files_extra,src=/app/extra-requirements/,target=/app/extra-requirements \
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist \
    cp /tmp/c.txt /etc/pip-constraints.txt && \
    pip3 install /dist/*.whl -r /app/vllm/requirements/rocm.txt && \
    pip3 install -r /app/extra-requirements/requirements.txt && \
    pip3 cache purge && \
    pip3 freeze | grep -E '^(torch|torchvision|torchaudio|triton|vllm|flash.attn)'

# ROCm 7.14: amdsmi enumerates 0 GPUs if torch loads first (running.md "7.14 gotcha");
# on this LXC that makes torch/vLLM device_count() return 0. Load amdsmi first.
COPY gfx906-amdsmi-first.pth /usr/local/lib/python3.12/dist-packages/gfx906-amdsmi-first.pth
COPY gfx906-blocking-sync.pth /usr/local/lib/python3.12/dist-packages/gfx906-blocking-sync.pth
ENV VLLM_GFX906_HIP_LIB_PATH=/opt/rocm/lib/libamdhip64.so.7

CMD ["/bin/bash"]
