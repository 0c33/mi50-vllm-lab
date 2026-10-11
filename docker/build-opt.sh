#!/bin/bash
# Build the -opt image: base image + patches/vllm/*.patch + gfx906_skinny.so.
#  1. copy the stock vLLM files out of the base image and apply the diffs
#  2. compile kernels/ (skinny GEMM, W4 GEMM, host all-reduce) inside the base image
#  3. docker build docker/opt.Dockerfile with the patched files + .so as context
# Usage: docker/build-opt.sh   (needs the base image built from docker/vllm.Dockerfile)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE:-vllm-gfx906-kintegrated:0.30.0-rocm7.14}
TAG=${TAG:-vllm-gfx906-kintegrated:0.30.0-rocm7.14-opt}
SITE=/usr/local/lib/python3.12/dist-packages
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT

declare -A FILES=(
  [qwen3_dflash2]=vllm/model_executor/models/qwen3_dflash2.py
  [gfx906_fa_paged]=vllm/gfx906_fa/gfx906_fa_paged.py
  [utils]=vllm/model_executor/layers/utils.py
  [exllama]=vllm/model_executor/kernels/linear/mixed_precision/exllama.py
  [cuda_communicator]=vllm/distributed/device_communicators/cuda_communicator.py
  [triton_attn]=vllm/v1/attention/backends/triton_attn.py
)
for name in "${!FILES[@]}"; do
  docker run --rm --entrypoint cat "$BASE" "$SITE/${FILES[$name]}" > "$CTX/$name.py"
  patch --quiet --no-backup-if-mismatch "$CTX/$name.py" "$ROOT/patches/vllm/$name.patch"
done
cp "$ROOT/patches/vllm/gfx906_host_allreduce.py" "$CTX/"

docker run --rm -v "$ROOT/kernels":/src:ro -v "$CTX":/out "$BASE" bash -c \
  'cp -r /src /tmp/k && python3 /tmp/k/build.py && cp /tmp/k/build/gfx906_skinny.so /out/'

docker build -t "$TAG" -f "$ROOT/docker/opt.Dockerfile" --build-arg BASE="$BASE" "$CTX"
echo "built $TAG"
