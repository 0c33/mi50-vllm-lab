#!/bin/bash
# Serve Qwen3.8-27B-AWQ-INT4 + z-lab DFlash2 drafter (k=7) on 2x 16 GB gfx906, TP=2.
#
# Image vllm-gfx906-kintegrated:0.30.0-rocm7.14-opt (docker/opt.Dockerfile) carries:
#  - fp16-safe DFlash2 drafter (residual-stream rescale; vllm#55250 overflow)
#  - non-causal FA that keeps the drafter's 2048 sliding window (else 0 % accepted
#    beyond ~2k context: freed KV blocks were read)
#  - M<=8 fp16 + 4-bit GEMM kernels and a TP=2 host-memory all-reduce (kernels/)
# Toggles: SKINNY=0 / HOSTAR=0 turn the kernels / all-reduce off, DIVISOR=1 turns
# the fp16 fix off (expect ~0 % acceptance), TIMING=1 logs a per-step GPU-time
# breakdown. KV is pinned (KV_BYTES) so boots do not depend on a warm compile cache.
# Usage: ./serve-dflash2.sh   (foreground; run inside tmux)
set -eu
. "$(dirname "$0")/common.sh"
SITE=/usr/local/lib/python3.12/dist-packages
K=${K:-7}

ARGS=()
[ "${SKINNY:-1}" = 1 ] || ARGS+=(-e VLLM_GFX906_SKINNY8_LIB=)
[ "${HOSTAR:-1}" = 1 ] || ARGS+=(-e VLLM_GFX906_HOSTAR_LIB=)
if [ "${TIMING:-0}" = 1 ]; then
  P=${TIMING_FILES:?set TIMING_FILES to a dir with the patched model_runner.py, dflash_speculator.py, gfx906_steptimer.py}
  ARGS+=(-v $P/gfx906_steptimer.py:$SITE/vllm/v1/worker/gpu/gfx906_steptimer.py:ro
         -v $P/model_runner.py:$SITE/vllm/v1/worker/gpu/model_runner.py:ro
         -v $P/dflash_speculator.py:$SITE/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:ro
         -e VLLM_GFX906_STEP_TIMING=1 -e VLLM_GFX906_STEP_TIMING_EVERY=100)
fi

exec docker run --rm --name vllm-qwen38 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --ipc host --shm-size 16g \
  -p 8000:8000 \
  -v "$MODELS/Qwen3.8-27B-AWQ-INT4":/models/Qwen3.8-27B-AWQ-INT4:ro \
  -v "$MODELS/Qwen3.8-27B-DFlash2":/models/Qwen3.8-27B-DFlash2:ro \
  -v "$CACHE_DIR":/root/.cache \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_GFX906_HIP_LIB_PATH=/opt/rocm/lib/libamdhip64.so.7 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_DFLASH2_FP16_RESIDUAL_DIVISOR="${DIVISOR:-32}" \
  -e HF_HUB_OFFLINE=1 \
  "${ARGS[@]}" \
  "$IMAGE" \
  vllm serve /models/Qwen3.8-27B-AWQ-INT4 \
    --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len "${MAX_LEN:-32768}" \
    --max-num-seqs "${SEQS:-1}" \
    --max-num-batched-tokens "${BATCHED:-1024}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
    --kv-cache-memory-bytes "${KV_BYTES:-2147483648}" \
    --no-enable-prefix-caching \
    --language-model-only \
    --compilation-config "{\"cudagraph_capture_sizes\":${CAPTURE:-[8]}}" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":$K}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --generation-config auto \
    ${EXTRA_ARGS:-}
