#!/bin/bash
# Serve cyankiwi/Qwen3.8-27B-AWQ-INT4 with MTP k=3 on 2x gfx906 (16 GB) with
# the KIntegrated vllm-gfx906-mobydick 0.30.0 image built in ./build.
# Recipe: README "Validated serve command" + running.md, scaled to 16 GB cards.
# Usage: ./serve.sh            (foreground; run inside tmux)
#        GPU_UTIL=0.92 ./serve.sh
#        long context: see serve-agent.sh
set -eu
. "$(dirname "$0")/common.sh"
# The -opt image enables the gfx906 GEMM kernels and the host-memory all-reduce
# (see serve-dflash2.sh); SKINNY=0 / HOSTAR=0 turn them off.
SPEC_K=${SPEC_K:-3}
SPEC_ARGS=""
[ "$SPEC_K" -gt 0 ] && SPEC_ARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}"
EXTRA_MOUNTS=()
[ "${SKINNY:-1}" = 1 ] || EXTRA_MOUNTS+=(-e VLLM_GFX906_SKINNY8_LIB=)
[ "${HOSTAR:-1}" = 1 ] || EXTRA_MOUNTS+=(-e VLLM_GFX906_HOSTAR_LIB=)
# int8 per-token-head KV only: the gfx906 prefill (PTH_FA) and decode (PTH_DEC) attention
# kernels; 0 keeps that path on vLLM's Triton kernel (A/B).
[ "${PTH_FA:-1}" = 1 ] || EXTRA_MOUNTS+=(-e VLLM_GFX906_PTH_FA=0)
[ "${PTH_DEC:-1}" = 1 ] || EXTRA_MOUNTS+=(-e VLLM_GFX906_PTH_DEC=0)

exec docker run --rm --name vllm-qwen38 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --ipc host --shm-size 16g \
  -p 8000:8000 \
  -v "$MODELS/Qwen3.8-27B-AWQ-INT4":/models/Qwen3.8-27B-AWQ-INT4:ro \
  -v "$CACHE_DIR":/root/.cache \
  "${EXTRA_MOUNTS[@]}" \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_GFX906_HIP_LIB_PATH=/opt/rocm/lib/libamdhip64.so.7 \
  -e VLLM_USE_V2_MODEL_RUNNER="${V2:-0}" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS="${SEGMENTS:-16}" \
  "$IMAGE" \
  vllm serve /models/Qwen3.8-27B-AWQ-INT4 \
    --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len "${MAX_LEN:-32768}" \
    --max-num-seqs "${SEQS:-4}" \
    --max-num-batched-tokens "${BATCHED:-2048}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
    --language-model-only \
    --compilation-config "{\"cudagraph_capture_sizes\":${CAPTURE:-[4,8,12,16]}${CG_MODE:+,\"cudagraph_mode\":\"$CG_MODE\"}}" \
    --kv-cache-dtype "${KV_DTYPE:-auto}" \
    ${SPEC_ARGS:-} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --generation-config auto \
    ${EXTRA_ARGS:-}
