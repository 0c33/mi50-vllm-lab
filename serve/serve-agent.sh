#!/bin/bash
# Long-context agent profile (Hermes agent / opencode): Qwen3.8-27B-AWQ-INT4, TP=2,
# MTP k=3, int8 KV (Triton attention), 200k max context, up to 4 concurrent
# requests sharing one 221k-token KV pool, prefix caching on, qwen3_coder
# tool parser + qwen3 reasoning parser.
# Validated 2026-10-09 on 2x 16 GB gfx906 @ 90 W: 143k-token prompt (needle OK,
# 17.6 t/s decode, no OOM); 200k config boots with a 221,748-token pool and runs
# a 63k agentic probe (29-32 t/s, needle OK). KV is pinned (4.1 GiB/rank).
# Usage: ./serve-agent.sh   (foreground; run inside tmux)
exec env \
  SEGMENTS=128 KV_DTYPE=int8_per_token_head MAX_LEN=${MAX_LEN:-204800} SEQS=${SEQS:-4} \
  "CAPTURE=${CAPTURE:-[4,8,12,16]}" BATCHED=2048 GPU_UTIL=0.92 \
  EXTRA_ARGS="--kv-cache-memory-bytes ${KV_BYTES:-4402341478} --long-prefill-token-threshold ${LONG_PREFILL:-256}" \
  "$(dirname "$0")/serve.sh"
