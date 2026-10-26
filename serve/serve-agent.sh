#!/bin/bash
# Long-context agent profile (Hermes agent / opencode): Qwen3.8-27B-AWQ-INT4, TP=2,
# MTP k=3, int8 KV (Triton attention), 200k max context, up to 4 concurrent
# requests sharing one 221k-token KV pool, prefix caching on, qwen3_coder
# tool parser + qwen3 reasoning parser.
# Validated 2026-10-09 on 2x 16 GB gfx906 @ 90 W: 143k-token prompt (needle OK,
# 17.6 t/s decode, no OOM); 200k config boots with a 221,748-token pool and runs
# a 63k agentic probe (29-32 t/s, needle OK). KV is pinned (4.1 GiB/rank).
# 2026-10-12/13: gfx906 attention kernels for the int8 KV cache (prefill 64k 90 -> 160 tok/s,
# decode at 61k 29 -> 48 t/s) and prefix-cache hits every 16 tokens instead of every
# 1584-token hybrid block, plus Mamba checkpoints at shared-prefix junctions: an agent turn
# re-reads only its new tokens (16k context: turn TTFT 12-14.5 s -> 4.4-5.3 s, identical
# outputs). Rollback knobs: PTH_FA=0, PTH_DEC=0, PMU=1584 (block-level matching), FINE=0.
# Usage: ./serve-agent.sh   (foreground; run inside tmux)
PMU=${PMU:-16}
FINE=${FINE:-1}
FINE_ARG=""
[ "$FINE" = 1 ] && FINE_ARG=" --enable-mamba-fine-grained-prefix-cache"
exec env \
  SEGMENTS=128 KV_DTYPE=int8_per_token_head MAX_LEN=${MAX_LEN:-204800} SEQS=${SEQS:-4} \
  "CAPTURE=${CAPTURE:-[4,8,12,16]}" BATCHED=2048 GPU_UTIL=0.92 \
  EXTRA_ARGS="--kv-cache-memory-bytes ${KV_BYTES:-4402341478} --long-prefill-token-threshold ${LONG_PREFILL:-0} --prefix-match-unit ${PMU}${FINE_ARG}" \
  bash "$(dirname "$0")/serve.sh"
