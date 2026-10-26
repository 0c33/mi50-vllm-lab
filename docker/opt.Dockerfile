# gfx906 DFlash2/MTP optimizations on top of the KIntegrated 0.30.0 image.
#  - qwen3_dflash2.py: fp16-safe DFlash2 drafter (residual-stream rescale, vllm#55250)
#  - gfx906_fa_paged.py: non-causal FA keeps the drafter's sliding window
#    (fixes 0 % acceptance beyond ~2k context)
#  - utils.py + exllama.py + gfx906_skinny.so: M<=8 fp16 and 4-bit GEMM kernels
#  - cuda_communicator.py + gfx906_host_allreduce.py: TP=2 all-reduce via host memory
#  - triton_attn.py: drop the removed _seq_lens_cpu read (TRITON_ATTN / int8 KV crashed);
#    route int8 per-token-head KV attention to the gfx906 kernels below
#  - triton_unified_attention.py: 64 query rows per program for gfx906 prefill (1.8x)
#  - _gfx906_fa_pth.so: prefill rows (>= 64 queries) of the int8 per-token-head KV cache
#    through the fork's gfx906 FA kernel, reading the cache in place
#    (VLLM_GFX906_PTH_FA=0 keeps Triton)
#  - _gfx906_pth_dec.so: rows with <= 4 queries (decode, MTP verify, MTP drafter) through a
#    split-K decode kernel for the same cache, HIP-graph safe (VLLM_GFX906_PTH_DEC=0 keeps Triton)
# Disable the GEMM kernels/all-reduce by setting the *_LIB env vars to empty.
# Build with docker/build-opt.sh (it applies patches/vllm/*.patch and compiles kernels/).
ARG BASE=vllm-gfx906-kintegrated:0.30.0-rocm7.14
FROM ${BASE}
ARG SITE=/usr/local/lib/python3.12/dist-packages
COPY qwen3_dflash2.py $SITE/vllm/model_executor/models/qwen3_dflash2.py
COPY gfx906_fa_paged.py $SITE/vllm/gfx906_fa/gfx906_fa_paged.py
COPY utils.py $SITE/vllm/model_executor/layers/utils.py
COPY exllama.py $SITE/vllm/model_executor/kernels/linear/mixed_precision/exllama.py
COPY cuda_communicator.py $SITE/vllm/distributed/device_communicators/cuda_communicator.py
COPY gfx906_host_allreduce.py $SITE/vllm/distributed/device_communicators/gfx906_host_allreduce.py
COPY triton_attn.py $SITE/vllm/v1/attention/backends/triton_attn.py
COPY triton_unified_attention.py $SITE/vllm/v1/attention/ops/triton_unified_attention.py
COPY gfx906_skinny.so /opt/gfx906/gfx906_skinny.so
COPY _gfx906_fa_pth.so /opt/gfx906/_gfx906_fa_pth.so
COPY _gfx906_pth_dec.so /opt/gfx906/_gfx906_pth_dec.so
ENV VLLM_GFX906_SKINNY8_LIB=/opt/gfx906/gfx906_skinny.so \
    VLLM_GFX906_HOSTAR_LIB=/opt/gfx906/gfx906_skinny.so
