# Results

All numbers were measured on one box between 2026-10-08 and 2026-10-10. Every run used the scripts in `serve/` and `bench/`.

## Box

| Part | Spec |
|---|---|
| GPUs | 2× AMD MI50 16 GB (gfx906, Vega 20). They report as "Radeon Pro VII" (PCI 0x66a1). Passively cooled. **Power-capped at 90 W each** (the driver default is 190 W). |
| Interconnect | PCIe with no GPU peer-to-peer. RCCL goes through host shared memory at ~72–80 µs per all-reduce. |
| Host | Xeon E5-2680 v4, 32 GB DDR4, X99-F8D Plus, Proxmox VE 8 (kernel 6.8.12). vLLM runs in Docker inside an LXC with 10 cores. |
| Software | ROCm 7.14 and PyTorch 2.13 (gfx906 builds by mixa3607), [KIntegrated/vllm-gfx906-mobydick](https://github.com/KIntegrated/vllm-gfx906-mobydick) 0.30.0 @ `6ccbb03`, Triton 3.8.0 |
| Models | `cyankiwi/Qwen3.8-27B-AWQ-INT4` as the target; `z-lab/Qwen3.8-27B-DFlash2` (3.6 GB) as the block-diffusion drafter; MTP from the target's own head |

## How things are measured

- **`bench/bench.py`** runs six chat-templated generations of 512 tokens each: prose, code and reasoning, two reps of each. It reports the mean. Decode t/s is `(completion_tokens - 1) / (t_last_chunk - t_first_chunk)`, so prefill is excluded. Acceptance comes from `/metrics` deltas.
- **`bench/longctx.py`** decodes 384 tokens after a long code-reading prompt.
- **`bench/agentctx.py`** builds a prompt from many distinct vLLM source files, hides a "needle" fact at the top and asks a task at the end. It then sends a follow-up turn on the same conversation. This is the agent-loop pattern.

## Short-context decode (greedy, one sequence)

| Config | t/s | Tokens per verify step |
|---|---|---|
| DFlash2 k=7, fork `main` as shipped | 10.21 | 1.01 |
| DFlash2 k=7 + fp16 fix + window fix, stock kernels | 46.54 | 4.65 |
| MTP k=3, stock | 49.94 | 3.22 |
| MTP k=3 + custom kernels | 65.09 | 3.22 |
| **DFlash2 k=7 (fixed) + custom kernels** | **73.66** (85.1 peak, reasoning) | 4.73 |

At temperature 0.7: MTP k=3 stock 45.7, MTP k=3 + kernels 59.6, DFlash2 k=7 + kernels 67.0.

## DFlash2 vs context: the A/B behind the upstream PR

Setup: fork `main` `6ccbb03`, k=7, model-runner V2, TP=2, max-len 32768, KV pinned at 2 GiB, stock kernels. Each cell is decode t/s, with tokens per verify step in parentheses.

| Arm | bench | 1.5k | 4.6k | 14.6k | 24.4k |
|---|---|---|---|---|---|
| `main` | 10.21 (1.01) | 9.65 (1.00) | — | — | — |
| + fp16 rescale only | 48.42 (4.82) | 39.47 (4.03) | 9.61 (1.00) | 8.76 (1.00) | — |
| + rescale + window fix | 46.54 (4.65) | 39.63 (4.03) | 34.84 (3.66) | 32.75 (3.70) | 31.40 (3.85) |

Tests: `test_gfx906_fa.py`, `test_dflash2.py` and `test_dflash_causality.py` give 127 passed. The new window test fails on `main` with "read a freed block outside the window".

## Long context: the agent profile (`serve/serve-agent.sh`)

Settings:

- int8 per-token-head KV with Triton attention (128 softmax segments);
- MTP k=3;
- `--max-model-len 204800` (the KV pool holds 221,748 tokens), one sequence, prefix caching on;
- `qwen3_coder` tool parser and `qwen3` reasoning parser.

| Prompt | Prefill | Decode | Accept len | Follow-up turn TTFT | Needle |
|---|---|---|---|---|---|
| short | — | 70.2 t/s | — | — | — |
| 62.8k tokens | 107 t/s | 29–32 t/s | — | ~45 s | recalled |
| 143k tokens (160k config) | 57 t/s (2505 s cold) | 17.6 t/s | 3.19 | ~75 s | recalled |

- With the 200k config, 15,585 MiB per card is in use after boot and 15,843 MiB after a 63k run, which leaves ~520 MiB of headroom per card. The longest prompt actually run is 143k tokens.
- Follow-up turns are cheap because of prefix caching. Qwen3.8 is a hybrid (Gated DeltaNet + attention) model, so vLLM caches in "align" blocks of 1,584 tokens. Each turn re-prefills at most one block plus the new message.

KV options that were tried:

| KV cache | Max context on 2×16 GB | Why (not) |
|---|---|---|
| fp16 (custom FA) | ~96k | the baseline |
| **int8 per-token-head** | **200k** | ~16.6 KB/token/rank, works with MTP k=3 |
| TurboQuant k8v4 | ~220k | V1 drafter rejects MTP k>1; full CUDA graphs fail on verify; k=1 + piecewise decodes at only 13.1 t/s at 63k |

## Kernels (`kernels/`)

All three are built as one torch extension, `gfx906_skinny.so`. Speculative decoding verifies k+1 tokens per step, so most matmuls in a step have M ≤ 8 rows. At that size they are memory-bound and a poor fit for the general GEMM paths.

**Skinny fp16 GEMM** (lm_head and drafter layers, µs, vs the fork's fp16 dispatch):

| Shape | M | Fork | New |
|---|---|---|---|
| lm_head 124160×5120 | 7 | 10,295 | 3,410 |
| lm_head 124160×5120 | 8 | 6,702 | 3,507 |
| drafter fc 5120 | 7 | 2,241 | 584 |

**W4 GEMM** (AWQ/GPTQ 4-bit target layers, M=8, µs, vs the Exllama kernel):

| Layer | Exllama | New |
|---|---|---|
| attn_o 3072×5120 | 77.0 | 45.3 |
| gdn_out 3072×5120 | 77.0 | 52.0 |
| attn_qkv 5120×7168 | 149.8 | 106.7 |
| mlp_gate_up 5120×17408 | 345.6 | 249.0 |
| mlp_down 8704×5120 | 159.5 | 144.3 |

It accumulates in fp32 with no fp16 atomics, and is checked against a float64 dequantization reference (`kernels/check_w4.py`).

**Host-memory all-reduce** for TP=2 without P2P: 80 KB costs **28 µs vs ~72 µs** for RCCL. Over 3,000 random calls the results are bit-identical to RCCL (`kernels/test_hostar.py`). A verify step issues ~140 of these.

## Other things found along the way

- **ROCm 7.14 + amdsmi:** if torch is imported before amdsmi, amdsmi enumerates 0 GPUs, and vLLM then sees `device_count() == 0`. Fixed with a one-line `.pth` that imports amdsmi first (`docker/gfx906-amdsmi-first.pth`).
- **Cold `torch.compile` boots** inflate vLLM's profiled peak by ~1.2 GiB, so a config that fits warm can fail cold. The serve scripts pin `--kv-cache-memory-bytes`.
- **`triton_attn.py`** read `CommonAttentionMetadata._seq_lens_cpu`, which 0.30.0 removed, so every int8-KV / Triton-attention boot crashed. Fixed in a one-line upstream PR.
