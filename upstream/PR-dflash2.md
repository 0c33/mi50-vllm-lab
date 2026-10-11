## Summary

DFlash2 was parked as "dead for this family" (`docs/gfx906/DEVLOG-dflash2.md`, `HANDOVER-dflash2.md`). The checkpoint is fine. On this fork the drafter fails for two independent reasons, and this PR fixes both:

1. **fp16 overflow (every context).** The bf16-trained draft's residual stream passes fp16's max (65504) by its second layer, goes inf/NaN, the candidates degrade to ids `0..top_k-1`, and acceptance is exactly 0 (vllm-project/vllm#55250). gfx906 has no bf16, so every DFlash2 run on this fork drafted in fp16. That includes all the "excluded suspects" runs in the handover.
2. **The drafter's sliding window was dropped (context > ~2k).** FA-NONCAUSAL stage 1 zeroes `window` for non-causal batches, so the drafter attended over `[0, seq_len)`. The KV-cache manager frees a sliding layer's blocks once they fall out of the window, so past ~2–4k context the drafter read freed blocks and acceptance went back to exactly 0.

Found and measured on @0c33's box: 2× MI50 16 GB, TP=2, both cards capped at 90 W.

## Changes

| File | What |
|---|---|
| `vllm/model_executor/models/qwen3_dflash2.py` | `rescale_residual_stream`, applied in `load_weights` for fp16 drafts only. It scales o_proj, the up half of gate_up_proj and down_proj (and the layer-0 residual) by a power of two. Every branch reads the residual stream through an RMSNorm, the final norm feeds the candidates and selector, and the grouped conv's `finish` is linear in the branch output. So the draft's outputs are unchanged, and the stream stays inside fp16 range. Zero runtime cost. `VLLM_DFLASH2_FP16_RESIDUAL_DIVISOR=1` turns it off. |
| `vllm/gfx906_fa/gfx906_fa_paged.py` | For non-causal batches with `window > 0`, keep the window. The gather clip still starts at the first row's window start. `q_abs_offset = seq_len` plus `window + n_q` express "no causal clip, left window only" with the existing kernel mask. Python only, no kernel change; the gather clip and k-loop start are the ones causal sliding layers already use. |
| `tests/v1/spec_decode/test_dflash2.py` | `test_dflash2_residual_rescale_is_exact`, `test_dflash2_residual_rescale_keeps_fp16_finite`. A tiny real `DFlash2Qwen3Model` with the attention core stubbed through `o_proj`. |
| `tests/kernels/attention/test_gfx906_fa.py` | `test_non_causal_sliding_window_never_reads_freed_blocks[64,200]`. Out-of-window blocks point at a NaN block; the result must match the windowed reference. Fails on `main` with "read a freed block outside the window". |
| `docs/gfx906/*` | DEVLOG entries (verdict-first, A/B numbers), a superseded banner on the handover, and the DFL2-8 resolution. |

## Serving A/B (gate)

Fork `main` `6ccbb03`, Qwen3.8-27B-AWQ-INT4 + `z-lab/Qwen3.8-27B-DFlash2`, a byte-identical mirror of `incoai/Qwen3.8-27B-DFlash2`. Config: k=7, V2, float16, TP=2, max-len 32768, max-num-seqs 1, bt 1024, capture `[8]`, prefix caching off, KV pinned at 2 GiB, stock fork kernels. The bench is 6 chat-templated 512-token greedy generations (prose / code / reasoning × 2); each long column is one code-reading prompt (384 tokens out, greedy). Values are decode t/s, with tokens per verify step in parentheses.

| arm | bench | 1.5k | 4.6k | 14.6k | 24.4k |
|---|---|---|---|---|---|
| `main` | 10.21 (1.01) | 9.65 (1.00) | — | — | — |
| + fp16 rescale only | 48.42 (4.82) | 39.47 (4.03) | 9.61 (1.00) | 8.76 (1.00) | — |
| **+ rescale + window (this PR)** | **46.54 (4.65)** | **39.63 (4.03)** | **34.84 (3.66)** | **32.75 (3.70)** | **31.40 (3.85)** |

For reference, MTP k=3 on the same box (V1, max-len 32768, 4 seqs) reads 49.9 t/s on the bench. DFlash2 wins on code and reasoning and loses on prose. These cards are 16 GB at 90 W, so the 32 GB-box gate (agentic corpus at 64k/120k against MTP k=3 + CAT-1) is for the maintainers to run.

## Tests

```
pytest tests/kernels/attention/test_gfx906_fa.py tests/v1/spec_decode/test_dflash2.py \
       tests/v1/spec_decode/test_dflash_causality.py
127 passed
pytest tests/kernels/attention/test_gfx906_fa.py -k never_reads_freed    # on main, without this PR
2 failed: AssertionError: read a freed block outside the window
```

## Relation to #38 / #41 (not a duplicate)

#38 (phoenixclyde, against `gfx906/v0.28.0`, open) independently found the fp16 overflow. It fixes it with an fp32 residual stream (fp32 casts and norms every step) and adds a non-causal flag in the C++ FA kernels. #41 ported #38's small fixes to the 0.30.0 line without either of those changes, so `main` still has both bugs. This PR targets `main`, and both approaches differ:

- the rescale is load-time only, so the drafter stays fp16 with no per-step cost;
- the window fix is Python-only on top of `main`'s FA-NONCAUSAL stage 1 (no extension rebuild).

The PR also documents the freed-KV mechanism behind the long-context collapse.

## Limits

- The 3.6 GB draft costs ~1.8 GiB/rank, so on 16 GB cards DFlash2 tops out around 32k context and 1 sequence.
- Prefix caching stays off for DFlash2 on hybrid targets (vLLM#48375 is unmerged).

## AI assistance

This change was investigated, implemented and measured with AI assistance (Claude Code) on @0c33's hardware, and submitted by @0c33. Every number above comes from the runs described; the test commands are the ones listed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
