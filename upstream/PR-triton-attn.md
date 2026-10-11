## Summary

The 0.30.0 merge removed the `seq_lens_cpu` attention-metadata property (#55353), but `TritonAttentionMetadataBuilder.build` still reads `common_attn_metadata._seq_lens_cpu`. As a result, every `TRITON_ATTN` engine dies at CUDA-graph capture with `AttributeError: 'CommonAttentionMetadata' object has no attribute '_seq_lens_cpu'`.

On gfx906 this covers any config the custom FA rejects. The practical one is `--kv-cache-dtype int8_per_token_head`, which is what fits a 160k–200k context next to Qwen3.8-27B-AWQ-INT4 + MTP k=3 on 2× 16 GB cards. Only the opt-in torch-SDPA fast paths read `seq_lens_cpu`, and they skip when it is `None`. Reported by @0c33.

## Test

Qwen3.8-27B-AWQ-INT4, TP=2, MTP k=3, `int8_per_token_head` KV. A 163,840 max-len and a 204,800 max-len config both boot and serve. At a 143,015-token prompt the needle is recalled, decode runs at 17.6 t/s (3.19 tokens/step), and there is no OOM. Without the fix, the engine fails at graph capture.

## AI assistance

Found and fixed with AI assistance (Claude Code) on @0c33's hardware; submitted by @0c33.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
