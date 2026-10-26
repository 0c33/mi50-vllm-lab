# mi50-vllm-lab

**Qwen3.8-27B on two 2018-era AMD MI50 cards (gfx906, 16 GB each): a 200k-token context for coding agents, 74 t/s with speculative decoding, custom attention kernels that make long context usable, and three bug fixes for the gfx906 vLLM fork.**

This repo holds everything behind my write-up ([blog post](https://blog.sbm-labs.com/en/posts/mi50-vllm-part-2/)):

- the Docker build;
- the serve profiles I use daily for Hermes Agent and opencode;
- five custom HIP kernels: three for speculative decoding, two for attention over vLLM's int8 KV cache;
- the vLLM patches;
- the benchmark scripts.

![Decode speed](docs/charts/decode-speed.svg)

## Results

| | Before | After |
|---|---|---|
| DFlash2 (k=7) speculative decoding | 10.2 t/s, every draft rejected | **73.7 t/s** mean, 85 peak |
| MTP (k=3) speculative decoding | 49.9 t/s | **65.1 t/s** with the custom kernels |
| Max context, coding-agent profile | ~96k (fp16 KV) | **200k** (int8 KV); 143k-token prompt verified end to end |
| TP=2 all-reduce without P2P | ~72 µs (RCCL) | **28 µs**, bit-identical |
| Decode at 61k context (agent profile) | 29.1 t/s | **48.2 t/s** |
| Prefill, cold 63k-token prompt | 90.4 tok/s | **160.4 tok/s** |
| Agent follow-up turn, 16k conversation | 12.1–14.5 s | **4.4–5.3 s** to first token |
| Agent follow-up turn, ~60k conversation | ~45 s | **6.2 s** to first token |

The cards are passively cooled and power-capped at **90 W each**. Full numbers and methodology are in [`docs/RESULTS.md`](docs/RESULTS.md), with the raw logs in [`docs/logs/`](docs/logs/).

## What I found and fixed

1. **The DFlash2 drafter overflows fp16.** gfx906 has no bf16, so the bf16-trained drafter runs in fp16. By its second layer its residual stream passes 65504 and turns into inf/NaN, and acceptance drops to exactly 0. The fix divides the residual stream by a power of two once, at load time. Every branch reads the stream through an RMSNorm, so the outputs don't change, and it costs nothing at runtime.
2. **The drafter read freed KV-cache blocks.** The fork's non-causal attention path dropped the drafter's 2048-token sliding window. Past ~2–4k context it attended to blocks the cache manager had already freed, and acceptance fell back to 0. The fix keeps the window inside the existing kernel mask (Python only), and a test poisons out-of-window blocks with NaN to catch any regression.
3. **int8 KV crashed at boot.** `triton_attn.py` read a field that vLLM 0.30.0 removed. This was a one-line fix, and it is what unlocks the 200k profile.

![DFlash2 vs context](docs/charts/dflash2-context.svg)

Upstream: these fixes are prepared for [KIntegrated/vllm-gfx906-mobydick](https://github.com/KIntegrated/vllm-gfx906-mobydick) as two PRs, (1) and (2) together and (3) on its own. The PR texts and `git am`-ready patch series are in [`upstream/`](upstream/).

## Long context, made fast

The 200k profile worked, but long prompts were slow: at 63k tokens, decode ran at 29 t/s and a cold prompt took 11.6 minutes. Measuring each piece showed the same culprit in both phases. vLLM's Triton attention kernel, the one that reads the int8 KV cache, reached only ~1.1–1.3 TFLOPS on gfx906.

1. **Prefill.** The Triton kernel now processes 64 query rows per program instead of 16 (1.8× faster). Prefill chunks then go to the fork's own gfx906 FlashAttention kernel, which I taught to read vLLM's int8 per-token-head cache in place ([`kernels/fa_pth`](kernels/fa_pth/)): 4.45 TFLOPS, 3.5× the original. A cold 63k-token prompt went from 90 to **160 tok/s**.
2. **Decode.** MTP verifies 4 tokens per step, and 6 query heads share each KV head, so 24 rows read the same K/V and decode attention is compute-bound. A new split-K kernel ([`kernels/pth_dec`](kernels/pth_dec/)) computes all 24 rows from one read of each key, using fp16 dot products and no quantization of Q. It is safe inside captured HIP graphs and serves the MTP drafter too. Attention is 3.2–4.5× faster, and decode at 61k context went from 29 to **48 t/s**.
3. **Agent turns.** On this hybrid model, vLLM matched cached prefixes per 1,584-token block, and with MTP it dropped the last matched block of every hit, so each agent turn re-read ~2.4–2.9k tokens. Two existing vLLM options (`--prefix-match-unit 16`, `--enable-mamba-fine-grained-prefix-cache`) cut that to the new tokens only: **4.4–5.3 s** to first token instead of 12–14.5 s on a 16k conversation, with byte-identical replies.

Both kernels were checked against a float64 reference and compared end to end with the Triton path on the same prompts. Details and quality checks are in [`docs/RESULTS.md`](docs/RESULTS.md#prefill-attention-2026-10-12).

![Decode vs context](docs/charts/long-context.svg)

![Agent turns](docs/charts/agent-turns.svg)

## Kernels

Speculative decoding verifies k+1 tokens per step, so most matmuls have M ≤ 8 rows. At that size they are bound by memory bandwidth, not compute, and the general GEMM paths handle them badly. [`kernels/`](kernels/) builds one extension, `gfx906_skinny.so`, with three kernels for that, plus two attention kernels:

| Kernel | What it does | Speedup |
|---|---|---|
| `skinny.cu` | fp16 GEMM for M ≤ 8. Coalesced 16-byte weight loads, activations staged in LDS, butterfly reduction. | lm_head at M=7: 10.3 → 3.4 ms |
| `w4.cu` | 4-bit (Exllama layout) GEMM for M ≤ 8. Each weight is dequantized once and reused for all rows, with fp32 accumulation and no fp16 atomics. | 1.1–1.7× per layer |
| `hostar.cu` | TP=2 all-reduce through pinned host memory with a generation-flag handshake. Safe to replay in CUDA graphs. | 72 → 28 µs |
| `fa_pth/` | Prefill attention: the fork's gfx906 FA kernel (from [gfx906-fa-vllm](https://github.com/cassettesgoboom/gfx906-fa-vllm)), changed to read vLLM's int8 per-token-head KV cache in place, with fp32 scale products and P·V accumulation. [`fa_pth.patch`](kernels/fa_pth/fa_pth.patch) shows the changes. | 1.9× the tuned Triton kernel |
| `pth_dec/` | Decode attention for the same cache: split-K, one read of each key for all 24 query rows, fp16 dot products, HIP-graph safe. | 3.2–4.5× the Triton decode path |

## Run it

```bash
docker build -t vllm-gfx906-kintegrated:0.30.0-rocm7.14 -f docker/vllm.Dockerfile docker/  # long build; run in tmux
bash docker/build-opt.sh                 # applies patches/vllm/*.patch and compiles kernels/ (no GPU needed)
MODELS=~/models bash serve/serve-agent.sh  # 200k context, MTP k=3, tool calling
```

Kernel tests run inside the built image, one GPU, under the temperature watchdog:

```bash
bash bench/safe-run.sh -- docker run --rm --label gfxjob=1 --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  -v "$PWD/kernels":/k -e HIP_VISIBLE_DEVICES=0 \
  vllm-gfx906-kintegrated:0.30.0-rocm7.14-opt python3 /k/pth_dec/test_dec.py quick
```

The model directory must hold `cyankiwi/Qwen3.8-27B-AWQ-INT4`, plus `z-lab/Qwen3.8-27B-DFlash2` if you use `serve/serve-dflash2.sh`.

The endpoint is OpenAI-compatible at `http://<host>:8000/v1` with model name `qwen3.8-27b`.

| Script | Use |
|---|---|
| `serve/serve-agent.sh` | Coding agents (Hermes, opencode): int8 KV with the gfx906 attention kernels, 200k context, MTP k=3, up to 4 concurrent requests, prefix caching matched every 16 tokens. Knobs: `PMU`, `FINE` |
| `serve/serve.sh` | MTP general profile. Knobs: `KV_DTYPE`, `MAX_LEN`, `SEQS`, `SPEC_K`, `CAPTURE`, `PTH_FA`, `PTH_DEC` (0 = Triton attention) |
| `serve/serve-dflash2.sh` | Fastest short-context decode: DFlash2 k=7, up to 32k context, prefix caching off |

## Limits

- At 90 W, prefill is compute-bound: a cold 63k-token prompt takes 6.5 min (160 tok/s). A cold 143k-token prompt took ~42 min before the new kernels and has not been re-measured. Later turns reuse the prefix cache: 6.2 s to first token at 59k.
- The cards are passively cooled and hold only ~72–78 W each for long, so long runs get throttled below 90 W. Run benchmarks under [`bench/safe-run.sh`](bench/safe-run.sh), which stops the load at 80 °C on any sensor (see [Thermal safety](docs/RESULTS.md#thermal-safety)).
- The 200k profile serves up to 4 requests at once from one shared 221k-token KV pool. If the active sessions together need more than that, vLLM pauses one and it re-reads part of its prompt later. While a long prompt is being read, other requests slow to ~2 t/s (measured before the new kernels): prefill is compute-bound at 90 W, and smaller prefill chunks only made both sides slower. The full 200k length boots, with a 221k-token KV pool, but the longest prompt actually run is 143k.
- DFlash2's 3.6 GB drafter limits it to ~32k context on 16 GB cards, and it needs prefix caching off. For agents, the MTP profile is the one to use.

## How this was built

I set the goals, constraints and hardware, and reviewed the results. The engineering work (reading the fork's code, writing the kernels and fixes, running the A/B measurements) was done with [Claude Code](https://claude.com/claude-code) running as an agent on the box under fixed rules: everything in Docker, host ROCm untouched, builds in tmux with logs. Every number here comes from a logged run.

## Credits

- [KIntegrated/vllm-gfx906-mobydick](https://github.com/KIntegrated/vllm-gfx906-mobydick): the gfx906 vLLM fork this builds on.
- [cassettesgoboom/gfx906-fa-vllm](https://github.com/cassettesgoboom/gfx906-fa-vllm) (Nick, Kevin Read): the gfx906 FlashAttention kernel that the fork vendors and `kernels/fa_pth` extends.
- [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906): the gfx906 PyTorch/ROCm images. `docker/vllm.Dockerfile` is derived from its vllm Dockerfile (MIT).
- [ai-infos/flash-attention-gfx906](https://github.com/ai-infos/flash-attention-gfx906), [vLLM](https://github.com/vllm-project/vllm), [z-lab](https://huggingface.co/z-lab) (DFlash2) and [cyankiwi](https://huggingface.co/cyankiwi) (AWQ quant).

## License

Apache-2.0 (see [LICENSE](LICENSE)). `patches/vllm/` modifies Apache-2.0 vLLM code. `kernels/fa_pth/` contains modified Apache-2.0 files from the fork's gfx906 FA kernel. `docker/vllm.Dockerfile` is derived from MIT-licensed work (see [NOTICE](NOTICE)).
