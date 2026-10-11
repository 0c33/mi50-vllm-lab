"""Correctness + timing of gfx906_skinny.gemm vs the current vLLM fp16 dispatch."""
import sys

import torch

from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

torch.ops.load_library(sys.argv[1])
dev = torch.device("cuda:0")
torch.manual_seed(0)
VARIANTS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,1,3,4").split(",")]
NS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,3,4,7,8").split(",")]


def timeit(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
    return t[len(t) // 2]


big = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
for _ in range(200):
    big @ big
torch.cuda.synchronize()

SHAPES = {
    "lm_head": (124160, 5120), "dr_qkv": (3072, 5120), "dr_o": (5120, 2048),
    "dr_gate_up": (17408, 5120), "dr_down": (5120, 8704), "dr_fc": (5120, 25600),
    "dr_ctx_kv": (5120, 5120), "dr_kproj": (1280, 5120),
}
print("shape            n  dispatch | " + " | ".join(f"v{v:<6d}" for v in VARIANTS) + " | max rel err | GB/s best")
for name, (n_out, k) in SHAPES.items():
    w = torch.randn(n_out, k, device=dev, dtype=torch.float16) * 0.02
    for n in NS:
        x = torch.randn(n, k, device=dev, dtype=torch.float16)
        ref = (x.float() @ w.float().T)
        errs, ts = [], []
        for v in VARIANTS:
            y = torch.ops.gfx906_skinny.gemm(x, w, v)
            errs.append(((y.float() - ref).abs().max() / ref.abs().max()).item())
            ts.append(timeit(lambda: torch.ops.gfx906_skinny.gemm(x, w, v)))
        td = timeit(lambda: rocm_unquantized_gemm_impl(x, w, None))
        gbs = n_out * k * 2 / min(ts) / 1e3
        print(f"{name:10s} {n_out:6d} {n:2d} {td:9.1f} | " + " | ".join(f"{t:7.1f}" for t in ts)
              + f" | {max(errs):.2e} | {gbs:6.0f}")
    del w
