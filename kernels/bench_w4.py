"""Correctness + timing: gfx906_skinny.w4gemm vs ops.gptq_gemm (Exllama, v2 zeros)."""
import sys
import torch
from vllm import _custom_ops as ops
torch.ops.load_library(sys.argv[1])
CFGS = [tuple(int(v) for v in c.split("x")) for c in (sys.argv[2] if len(sys.argv) > 2 else "128x4,256x4,64x4,128x2,256x2").split(",")]
NS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8").split(",")]
dev = torch.device("cuda:0"); torch.manual_seed(0)
def timeit(fn, iters=100):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev: s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev); return t[len(t) // 2]
big = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
for _ in range(200): big @ big
torch.cuda.synchronize()
W4 = {"gdn_qkvz": (5120, 8192), "gdn_out": (3072, 5120), "attn_qkv": (5120, 7168),
      "attn_o": (3072, 5120), "mlp_gate_up": (5120, 17408), "mlp_down": (8704, 5120)}
G = 32
print("shape                 n   exllama | " + " | ".join(f"kw{c[0]}w{c[1]:<3d}" for c in CFGS) + " | rel err vs exllama")
for name, (k, n_out) in W4.items():
    qw = torch.randint(-2**31, 2**31 - 1, (k // 8, n_out), device=dev, dtype=torch.int32)
    qz = torch.randint(-2**31, 2**31 - 1, (k // G, n_out // 8), device=dev, dtype=torch.int32)
    sc = (torch.rand(k // G, n_out, device=dev, dtype=torch.float16) * 0.02 + 0.001)
    for n in NS:
        x = torch.randn(n, k, device=dev, dtype=torch.float16) * 0.5
        ref = ops.gptq_gemm(x, qw, qz, sc, True, True, 4).float()
        te = timeit(lambda: ops.gptq_gemm(x, qw, qz, sc, True, True, 4))
        ts, errs = [], []
        for kw, wv in CFGS:
            try:
                y = torch.ops.gfx906_skinny.w4gemm(x, qw, qz, sc, kw, wv).float()
                errs.append(((y - ref).abs().max() / ref.abs().max()).item())
                ts.append(timeit(lambda: torch.ops.gfx906_skinny.w4gemm(x, qw, qz, sc, kw, wv)))
            except Exception as e:
                errs.append(float("nan")); ts.append(float("nan"))
        print(f"{name:12s} {str((k, n_out)):13s} {n:2d} {te:8.1f} | " + " | ".join(f"{t:9.1f}" for t in ts) + f" | {max(e for e in errs if e == e) if any(e == e for e in errs) else float('nan'):.2e}")
