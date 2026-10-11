"""Exact float64 reference for the Exllama-shuffled 4-bit layout vs both kernels."""
import sys
import torch
from vllm import _custom_ops as ops
torch.ops.load_library(sys.argv[1])
dev = torch.device("cuda:0"); torch.manual_seed(1)
K, N, G = 5120, 1024, 32
qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), device=dev, dtype=torch.int32)
qz = torch.randint(-2**31, 2**31 - 1, (K // G, N // 8), device=dev, dtype=torch.int32)
sc = torch.rand(K // G, N, device=dev, dtype=torch.float16) * 0.02 + 0.001
u = qw.to(torch.int64) & 0xFFFFFFFF
w = torch.empty(K, N, dtype=torch.float64, device=dev)
for i in range(4):  # shuffled word: nibble i -> k=2i, nibble 4+i -> k=2i+1
    w[2 * i::8] = ((u >> (4 * i)) & 0xF).double()
    w[2 * i + 1::8] = ((u >> (16 + 4 * i)) & 0xF).double()
zu = qz.to(torch.int64) & 0xFFFFFFFF
z = torch.stack([(zu >> (4 * c)) & 0xF for c in range(8)], dim=-1).reshape(K // G, N).double()
deq = (w - z.repeat_interleave(G, 0)) * sc.double().repeat_interleave(G, 0)
for m in (1, 4, 8):
    x = torch.randn(m, K, device=dev, dtype=torch.float16) * 0.5
    ref = x.double() @ deq
    ex = ops.gptq_gemm(x, qw, qz, sc, True, True, 4).double()
    ours = torch.ops.gfx906_skinny.w4gemm(x, qw, qz, sc, 128, 14).double()
    scale = ref.abs().max()
    print(f"M={m}: exllama max rel err {((ex - ref).abs().max() / scale).item():.2e} | "
          f"new kernel {((ours - ref).abs().max() / scale).item():.2e}")
