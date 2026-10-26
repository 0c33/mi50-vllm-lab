#!/usr/bin/env python3
"""Correctness + speed of _gfx906_fa_pth.forward_pth (gfx906 FA reading vLLM's int8
per-token-head KV cache in place) against a float64 reference and vLLM's Triton
unified attention (the serving path), at Qwen3.8-27B TP=2 rank shapes.
Usage (inside the -opt image, one GPU, under bench/safe-run.sh): test_pth.py [quick]
Env PTH_FA_SO overrides the extension path.
"""
import importlib.machinery
import importlib.util
import math
import os
import sys
import time

import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

so = os.environ.get("PTH_FA_SO", "/opt/gfx906/_gfx906_fa_pth.so")
spec = importlib.util.spec_from_loader("_gfx906_fa_pth", importlib.machinery.ExtensionFileLoader("_gfx906_fa_pth", so))
pth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pth)

dev = "cuda"
NQ, NKV, D, BS, PAD = 12, 2, 256, 1584, 4
QPK = NQ // NKV


def make_cache(seq, seed, nhd=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    nblk = (seq + BS - 1) // BS + 2
    shape = (nblk, NKV, BS, 2 * (D + PAD))
    if nhd:  # physical NHD: (blocks, slots, heads, row) viewed as logical (blocks, heads, slots, row)
        kv = torch.randint(-127, 128, (nblk, BS, NKV, 2 * (D + PAD)), dtype=torch.int8, device=dev, generator=g).transpose(1, 2)
    else:
        kv = torch.randint(-127, 128, shape, dtype=torch.int8, device=dev, generator=g)
    f32 = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = [x // 4 for x in kv.stride()]
    ks = torch.as_strided(f32, (nblk, BS, NKV), (st[0], st[2], st[1]), D // 4)
    vs = torch.as_strided(f32, (nblk, BS, NKV), (st[0], st[2], st[1]), (2 * D + PAD) // 4)
    ks.copy_(torch.rand(ks.shape, device=dev, generator=g) * 0.02 + 0.002)
    vs.copy_(torch.rand(vs.shape, device=dev, generator=g) * 0.02 + 0.002)
    kc, vc = kv.transpose(1, 2).split(D + PAD, dim=-1)  # (blocks, slots, heads, 260) views
    bt = torch.randperm(nblk, device=dev, generator=g).to(torch.int32)[None, : (seq + BS - 1) // BS].contiguous()
    return kv, kc, vc, ks, vs, bt


def triton_ref(q, kc, vc, ks, vs, bt, seq):
    out = torch.empty_like(q)
    n = q.shape[0]
    unified_attention(q=q, k=kc, v=vc, out=out,
                      cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device=dev), max_seqlen_q=n,
                      seqused_k=torch.tensor([seq], dtype=torch.int32, device=dev), max_seqlen_k=seq,
                      softmax_scale=1 / math.sqrt(D), causal=True, window_size=(-1, -1), block_table=bt,
                      softcap=0, q_descale=None, k_descale=None, v_descale=None,
                      kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, k_scale_cache=ks, v_scale_cache=vs)
    return out


def fa_pth(q, kc, vc, bt, seq):
    n = q.shape[0]
    ncols = 64 if n > 32 else 32 if n > 16 else 16 if n > 8 else 8 if n > 4 else 4 if n > 2 else 2
    sq = (n + ncols - 1) // ncols * ncols
    qp = torch.zeros((1, NQ, sq, D), dtype=torch.float32, device=dev)
    qp[0, :, :n] = q.permute(1, 0, 2)
    o = pth.forward_pth(qp, kc, vc, bt, torch.tensor([seq], dtype=torch.int32, device=dev), 1 / math.sqrt(D),
                        torch.tensor([seq - n], dtype=torch.int32, device=dev))
    return o[0, :n].to(torch.float16)


def exact(q, kc, vc, ks, vs, bt, seq):
    blocks = bt[0].long()
    k = (kc[blocks].reshape(-1, NKV, D + PAD)[:seq, :, :D].double() * ks[blocks].reshape(-1, NKV)[:seq, :, None].double())
    v = (vc[blocks].reshape(-1, NKV, D + PAD)[:seq, :, :D].double() * vs[blocks].reshape(-1, NKV)[:seq, :, None].double())
    n = q.shape[0]
    pos = seq - n + torch.arange(n, device=dev)
    mask = torch.arange(seq, device=dev)[None, :] <= pos[:, None]
    out = torch.empty((n, NQ, D), dtype=torch.float64, device=dev)
    for h in range(NQ):
        s = (q[:, h].double() @ k[:, h // QPK].T) / math.sqrt(D)
        out[:, h] = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v[:, h // QPK]
    return out


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn(); torch.cuda.synchronize()
    return (time.time() - t0) / reps


print("correctness vs float64 (max abs err; outputs are O(0.1))")
ok = True
for ctx, n, nhd in [(0, 64, False), (0, 1500, False), (1000, 777, False), (1584 * 3 - 5, 2048, False),
                    (5000, 2048, True), (20000, 33, False), (12345, 5, False)]:
    seq = ctx + n
    kv, kc, vc, ks, vs, bt = make_cache(seq, seed=ctx + n)
    q = torch.randn((n, NQ, D), dtype=torch.float16, device=dev)
    ref = exact(q, kc, vc, ks, vs, bt, seq)
    e_tri = (triton_ref(q, kc, vc, ks, vs, bt, seq).double() - ref).abs().max().item()
    e_fa = (fa_pth(q, kc, vc, bt, seq).double() - ref).abs().max().item()
    mag = ref.abs().mean().item()
    good = e_fa < 2e-2 and math.isfinite(e_fa)
    ok &= good
    print(f"  ctx {ctx:6d} q {n:5d} {'NHD' if nhd else 'HND'}: triton {e_tri:.1e}  fa_pth {e_fa:.1e}  (|ref| {mag:.3f})"
          f"  {'OK' if good else 'FAIL'}", flush=True)
    del kv, kc, vc, ks, vs, bt
print("ALL OK" if ok else "SOME FAILED", flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "quick":
    sys.exit(0 if ok else 1)

print("speed, one 2048-token chunk on top of ctx cached tokens (one layer, one rank)")
for ctx in (0, 16384, 65536, 131072):
    seq = ctx + 2048
    kv, kc, vc, ks, vs, bt = make_cache(seq, seed=1)
    q = torch.randn((2048, NQ, D), dtype=torch.float16, device=dev)
    fl = 4 * D * NQ * (2048 * ctx + 2048 * 2049 / 2)
    t_tri = timed(lambda: triton_ref(q, kc, vc, ks, vs, bt, seq), reps=2)
    t_fa = timed(lambda: fa_pth(q, kc, vc, bt, seq))
    print(f"  ctx {ctx // 1024:4d}k: triton {t_tri * 1e3:8.1f} ms ({fl / t_tri / 1e12:4.2f} TF)   "
          f"fa_pth {t_fa * 1e3:8.1f} ms ({fl / t_fa / 1e12:4.2f} TF)   x{t_tri / t_fa:4.2f}", flush=True)
    del kv, kc, vc, ks, vs, bt
