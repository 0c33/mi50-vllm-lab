#!/usr/bin/env python3
"""Correctness + speed of _gfx906_pth_dec.decode (split-K decode attention over vLLM's int8
per-token-head KV cache) against a float64 reference and vLLM's Triton unified attention
(3D decode path, as configured in the server), at Qwen3.8-27B TP=2 rank shapes.

Usage (inside the -opt image, one GPU, under bench/safe-run.sh): test_dec.py [quick|speed|all] [nsplit,...]
Env PTH_DEC_SO overrides the extension path.
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

so = os.environ.get("PTH_DEC_SO", "/opt/gfx906/_gfx906_pth_dec.so")
spec = importlib.util.spec_from_loader("_gfx906_pth_dec", importlib.machinery.ExtensionFileLoader("_gfx906_pth_dec", so))
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)

dev = "cuda"
NQ, NKV, D, BS, PAD = 12, 2, 256, 1584, 4
QPK = NQ // NKV
MAXT, NSEG, THRESH = 64, 128, 16
PART_O = torch.empty((MAXT, NQ, NSEG, D), dtype=torch.float32, device=dev)
PART_M = torch.empty((MAXT, NQ, NSEG), dtype=torch.float32, device=dev)
PART_L = torch.empty((MAXT, NQ, NSEG), dtype=torch.float32, device=dev)


def make_case(ctxs, qlens, seed=0, nhd=False, q_noncontig=False, scale_hi=0.02):
    """ctxs[b] cached tokens + qlens[b] new tokens per sequence (qlens may contain 0 = padding)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    seqs = [c + q for c, q in zip(ctxs, qlens)]
    per = [max(1, (s + BS - 1) // BS) for s in seqs]
    nblk = sum(per) + 3
    if nhd:
        kv = torch.randint(-128, 128, (nblk, BS, NKV, 2 * (D + PAD)), dtype=torch.int8, device=dev, generator=g).transpose(1, 2)
    else:
        kv = torch.randint(-128, 128, (nblk, NKV, BS, 2 * (D + PAD)), dtype=torch.int8, device=dev, generator=g)
    f32 = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = [x // 4 for x in kv.stride()]
    ks = torch.as_strided(f32, (nblk, BS, NKV), (st[0], st[2], st[1]), D // 4)
    vs = torch.as_strided(f32, (nblk, BS, NKV), (st[0], st[2], st[1]), (2 * D + PAD) // 4)
    ks.copy_(torch.rand(ks.shape, device=dev, generator=g) * scale_hi + 0.002)
    vs.copy_(torch.rand(vs.shape, device=dev, generator=g) * scale_hi + 0.002)
    kc, vc = kv.transpose(1, 2).split(D + PAD, dim=-1)
    perm = torch.randperm(nblk, device=dev, generator=g).to(torch.int32)
    maxb = max(per) + 2
    bt = torch.zeros((len(seqs), maxb), dtype=torch.int32, device=dev)
    o = 0
    for b, n in enumerate(per):
        bt[b, :n] = perm[o:o + n]
        o += n
    T = sum(qlens)
    if q_noncontig:
        big = torch.randn((T, NQ + 4, D), dtype=torch.float16, device=dev, generator=g)
        q = big[:, 2:2 + NQ]
    else:
        q = torch.randn((T, NQ, D), dtype=torch.float16, device=dev, generator=g)
    qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    sl = torch.tensor(seqs, dtype=torch.int32, device=dev)
    return dict(q=q, kc=kc, vc=vc, ks=ks, vs=vs, bt=bt, qsl=qsl, sl=sl, ctxs=ctxs, qlens=qlens, seqs=seqs,
                T=T, B=len(seqs), kv=kv)


def reference(c):
    out = torch.zeros((c["T"], NQ, D), dtype=torch.float64, device=dev)
    for b in range(c["B"]):
        ql, seq = c["qlens"][b], c["seqs"][b]
        if ql == 0:
            continue
        blocks = c["bt"][b, : (seq + BS - 1) // BS].long()
        k = c["kc"][blocks].reshape(-1, NKV, D + PAD)[:seq, :, :D].double()
        v = c["vc"][blocks].reshape(-1, NKV, D + PAD)[:seq, :, :D].double()
        k = k * c["ks"][blocks].reshape(-1, NKV)[:seq, :, None].double()
        v = v * c["vs"][blocks].reshape(-1, NKV)[:seq, :, None].double()
        q0 = c["qsl"][b].item()
        q = c["q"][q0:q0 + ql].double()
        pos = seq - ql + torch.arange(ql, device=dev)
        mask = torch.arange(seq, device=dev)[None, :] <= pos[:, None]
        for h in range(NQ):
            s = (q[:, h] @ k[:, h // QPK].T) / math.sqrt(D)
            out[q0:q0 + ql, h] = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v[:, h // QPK]
    return out


def run_dec(c, out, nsplit, qmax=None):
    qmax = qmax or max(c["qlens"])
    pd.decode(c["q"], c["kc"], c["vc"], c["ks"], c["vs"], c["bt"], c["sl"], c["qsl"], out, 1 / math.sqrt(D),
              PART_O, PART_M, PART_L, c["B"], qmax, nsplit)


def run_triton(c, out):
    unified_attention(q=c["q"], k=c["kc"], v=c["vc"], out=out, cu_seqlens_q=c["qsl"], max_seqlen_q=max(c["qlens"]),
                      seqused_k=c["sl"], max_seqlen_k=max(c["seqs"]), softmax_scale=1 / math.sqrt(D), causal=True,
                      window_size=(-1, -1), block_table=c["bt"], softcap=0, q_descale=None, k_descale=None,
                      v_descale=None, seq_threshold_3D=THRESH, num_par_softmax_segments=NSEG,
                      softmax_segm_output=PART_O, softmax_segm_max=PART_M, softmax_segm_expsum=PART_L,
                      kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, k_scale_cache=c["ks"], v_scale_cache=c["vs"])


def timed(fn, min_s=0.4):
    fn(); torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < min_s or n < 5:
        fn(); n += 1
    torch.cuda.synchronize()
    return (time.time() - t0) / n


mode = sys.argv[1] if len(sys.argv) > 1 else "all"
splits = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["64"])]
ok = True
if mode in ("quick", "all"):
    print("correctness vs float64 (max abs err; |ref| ~ 0.01-0.1)", flush=True)
    cases = [([0], [1], False, False), ([5], [4], False, False), ([255], [1], False, False),
             ([252], [4], False, False), ([1000], [4], True, False), ([2 * BS + 17], [4], False, True),
             ([20000], [4], False, False), ([20000, 300, 7], [4, 4, 4], True, False),
             ([5000, 70000], [1, 1], False, False), ([3000, 9, 40000, 1], [4, 1, 3, 2], False, True),
             ([4000, 0, 0], [4, 0, 0], False, False), ([BS - 4], [4], False, False), ([131072], [4], False, False),
             # last split holds only the final 2 keys: the first query rows see none of them there
             ([510], [4], False, False), ([1022, 3000], [4, 4], True, False)]
    for ctxs, qlens, nhd, nc in cases:
        c = make_case(ctxs, qlens, seed=sum(ctxs) + len(ctxs), nhd=nhd, q_noncontig=nc)
        ref = reference(c)
        mag = ref.abs().mean().item()
        tri = torch.zeros((c["T"], NQ, D), dtype=torch.float16, device=dev)
        run_triton(c, tri)
        e_tri = (tri.double() - ref).abs().max().item()
        for ns in sorted(set(splits + [1, 7, 128])):
            out = torch.full((c["T"], NQ, D), float("nan"), dtype=torch.float16, device=dev)
            run_dec(c, out, ns)
            torch.cuda.synchronize()
            valid = torch.ones(c["T"], dtype=torch.bool, device=dev)
            e = (out.double() - ref)[valid].abs().max().item()
            good = math.isfinite(e) and e < 4e-3
            ok &= good
            print(f"  ctx {str(ctxs):24s} q {str(qlens):14s} {'NHD' if nhd else 'HND'}{' qnc' if nc else '    '} "
                  f"nsplit {ns:3d}: pth_dec {e:.1e}  triton {e_tri:.1e}  |ref| {mag:.3f}  {'OK' if good else 'FAIL'}",
                  flush=True)
        del c
    print("ALL OK" if ok else "SOME FAILED", flush=True)
    if mode == "quick":
        sys.exit(0 if ok else 1)

print("speed: one layer, one rank (ms per call); Triton = server config (3D, 128 segments)", flush=True)
for B, ql in ((1, 4), (2, 4), (4, 4), (1, 1), (2, 1)):
    for ctx in (4096, 16384, 32768, 65536, 131072):
        if B * ctx > 270000:
            continue
        c = make_case([ctx] * B, [ql] * B, seed=1)
        out = torch.empty((c["T"], NQ, D), dtype=torch.float16, device=dev)
        t_tri = timed(lambda: run_triton(c, out))
        cells = []
        for ns in splits:
            t = timed(lambda: run_dec(c, out, ns))
            cells.append(f"ns{ns} {t * 1e3:6.3f}ms x{t_tri / t:5.2f}")
        fl = 4 * D * NQ * B * ql * (ctx + ql)
        print(f"  B={B} QL={ql} ctx {ctx // 1024:4d}k: triton {t_tri * 1e3:6.3f}ms ({fl / t_tri / 1e12:4.2f} TF)  "
              + "  ".join(cells), flush=True)
        del c
