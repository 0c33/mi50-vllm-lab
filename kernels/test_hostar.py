import os, sys, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
torch.ops.load_library(sys.argv[1])
name = f"/hostar_test_{os.getppid()}"
if r == 0: h = torch.ops.gfx906_skinny.hostar_create(name, 1 << 20, True)
dist.barrier()
if r == 1: h = torch.ops.gfx906_skinny.hostar_create(name, 1 << 20, False)
dist.barrier()
if r == 0: torch.ops.gfx906_skinny.hostar_unlink(name)
g = torch.Generator(device="cuda").manual_seed(100 + r)
bad = 0
for it in range(3000):
    n = [8 * 5120, 5120, 2 * 5120, 64, 8 * 8192][it % 5]
    dt = torch.float16 if it % 7 else torch.float32
    if n * (2 if dt == torch.float16 else 4) > torch.ops.gfx906_skinny.hostar_max_bytes(): dt = torch.float16
    x = torch.randn(n, device="cuda", dtype=dt, generator=g) * (10 ** (it % 4 - 1))
    ref = x.clone(); dist.all_reduce(ref)
    y = torch.ops.gfx906_skinny.hostar_allreduce(x, h, r)
    if not torch.equal(y, ref): bad += 1
torch.cuda.synchronize()
print(f"rank {r}: correctness vs RCCL over 3000 random calls: {bad} mismatches", flush=True)
x = torch.randn(8 * 5120, device="cuda", dtype=torch.float16)
for name_, fn in (("rccl", lambda: dist.all_reduce(x)), ("hostar", lambda: torch.ops.gfx906_skinny.hostar_allreduce(x, h, r))):
    for _ in range(50): fn()
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(1000): fn()
    torch.cuda.synchronize()
    if r == 0: print(f"{name_:7s} 80 KB eager: {(time.perf_counter() - t0) / 1000 * 1e6:6.1f} us/op", flush=True)
# CUDA graph: 200 chained all-reduces, verify against RCCL chain
s = torch.cuda.Stream()
base = torch.randn(8 * 5120, device="cuda", dtype=torch.float16, generator=g) * 0.01
buf = base.clone()
with torch.cuda.stream(s):
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        y = buf
        for _ in range(200): y = torch.ops.gfx906_skinny.hostar_allreduce(y, h, r) * 0.5
torch.cuda.synchronize(); dist.barrier()
t0 = time.perf_counter()
for _ in range(5): gr.replay()
torch.cuda.synchronize()
el = (time.perf_counter() - t0) / 1000 * 1e6
ref = base.clone()
for _ in range(200): dist.all_reduce(ref); ref = ref * 0.5
same = torch.equal(y, ref)
if r == 0: print(f"hostar in CUDA graph: {el:6.1f} us/op (incl. a scale kernel), result == RCCL chain: {same}", flush=True)
dist.barrier(); dist.destroy_process_group()
