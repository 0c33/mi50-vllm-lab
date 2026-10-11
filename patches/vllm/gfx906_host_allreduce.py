# SPDX-License-Identifier: Apache-2.0
"""TP=2 all-reduce through pinned host memory for gfx906 boxes without GPU P2P.

RCCL falls back to host shared memory plus a proxy thread on such hosts and
costs ~75 us per call regardless of size; spec-decode verify steps issue ~140
small all-reduces. The kernel (gfx906_skinny.so, hostar.cu) writes each rank's
slice into a shared hipHostRegister'ed buffer and adds the peer's slice after a
flag handshake: ~28 us for 80 KB, bit-identical to RCCL.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class Gfx906HostAllReduce:
    def __init__(self, cpu_group: ProcessGroup, rank: int, ranks: list[int], lib: str):
        torch.ops.load_library(lib)
        self.rank = rank
        self.max_bytes = torch.ops.gfx906_skinny.hostar_max_bytes()
        name = [f"/vllm_gfx906_hostar_{os.getpid()}_{ranks[0]}" if rank == 0 else None]
        dist.broadcast_object_list(name, src=ranks[0], group=cpu_group)
        if rank == 0:
            self.handle = torch.ops.gfx906_skinny.hostar_create(name[0], self.max_bytes, True)
        dist.barrier(group=cpu_group)
        if rank != 0:
            self.handle = torch.ops.gfx906_skinny.hostar_create(name[0], self.max_bytes, False)
        dist.barrier(group=cpu_group)
        if rank == 0:
            torch.ops.gfx906_skinny.hostar_unlink(name[0])
        logger.info_once(
            "gfx906 host-memory all-reduce enabled for TP=2 (<= %d bytes)", self.max_bytes
        )

    def should_use(self, t: torch.Tensor) -> bool:
        if t.dtype not in (torch.float16, torch.float32) or not t.is_contiguous():
            return False
        nbytes = t.numel() * t.element_size()
        return 0 < nbytes <= self.max_bytes and nbytes % 16 == 0

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ops.gfx906_skinny.hostar_allreduce(t, self.handle, self.rank)
