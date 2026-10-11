# SPDX-License-Identifier: Apache-2.0
"""Opt-in GPU-time breakdown of decode steps (VLLM_GFX906_STEP_TIMING=1).

CUDA events around named regions; harvested only once complete (no syncs).
Logs per-step averages from global rank 0 every LOG_EVERY steps.
"""
import collections
import contextlib
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("VLLM_GFX906_STEP_TIMING", "0") == "1"
LOG_EVERY = int(os.environ.get("VLLM_GFX906_STEP_TIMING_EVERY", "200"))


class _Timer:
    def __init__(self):
        self.pending = collections.deque()
        self.tot = collections.defaultdict(float)
        self.cnt = collections.defaultdict(int)
        self.steps = 0
        self._rank0 = None

    def rank0(self) -> bool:
        if self._rank0 is None:
            self._rank0 = (not torch.distributed.is_initialized()
                           or torch.distributed.get_rank() == 0)
        return self._rank0

    @contextlib.contextmanager
    def region(self, name: str):
        if not ENABLED or not self.rank0() or torch.cuda.is_current_stream_capturing():
            yield
            return
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            yield
        finally:
            e.record()
            self.pending.append((name, s, e))

    def step_done(self):
        if not ENABLED or not self.rank0():
            return
        self.steps += 1
        while self.pending and self.pending[0][2].query():
            name, s, e = self.pending.popleft()
            self.tot[name] += s.elapsed_time(e)
            self.cnt[name] += 1
        if self.steps % LOG_EVERY == 0 and self.cnt:
            n = max(self.cnt.get("sample_tokens", 1), 1)
            parts = ", ".join(f"{k}={self.tot[k] / n:.2f}ms(x{self.cnt[k] / n:.2f})"
                              for k in sorted(self.tot))
            logger.info("gfx906 step timing over %d steps (per step): %s", n, parts)
            self.tot.clear()
            self.cnt.clear()


TIMER = _Timer()


def wrap(fn, name):
    def inner(*args, **kwargs):
        t0 = time.perf_counter()
        with TIMER.region(name):
            out = fn(*args, **kwargs)
        if ENABLED and TIMER.rank0():
            TIMER.tot["cpu_" + name] += (time.perf_counter() - t0) * 1000
            TIMER.cnt["cpu_" + name] += 1
        if name == "sample_tokens":
            TIMER.step_done()
        return out
    inner.__wrapped__ = fn
    return inner
