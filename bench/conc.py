#!/usr/bin/env python3
"""Concurrency probe: N simultaneous streaming chat requests against the local server.

Usage: conc.py <n_parallel> [max_tokens] [long_prompt_tokens]
Prints per-request TTFT and decode t/s, aggregate output t/s over the wall time,
and the peak vllm:num_requests_running seen while the requests were in flight.
With long_prompt_tokens > 0, request 0 gets a long code-reading prompt (built from
distinct vLLM source files) while the others are short: the agent case of one long
session plus quick side requests.
"""
import glob
import os
import json
import re
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000"
N = int(sys.argv[1])
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string and returns seconds. Include tests.",
    "Explain how a B-tree insert works, step by step, including node splits.",
    "Write a bash script that rotates log files older than 7 days and compresses them.",
    "Compare TCP congestion control algorithms CUBIC and BBR in detail.",
    "Write a SQL schema for a library system with books, members and loans, plus three useful queries.",
    "Explain the CAP theorem with concrete examples of real databases.",
    "Write a Go HTTP middleware that rate-limits per client IP with a token bucket.",
    "Describe how a garbage collector with generational collection works.",
]


def long_prompt(tokens):
    parts, size, budget = [], 0, int(tokens * 3.6)
    for f in sorted(glob.glob(os.environ.get("VLLM_SRC", "vllm-src") + "/vllm/**/*.py", recursive=True)):
        src = open(f).read()
        if len(src) < 2000 or size + len(src) > budget:
            continue
        parts.append(f"\n\n===== FILE: {f} =====\n{src}")
        size += len(src)
    # unique first line: a fresh prompt every run, so the prefix cache cannot hide prefill
    return f"Session {time.time_ns()}\n" + "".join(parts) + "\n\nSummarize what the files above do, file by file."


def running():
    t = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    m = re.search(r"^vllm:num_requests_running\{[^}]*\} (\S+)", t, re.M)
    return float(m.group(1)) if m else 0.0


def chat(i, prompt, out):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = tl = None; usage = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            if c.get("choices"):
                now = time.perf_counter(); tf = tf or now; tl = now
    n = usage["completion_tokens"]
    out[i] = {"prompt": usage["prompt_tokens"], "n": n, "t0": t0, "tf": tf, "tl": tl,
              "ttft": tf - t0, "tps": (n - 1) / (tl - tf) if tl > tf else 0.0}


prompts = [PROMPTS[i % len(PROMPTS)] for i in range(N)]
if LONG:
    prompts[0] = long_prompt(LONG)
out, peak, done = {}, [0.0], threading.Event()


def watch():
    while not done.is_set():
        peak[0] = max(peak[0], running())
        time.sleep(0.5)


w = threading.Thread(target=watch); w.start()
threads = [threading.Thread(target=chat, args=(i, p, out)) for i, p in enumerate(prompts)]
start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - start
done.set(); w.join()
for i in sorted(out):
    r = out[i]
    print(f"req{i}: prompt {r['prompt']:6d} tok  TTFT {r['ttft']:7.2f}s  decode {r['tps']:6.2f} t/s  "
          f"done at {r['tl'] - start:7.1f}s", flush=True)
total = sum(r["n"] for r in out.values())
print(f"N={N}: {total} output tokens in {wall:.1f}s = {total / wall:.1f} t/s aggregate, "
      f"peak running {peak[0]:.0f}", flush=True)
