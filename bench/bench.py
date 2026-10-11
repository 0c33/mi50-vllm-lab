#!/usr/bin/env python3
"""Single-request decode t/s + MTP acceptance against the local vLLM server.

Decode t/s = (completion_tokens - 1) / (t_last_chunk - t_first_chunk), so prefill
is excluded. Acceptance comes from /metrics deltas around each request.
Usage: bench.py [max_tokens] [reps] [temperature]
"""
import json
import re
import sys
import time
import urllib.request

URL = "http://localhost:8000"
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

PROMPTS = {
    "prose": "Explain in detail how a modern CPU executes instructions out of order, "
    "covering the reorder buffer, register renaming and branch prediction.",
    "code": "Write a complete, well-commented Python module implementing an LRU cache "
    "class with get/put, a decorator version, and unit tests using unittest.",
    "think": "A train leaves at 14:05 travelling 84 km/h; a second leaves the same "
    "station at 14:50 at 112 km/h on the same track. When and where does the second "
    "catch the first? Show your reasoning.",
}


def metrics():
    text = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    out = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(rf"^vllm:spec_decode_{name}_total\{{[^}}]*\}} (\S+)", text, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def run(prompt, thinking):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
    n = usage["completion_tokens"]
    return n, t_first - t0, (n - 1) / (t_last - t_first)


print(f"max_tokens={MAX_TOKENS} reps={REPS} temperature={TEMP}")
run("Say hi.", False)  # warmup
rows = []
for name, prompt in PROMPTS.items():
    for rep in range(REPS):
        before = metrics()
        n, ttft, tps = run(prompt, thinking=(name == "think"))
        after = metrics()
        d = {k: after[k] - before[k] for k in after}
        acc = d["num_accepted_tokens"] / d["num_draft_tokens"] if d["num_draft_tokens"] else 0
        mal = 1 + d["num_accepted_tokens"] / d["num_drafts"] if d["num_drafts"] else 0
        rows.append((tps, acc, mal))
        print(f"{name:6s} rep{rep}: {n} tok  TTFT {ttft:5.2f}s  decode {tps:6.2f} t/s  "
              f"draft acceptance {acc:5.1%}  mean accept len {mal:4.2f}")
k = len(rows)
print(f"MEAN: decode {sum(r[0] for r in rows)/k:.2f} t/s  acceptance "
      f"{sum(r[1] for r in rows)/k:.1%}  mean accept len {sum(r[2] for r in rows)/k:.2f}")
