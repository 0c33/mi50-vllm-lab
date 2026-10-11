#!/usr/bin/env python3
"""Long-context agentic probe: codebase dump + task, needle check, follow-up turn.

Usage: VLLM_SRC=/path/to/vllm agentctx.py <target_prompt_tokens> [max_tokens] [chars_per_token]
Builds a prompt from many distinct vLLM source files (not a repeated file, which
inflates speculative acceptance), hides a needle fact near the start, asks a
task at the end, then sends a follow-up turn on the same conversation (the
agent-loop pattern that prefix caching should make cheap).
"""
import glob
import os
import json
import re
import sys
import time
import urllib.request

URL = "http://localhost:8000"
target_tokens = int(sys.argv[1])
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
CHARS_PER_TOKEN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.6
NEEDLE = "The internal codename of this project is ORCHID-7731-TANGO."

SRC = os.environ.get("VLLM_SRC", "vllm-src")
files = sorted(glob.glob(f"{SRC}/vllm/**/*.py", recursive=True))
parts, size = [f"# Repository dump\n\n# NOTE: {NEEDLE}\n"], 0
budget = int(target_tokens * CHARS_PER_TOKEN)
for f in files:
    try:
        src = open(f).read()
    except OSError:
        continue
    if len(src) < 2000:
        continue
    chunk = f"\n\n===== FILE: {os.path.relpath(f, SRC)} =====\n{src}"
    if size + len(chunk) > budget:
        continue
    parts.append(chunk)
    size += len(chunk)
dump = "".join(parts)
print(f"dump: {len(dump)} chars from {len(parts) - 1} files", flush=True)
task = ("\n\nYou are a coding agent working in this repository. First, state the internal "
        "codename mentioned in the NOTE at the top of the dump. Then explain how speculative "
        "decoding proposals are verified in this codebase, naming the relevant files.")


def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    def g(n):
        m = re.search(rf"^vllm:spec_decode_{n}_total\{{[^}}]*\}} (\S+)", t, re.M)
        return float(m.group(1)) if m else 0.0
    return g("num_drafts"), g("num_accepted_tokens")


def chat(messages):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m0 = metrics()
    t0 = time.perf_counter(); tf = tl = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            if c.get("choices"):
                d = c["choices"][0].get("delta", {}).get("content") or ""
                text.append(d)
                now = time.perf_counter(); tf = tf or now; tl = now
    m1 = metrics()
    n = usage["completion_tokens"]
    drafts, acc = m1[0] - m0[0], m1[1] - m0[1]
    return {"prompt": usage["prompt_tokens"], "ttft": tf - t0,
            "tps": (n - 1) / (tl - tf) if tl and tl > tf else 0.0,
            "accept_len": 1 + acc / drafts if drafts else float("nan"), "text": "".join(text)}


msgs = [{"role": "user", "content": dump + task}]
r1 = chat(msgs)
needle_ok = "ORCHID-7731-TANGO" in r1["text"]
print(f"turn1: prompt {r1['prompt']:6d} tok  TTFT {r1['ttft']:7.1f}s ({r1['prompt'] / r1['ttft']:6.1f} tok/s prefill)  "
      f"decode {r1['tps']:6.2f} t/s  accept {r1['accept_len']:.2f}  needle {'OK' if needle_ok else 'MISSED'}", flush=True)
msgs += [{"role": "assistant", "content": r1["text"]},
         {"role": "user", "content": "Good. Now list three concrete risks in that verification code."}]
r2 = chat(msgs)
print(f"turn2: prompt {r2['prompt']:6d} tok  TTFT {r2['ttft']:7.1f}s  decode {r2['tps']:6.2f} t/s  "
      f"accept {r2['accept_len']:.2f}   (prefix-cached follow-up)", flush=True)
print("answer head:", r1["text"][:160].replace("\n", " "))
