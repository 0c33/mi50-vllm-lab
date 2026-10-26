#!/usr/bin/env python3
"""Agent-turn TTFT: how much of the previous turn the prefix cache really reuses.

turns (default):
  Turn 1: a long prompt (repository text, ~CTX tokens) + question; greedy reply of REPLY tokens.
  Turn 2: turn 1 + its reply (assistant message) + a new ~TOOL-token user message ("tool output").
  Turn 3: turn 2 + its reply + another new user message.
  Turn 4: identical repeat of turn 3's request (pure cache hit).
--siblings:
  X, Y, Z = the same ~CTX-token prefix + three different questions (agents sharing a system
  prompt / tool list).
For each request: TTFT, prompt tokens, prefix-cache hit tokens (from /metrics deltas) and the
reply text, saved to turn-<tag>.json / sib-<tag>.json so that two server configurations can be
compared for identical greedy outputs.

Usage: VLLM_SRC=/path/to/vllm turnbench.py [--siblings] <tag> <ctx_tokens> [reply_tokens] [tool_tokens]
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request

URL = "http://localhost:8000"
args = [a for a in sys.argv[1:] if a != "--siblings"]
SIBLINGS = "--siblings" in sys.argv[1:]
TAG, CTX = args[0], int(args[1])
REPLY = int(args[2]) if len(args) > 2 else 200
TOOL = int(args[3]) if len(args) > 3 else 600
SRC = os.environ.get("VLLM_SRC", "vllm-src")  # text corpus: a vLLM source checkout
FILES = sorted(glob.glob(f"{SRC}/vllm/**/*.py", recursive=True))


def text(tokens, start=0):
    parts, size = [], 0
    for f in FILES[start:]:
        src = open(f).read()
        parts.append(f"\n===== {f} =====\n{src}")
        size += len(src)
        if size > tokens * 3.4:
            break
    return "".join(parts)[: int(tokens * 3.4)]


def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    out = {}
    for name in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.search(rf"^vllm:{name}_total\{{[^}}]*\}} (\S+)", t, re.M)
        out[name] = float(m.group(1)) if m else float("nan")
    return out


def chat(messages, max_tokens):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m0 = metrics()
    t0 = time.perf_counter()
    tf, usage, out = None, None, []
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                tf = tf or time.perf_counter()
                out.append(ch.get("delta", {}).get("content") or "")
    t_end = time.perf_counter()
    m1 = metrics()
    hits = m1["prefix_cache_hits"] - m0["prefix_cache_hits"]
    return {"ttft": tf - t0, "total": t_end - t0, "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"], "cache_hit_tokens": hits,
            "recomputed": usage["prompt_tokens"] - hits, "text": "".join(out)}


res = []
if SIBLINGS:
    prefix = "System prompt / tools:\n" + text(CTX, 30)
    for q in ("Which module above handles configuration?", "List three functions above that do I/O.",
              "What is the purpose of the first file above?"):
        res.append(chat([{"role": "user", "content": prefix + "\n\n" + q}], REPLY))
    for i, r in enumerate(res):
        print(f"[{TAG}] sibling {'XYZ'[i]}: prompt {r['prompt_tokens']:6d}  cache hit {r['cache_hit_tokens']:7.0f}  "
              f"recomputed {r['recomputed']:6.0f}  TTFT {r['ttft']:6.2f}s", flush=True)
    json.dump(res, open(f"sib-{TAG}.json", "w"), indent=1)
else:
    msgs = [{"role": "user", "content": "Repository files:\n" + text(CTX) +
             "\n\nList the five most important classes above and what each does."}]
    r = chat(msgs, REPLY); res.append(r)
    for start in (400, 420):
        msgs = msgs + [{"role": "assistant", "content": r["text"]},
                       {"role": "user", "content": "Tool output:\n" + text(TOOL, start) + "\n\nWhat does this new file do? Be brief."}]
        r = chat(msgs, REPLY); res.append(r)
    r = chat(msgs, REPLY); res.append(r)  # identical repeat of turn 3
    for i, r in enumerate(res, 1):
        print(f"[{TAG}] turn {i}: prompt {r['prompt_tokens']:6d}  cache hit {r['cache_hit_tokens']:7.0f}  "
              f"recomputed {r['recomputed']:6.0f}  TTFT {r['ttft']:6.2f}s  total {r['total']:6.2f}s", flush=True)
    json.dump(res, open(f"turn-{TAG}.json", "w"), indent=1)
