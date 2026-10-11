#!/usr/bin/env python3
"""Decode t/s + acceptance after a long prompt (code-reading task, chat template)."""
import json, re, sys, time, urllib.request
URL = "http://localhost:8000"
def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    g = lambda n: float(re.search(rf"^vllm:spec_decode_{n}_total\{{[^}}]*\}} (\S+)", t, re.M).group(1))
    return g("num_drafts"), g("num_draft_tokens"), g("num_accepted_tokens")
src = open(sys.argv[1]).read()
target_chars = int(sys.argv[2])
body_txt = (src * (target_chars // len(src) + 1))[:target_chars]
prompt = f"Here is some Python source code:\n\n```python\n{body_txt}\n```\n\nExplain in detail what the code above does, function by function."
def run():
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 384,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = tl = None; usage = None
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            c = json.loads(line[6:])
            if c.get("usage"): usage = c["usage"]
            if c.get("choices"):
                now = time.perf_counter(); tf = tf or now; tl = now
    return usage, tf - t0, (usage["completion_tokens"] - 1) / (tl - tf)
m0 = metrics(); u, ttft, tps = run(); m1 = metrics()
d = [b - a for a, b in zip(m0, m1)]
print(f"prompt {u['prompt_tokens']:6d} tok  TTFT {ttft:6.2f}s  decode {tps:6.2f} t/s  accept len {1 + d[2] / d[0]:.2f}  draft acc {d[2] / d[1]:.1%}")
