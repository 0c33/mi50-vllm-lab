#!/usr/bin/env python3
"""Decode-path quality A/B: greedy continuations with per-step top-5 logprobs.

save: for each context length, one greedy request (max_tokens N, thinking off, no EOS stop)
      on deterministic repository text; stores tokens + top-5 logprobs per step.
compare: per length, the length of the identical greedy prefix; at the first divergence
      the top-2 logprob gap of both runs (a near tie => numerical noise, not a bug); mean
      |logprob difference| of the chosen tokens over the shared prefix.

Usage: decqual.py save <out.json> <N> <tokens> [<tokens> ...]
       decqual.py compare <a.json> <b.json>
"""
import glob
import json
import os
import math
import sys
import urllib.request

URL = "http://localhost:8000"
SRC = os.environ.get("VLLM_SRC", "vllm-src")  # text corpus: a vLLM source checkout
FILES = sorted(glob.glob(f"{SRC}/vllm/**/*.py", recursive=True))


def text(tokens):
    parts, size = [f"Context benchmark {tokens}: repository files\n"], 0
    for f in FILES:
        src = open(f).read()
        parts.append(f"\n===== {f} =====\n{src}")
        size += len(src)
        if size > tokens * 3.4:
            break
    return "".join(parts)[: int(tokens * 3.4)] + "\n\nExplain in detail what the code above does, file by file."


def ask(tokens, n):
    body = {"model": "qwen3.8-27b", "max_tokens": n, "temperature": 0, "logprobs": True, "top_logprobs": 5,
            "ignore_eos": True, "messages": [{"role": "user", "content": text(tokens)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    c = r["choices"][0]
    steps = [{"token": t["token"], "lp": t["logprob"], "top": [[x["token"], x["logprob"]] for x in t["top_logprobs"]]}
             for t in c["logprobs"]["content"]]
    return {"prompt_tokens": r["usage"]["prompt_tokens"], "steps": steps}


if sys.argv[1] == "save":
    n = int(sys.argv[3])
    out = {}
    for t in sys.argv[4:]:
        out[t] = ask(int(t), n)
        print(f"{out[t]['prompt_tokens']:7d} tokens: {len(out[t]['steps'])} steps", flush=True)
    json.dump(out, open(sys.argv[2], "w"))
else:
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for k in a:
        if k not in b:
            continue
        sa, sb = a[k]["steps"], b[k]["steps"]
        same = 0
        while same < min(len(sa), len(sb)) and sa[same]["token"] == sb[same]["token"]:
            same += 1
        dl = [abs(sa[i]["lp"] - sb[i]["lp"]) for i in range(same)]
        msg = f"{a[k]['prompt_tokens']:7d} tok: {same}/{len(sa)} greedy tokens identical, mean |dlogprob| {sum(dl) / max(1, len(dl)):.4f}, max {max(dl, default=0):.4f}"
        if same < min(len(sa), len(sb)):
            ga = sa[same]["top"][0][1] - sa[same]["top"][1][1]
            gb = sb[same]["top"][0][1] - sb[same]["top"][1][1]
            msg += (f"; diverge at step {same}: {sa[same]['token']!r} vs {sb[same]['token']!r}, top-2 gap "
                    f"{ga:.3f} / {gb:.3f}")
        print(msg)
