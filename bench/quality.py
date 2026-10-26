#!/usr/bin/env python3
"""Quality probe for prefill-attention changes: the next-token distribution after long prompts.

For each length, a fixed text (vLLM source, identical across runs) plus a question; one
greedy request with max_tokens=8 and the top-20 logprobs of every generated token. The first
token's distribution depends on the whole prompt's KV cache, so a prefill-attention error
shows up there. (prompt_logprobs would be stricter, but gathering 2048 x 248k logits per
chunk needs ~750 MB that the 200k profile does not have.)

Usage: quality.py save <out.json> <tokens> [<tokens> ...]
       quality.py compare <a.json> <b.json>
"""
import glob
import json
import os
import math
import sys
import urllib.request

URL = "http://localhost:8000"


def text(tokens):
    parts, size = [], 0
    for f in sorted(glob.glob(f"{os.environ.get('VLLM_SRC', 'vllm-src')}/vllm/**/*.py", recursive=True)):
        src = open(f).read()
        parts.append(f"\n===== {f} =====\n{src}")
        size += len(src)
        if size > tokens * 3.8:
            break
    return "".join(parts)[: int(tokens * 3.8)]


def ask(tokens):
    body = {"model": "qwen3.8-27b", "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "user", "content": text(tokens) + "\n\nWhich file above is the longest? Answer with its path."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    c = r["choices"][0]
    steps = [{"token": t["token"], "top": {x["token"]: x["logprob"] for x in t["top_logprobs"]}}
             for t in c["logprobs"]["content"]]
    return {"prompt_tokens": r["usage"]["prompt_tokens"], "text": c["message"]["content"], "steps": steps}


if sys.argv[1] == "save":
    out = {}
    for n in sys.argv[3:]:
        out[n] = ask(int(n))
        print(f"{out[n]['prompt_tokens']:7d} tokens -> {out[n]['text']!r}", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
else:
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    for n in a:
        if n not in b:
            continue
        sa, sb = a[n]["steps"], b[n]["steps"]
        same = 0
        while same < min(len(sa), len(sb)) and sa[same]["token"] == sb[same]["token"]:
            same += 1
        # first-token distribution: KL(a||b) over the shared top-20 tokens (renormalised)
        ta, tb = sa[0]["top"], sb[0]["top"]
        common = [t for t in ta if t in tb]
        za = sum(math.exp(ta[t]) for t in common); zb = sum(math.exp(tb[t]) for t in common)
        kl = sum(math.exp(ta[t]) / za * (ta[t] - math.log(za) - tb[t] + math.log(zb)) for t in common)
        top = max(ta, key=ta.get)
        print(f"{a[n]['prompt_tokens']:7d} tok: first token {top!r} p={math.exp(ta[top]):.4f} vs "
              f"p={math.exp(tb.get(top, -99)):.4f}, KL {kl:.2e}, {same}/{len(sa)} greedy tokens identical; "
              f"{a[n]['text']!r} | {b[n]['text']!r}")
