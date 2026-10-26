#!/usr/bin/env python3
"""Decode speed vs context length on the running server (MTP acceptance included).

Usage: decbench.py <csv_out> <tag> <tokens> [<tokens> ...]   e.g. decbench.py dec.csv base 1000 16000 32000 64000
Env: REPS (3), MAX_TOKENS (256), TEMP (0 = greedy; >0 uses a fixed seed per rep), COOL_C (55),
ABORT_C (80), VLLM_SRC (text corpus).
Per length: one prefill request (max_tokens=1) puts the prompt in the prefix cache, then
REPS streamed requests on the same prompt. decode t/s = (completion_tokens - 1) /
(t_last_chunk - t_first_chunk); acceptance and steps come from /metrics deltas, so
ms/step = decode time / drafts is comparable even when acceptance differs.
Logs power/clocks/temps every second to the CSV; aborts at ABORT_C on any sensor.
"""
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000"
ABORT_C = float(os.environ.get("ABORT_C", "80"))  # any sensor; run under bench/safe-run.sh as well
COOL_C = float(os.environ.get("COOL_C", "55"))
REPS = int(os.environ.get("REPS", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMP = float(os.environ.get("TEMP", "0"))
out = open(sys.argv[1], "a", buffering=1)
TAG = sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3:]]
CARDS = sorted(glob.glob("/sys/class/drm/card[01]/device/hwmon/hwmon*"))
SRC = os.environ.get("VLLM_SRC", "vllm-src")  # text corpus: a vLLM source checkout
FILES = sorted(glob.glob(f"{SRC}/vllm/**/*.py", recursive=True))
step = {"name": "idle"}
busy = []


def rd(p, div):
    return int(open(p).read()) / div


def logger():
    if out.tell() == 0:
        out.write("t,tag,step,card,power_w,cap_w,sclk_mhz,edge_c,junction_c,mem_c\n")
    while True:
        now = time.time()
        for h in CARDS:
            v = [rd(f"{h}/power1_input", 1e6), rd(f"{h}/power1_cap", 1e6), rd(f"{h}/freq1_input", 1e6),
                 rd(f"{h}/temp1_input", 1e3), rd(f"{h}/temp2_input", 1e3), rd(f"{h}/temp3_input", 1e3)]
            out.write(f"{now:.1f},{TAG},{step['name']},{h.split('/')[4]}," + ",".join(f"{x:.0f}" for x in v) + "\n")
            if step["name"].startswith("dec"):
                busy.append((step["name"], v))
            if max(v[3:]) >= ABORT_C:
                print(f"ABORT: a sensor reached {max(v[3:]):.0f} C", flush=True)
                os._exit(3)
        time.sleep(max(0.0, 1 - (time.time() - now)))


def prompt(tokens):
    # deterministic text (no nonce) so the measured requests hit the prefix cache
    parts, size = [f"Context benchmark {tokens}: repository files\n"], 0
    for f in FILES:
        src = open(f).read()
        parts.append(f"\n===== {f} =====\n{src}")
        size += len(src)
        if size > tokens * 3.4:
            break
    return "".join(parts)[: int(tokens * 3.4)] + "\n\nExplain in detail what the code above does, file by file."


def metrics():
    t = urllib.request.urlopen(f"{URL}/metrics").read().decode()
    g = lambda n: float(re.search(rf"^vllm:spec_decode_{n}_total\{{[^}}]*\}} (\S+)", t, re.M).group(1))
    return g("num_drafts"), g("num_draft_tokens"), g("num_accepted_tokens")


def request(text, max_tokens, seed=None):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": TEMP, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None and TEMP > 0:
        body["seed"] = seed
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tf = tl = None
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            if c.get("choices"):
                now = time.perf_counter()
                tf = tf or now
                tl = now
    return usage, tf - t0, tl - tf


def cool():
    while max(rd(f"{h}/temp1_input", 1e3) for h in CARDS) > COOL_C:
        time.sleep(2)


threading.Thread(target=logger, daemon=True).start()
print(f"tag={TAG} reps={REPS} max_tokens={MAX_TOKENS} temp={TEMP}", flush=True)
for n in LENGTHS:
    text = prompt(n)
    step["name"] = f"prefill{n}"
    t0 = time.time()
    u, _, _ = request(text, 1)
    print(f"[{TAG}] ctx {u['prompt_tokens']:6d}: prefill/cache {time.time() - t0:6.1f} s", flush=True)
    res = []
    for r in range(REPS):
        cool()
        step["name"] = f"dec{n}_{r}"
        busy.clear()
        m0 = metrics()
        u, ttft, dt = request(text, MAX_TOKENS, seed=1234 + r)
        m1 = metrics()
        step["name"] = "idle"
        d = [b - a for a, b in zip(m0, m1)]
        ntok = u["completion_tokens"]
        tps = (ntok - 1) / dt
        al = 1 + d[2] / d[0] if d[0] else float("nan")
        ms_step = dt * 1e3 / d[0] if d[0] else float("nan")
        pw = [b[1][0] for b in busy]
        pmean = sum(pw) / len(pw) if pw else float("nan")
        cap = sum(b[1][1] for b in busy) / len(busy) if busy else float("nan")
        jmax = max((b[1][4] for b in busy), default=float("nan"))
        res.append((tps, al, ms_step))
        print(f"[{TAG}] ctx {u['prompt_tokens']:6d} rep{r}: TTFT {ttft:5.2f}s decode {tps:6.2f} t/s  accept len {al:4.2f}"
              f"  {ms_step:6.1f} ms/step  power {pmean:5.1f} W (cap {cap:5.1f})  junction max {jmax:.0f} C", flush=True)
    k = len(res)
    print(f"[{TAG}] ctx {u['prompt_tokens']:6d} MEAN: {sum(r[0] for r in res) / k:6.2f} t/s  accept len "
          f"{sum(r[1] for r in res) / k:4.2f}  {sum(r[2] for r in res) / k:6.1f} ms/step", flush=True)
