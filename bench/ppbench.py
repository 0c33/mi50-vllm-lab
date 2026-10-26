#!/usr/bin/env python3
"""Prompt-processing (prefill) speed of the running server at several prompt lengths.

Usage: ppbench.py <csv_out> <tag> <tokens> [<tokens> ...]     e.g. ppbench.py pp.csv old 2048 8192 32768
Per length: wait until both edges are <= COOL_C, send one prompt that starts with a nonce
(so the prefix cache never hits) with max_tokens=1, and time it. Prints prompt tokens,
time and tokens/s; logs power/clocks/temps every second to the CSV. Aborts at ABORT_C
(default 80 C on any sensor). Env: VLLM_SRC (text corpus), COOL_C, ABORT_C.
"""
import glob
import json
import os
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000"
COOL_C = float(os.environ.get("COOL_C", "50"))
ABORT_C = float(os.environ.get("ABORT_C", "80"))  # any sensor; run under bench/safe-run.sh as well
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
            if step["name"] != "idle":
                busy.append((step["name"], v))
            if max(v[3:]) >= ABORT_C:
                print(f"ABORT: a sensor reached {max(v[3:]):.0f} C", flush=True)
                os._exit(3)
        time.sleep(max(0.0, 1 - (time.time() - now)))


def prompt(tokens):
    parts, size = [f"Benchmark {time.time_ns()}: repository files\n"], 0
    for f in FILES:  # ~3.4 characters per token for this source
        src = open(f).read()
        parts.append(f"\n===== {f} =====\n{src}")
        size += len(src)
        if size > tokens * 3.4:
            break
    return "".join(parts)[: int(tokens * 3.4)] + "\n\nSummarize."


def run(tokens):
    body = {"model": "qwen3.8-27b", "max_tokens": 1, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt(tokens)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("choices") and first is None:
                first = time.time()
            if c.get("usage"):
                usage = c["usage"]
    return usage["prompt_tokens"], (first or time.time()) - t0


threading.Thread(target=logger, daemon=True).start()
for n in LENGTHS:
    step["name"] = f"cool{n}"
    t0 = time.time()
    while max(rd(f"{h}/temp1_input", 1e3) for h in CARDS) > COOL_C:
        if time.time() - t0 > 900:
            sys.exit(f"not cool after 15 min before {n}")
        time.sleep(2)
    step["name"] = f"pp{n}"
    toks, secs = run(n)
    s = [v for name, v in busy if name == f"pp{n}"]
    pw = sum(v[0] for v in s) / max(len(s), 1); clk = sum(v[2] for v in s) / max(len(s), 1)
    pk = max((max(v[3:]) for v in s), default=0)
    print(f"{TAG:>6} {toks:7d} tokens  {secs:8.1f} s  {toks / secs:7.1f} tok/s   mean {pw:5.1f} W/card"
          f"  sclk {clk:5.0f} MHz  peak {pk:.0f} C", flush=True)
    step["name"] = "idle"
