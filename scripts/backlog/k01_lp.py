#!/usr/bin/env python3
"""Greedy logprobs probe for K-01's RADIANCE_HY4_FUSE_DUAL=0/1 comparison.

Same prompts as probe.py's `greedy` phase, at temperature 0, asking for the chosen token's logprob
and the four alternatives. Two arms agree exactly if every logprob is equal bit for bit -- the dual
launch is byte-identical to the two launches it replaces, so the pass is.
"""
import json
import os
import sys
import urllib.request

PORT = int(sys.argv[1])
OUT = sys.argv[2]
sys.path.insert(0, os.environ["PROBE_DIR"])
from prompts import CODE_PROMPTS as P, GREEDY  # noqa: E402

BASE = f"http://127.0.0.1:{PORT}"
prompts = list(GREEDY[:2]) + [P[7]]

out = []
for i, content in enumerate(prompts):
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": 32, "temperature": 0.0, "top_p": 1.0, "reasoning_effort": "none",
            "logprobs": True, "top_logprobs": 4, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        ev = json.load(r)
    ch = ev["choices"][0]
    lp = ch.get("logprobs") or {}
    rows = []
    for tok, top in zip(lp.get("content") or [], lp.get("top_logprobs") or []):
        rows.append({"token": tok["token"], "logprob": tok["logprob"],
                     "top": [[t["token"], t["logprob"]] for t in (top or [])]})
    out.append({"i": i, "content": ch["message"].get("content"), "rows": rows})
    print(f"prompt {i}: {len(rows)} logprob rows", flush=True)

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT)
