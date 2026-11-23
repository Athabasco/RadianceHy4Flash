#!/usr/bin/env python3
"""probe.py <port> <out.json> <phases,comma>: sampled (T 0.9, seeded) phases for the general-levers note.
phases: c1, c8, think (c1, reasoning high), c4, pf6 (prefill ~6.7K tokens: 22,000 characters of tree source),
pf20 (prefill ~24.9K: 72,000), pf2 (~2.4K: 7,400), stall1 / stall4 (a ~5.7K arrival while 1 / 4 decoders run),
tailN (a prompt of ~N characters), c1s (64 tok), c8s (48 tok), longctx (eight ~46K-token sessions:
a prefill wave of one token each, then the same eight prompts decoded 128 tokens, so all eight are at
long context together: the shape notes/backlog/refs/notes/longctx.md measured as N8 48K, where the
stager overflows its ring -- longctx-pre / longctx-decode in the JSON). The token counts are the ones
the backlog's own runs measured on these phases, so a lane's ttft compares with the numbers in
notes/backlog/."""
import json, sys, threading, time, urllib.request, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import CODE_PROMPTS as P, SRC, long_prompt
PORT, OUT, PH = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
BASE = f"http://127.0.0.1:{PORT}"
def get(p): return urllib.request.urlopen(BASE + p, timeout=10).read().decode()
def find(d, k):
    if isinstance(d, dict):
        if k in d: return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, k)
            if r is not None: return r
    return None
def stats():
    o = json.loads(get("/stats")); e = find(o, "experts") or {}; l = find(o, "link") or {}
    d = {k: v for k, v in e.items() if isinstance(v, (int, float))}
    d.update({"link_" + k: v for k, v in l.items() if isinstance(v, (int, float))}); return d
def metrics():
    m = {}
    for ln in get("/metrics").splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        k, _, v = ln.rpartition(" "); n = k.split("{")[0]
        if n.startswith("radiance:"):
            try: m[n] = m.get(n, 0) + float(v)
            except ValueError: pass
    return m
SEED = [1000]
def temperature():
    # RAD_TEMP, else TEMP when it holds a number: a measuring host often sets TEMP to a directory
    # (it is a system variable, so a value that is not a number is ignored).
    want, v = os.environ.get("RAD_TEMP"), os.environ.get("RAD_TEMP") or os.environ.get("TEMP") or "0.9"
    try: return float(v)
    except ValueError:
        if want: sys.exit(f"probe.py: RAD_TEMP '{want}' is not a number")
        return 0.9
def request(content, n, effort="none"):
    SEED[0] += 1
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": temperature(), "top_p": 1.0, "seed": SEED[0], "reasoning_effort": effort, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); tf = None; gen = 0; pt = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"): continue
            d = ln[5:].strip()
            if d == "[DONE]": break
            ev = json.loads(d)
            if ev.get("usage"): gen = ev["usage"].get("completion_tokens", 0); pt = ev["usage"].get("prompt_tokens", 0)
            for ch in ev.get("choices", []):
                c = (ch.get("delta", {}).get("content") or "") + (ch.get("delta", {}).get("reasoning_content") or "")
                if c and tf is None: tf = time.time()
    t1 = time.time()
    return {"gen": gen, "prompt": pt, "ttft": (tf or t1) - t0, "dec": (gen - 1) / (t1 - tf) if tf and gen > 1 else 0, "t_first": tf or t1, "t1": t1}
def request_gaps(content, n):
    SEED[0] += 1
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0.9, "top_p": 1.0, "seed": SEED[0], "reasoning_effort": "none", "stream": True, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ts = []; t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if ln.startswith("data:") and ln[5:].strip() != "[DONE]": ts.append(time.time() - t0)
    return ts
def stall(name, nbg, chars, ntok=420):
    out = {}; bg = [None] * nbg
    def w(i): bg[i] = request_gaps(P[i], ntok)
    th = [threading.Thread(target=w, args=(i,)) for i in range(nbg)]
    m0 = metrics(); t0 = time.time()
    for t in th: t.start()
    time.sleep(1.5); ta = time.time() - t0
    r = request(long_prompt(chars, 31 + nbg), 8); tb = time.time() - t0
    for t in th: t.join()
    m1 = metrics(); md = {k.split(":")[1]: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
    gaps = []
    for ts in bg:
        g = sorted(((b - a) * 1000, a) for a, b in zip(ts, ts[1:]) if ta <= b <= tb + 0.5)
        gaps.append([round(x[0], 1) for x in g[-8:]])
    typ = sorted((b - a) * 1000 for ts in bg for a, b in zip(ts, ts[1:]) if b < ta)
    o = {"name": name, "bg": nbg, "arrival_prompt": r["prompt"], "arrival_ttft": round(r["ttft"], 3), "arrive_at": round(ta, 2), "first_at": round(ta + r["ttft"], 2),
         "typical_gap_ms": round(typ[len(typ) // 2], 1) if typ else None, "worst_gaps_ms": gaps, "metrics": md, "clients": nbg, "mono_ns": [0, 0]}
    print(json.dumps({k: v for k, v in o.items() if k != "metrics"}), flush=True); time.sleep(2); return o
def phase(name, prompts, n=256, effort="none"):
    res = [None] * len(prompts)
    def w(i): res[i] = request(prompts[i], n, effort)
    s0, m0 = stats(), metrics(); ns0 = time.monotonic_ns(); t0 = time.time()
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0; ns1 = time.monotonic_ns(); s1, m1 = stats(), metrics()
    sd = {k: s1[k] - s0.get(k, 0) for k in s1}; md = {k.split(":")[1]: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
    lo = max(r["t_first"] for r in res); hi = min(r["t1"] for r in res)
    steps = md.get("engine_steps_decode_total", 0)
    o = {"name": name, "clients": len(prompts), "wall": wall, "gen": sum(r["gen"] for r in res), "prompt": [r["prompt"] for r in res],
         "ttft": [round(r["ttft"], 3) for r in res], "agg_tps": sum(r["gen"] for r in res) / wall,
         "sum_req_dec_tps": sum(r["dec"] for r in res), "decode_steps": steps,
         "ms_step": 1000 * (hi - lo) / max(1, steps) if len(prompts) == 1 else None,
         "tok_per_step": sum(r["gen"] for r in res) / max(1, steps),
         "mono_ns": [ns0, ns1], "all_decoding_s": [lo - t0, hi - t0], "stats": sd, "metrics": md,
         "hit": sd.get("routed_res", 0) / max(1, sd.get("routed", 1)),
         "routed_per_step": sd.get("routed", 0) / max(1, steps),
         "miss_units_per_step": (sd.get("routed", 0) - sd.get("routed_res", 0)) / max(1, steps)}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k not in ("stats", "metrics")}), flush=True)
    time.sleep(2)
    return o
def longctx(chars=140000, n=8, ntok=128):
    # Eight sessions arrive and fill. One prompt each, built once: the second wave must carry the
    # same strings or it misses the prefix cache and prefills again instead of decoding.
    ps = [long_prompt(chars, 40 + i, stamp=True) for i in range(n)]
    pre = phase("longctx-pre", ps, 1)
    dec = phase("longctx-decode", ps, ntok)
    return {"prefill": pre, "decode": dec}
R = {"phases": []}
if "nowarm" not in PH: request(P[11], 48); request(P[10], 48)
for ph in PH:
    if ph == "c1": R["phases"].append(phase("c1", [P[0]]))
    elif ph == "c1b": R["phases"].append(phase("c1b", [P[5]]))
    elif ph == "c8": R["phases"].append(phase("c8", [P[i] for i in range(8)]))
    elif ph == "c4": R["phases"].append(phase("c4", [P[i] for i in (1, 3, 5, 7)]))
    elif ph == "c2": R["phases"].append(phase("c2", [P[2], P[6]]))
    elif ph == "think": R["phases"].append(phase("think", [P[4]], 384, "high"))
    elif ph == "c1s": R["phases"].append(phase("c1s", [P[0]], 64))
    elif ph == "c8s": R["phases"].append(phase("c8s", [P[i] for i in range(8)], 48))
    elif ph == "pf6": R["phases"].append(phase("pf6", [long_prompt(22000, 7, stamp=True)], 2))
    elif ph == "pf20": R["phases"].append(phase("pf20", [long_prompt(72000, 9, stamp=True)], 2))
    elif ph == "pf2": R["phases"].append(phase("pf2", [long_prompt(7400, 5)], 2))
    elif ph == "stall1": R["phases"].append(stall("stall1", 1, 22000))
    elif ph == "stall4": R["phases"].append(stall("stall4", 4, 22000))
    elif ph.startswith("tail"):  # tailN: prompt of ~N chars lone, to size small prefill steps
        R["phases"].append(phase(ph, [long_prompt(int(ph[4:]), 3)], 2))
    elif ph == "longctx": R["phases"].append(longctx())
    elif ph == "pf6c": R["phases"].append(phase("pf6c", [long_prompt(22000, 7, stamp=True)], 2))
    else: sys.exit(f"probe.py: no phase '{ph}'; phases: c1 c1b c1s c2 c4 c8 c8s think pf2 pf6 pf6c pf20 "
                  "longctx stall1 stall4 tailN nowarm")
json.dump(R, open(OUT, "w"), indent=1)
