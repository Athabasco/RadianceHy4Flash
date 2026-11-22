#!/usr/bin/env python3
"""Throughput bench against the radiance HTTP API (stdlib only).

bench.py <port> <tag> <outdir> [--reps N] [--cold]
Phases: (cold: one c1 request + one 5.7K prefill, reported separately), warm-up, then per rep:
c1 (3 requests), c4 (4 clients x 2), c8 (8 clients x 2), prefill ~5.7K (x2) and ~18K (x2);
then greedy texts of 3 prompts (no ignore_eos) for the DFly2 on/off comparison.
Every decode request: 256 generated tokens (ignore_eos), temperature 0, reasoning none, streamed.
Decode tok/s for N clients = sum of generated tokens / wall time of the phase.  /metrics is scraped
around each phase (decode tokens, decode steps, draft tokens/accepted)."""
import json, os, random, sys, threading, time, urllib.request

PORT, TAG, OUT = int(sys.argv[1]), sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "res")
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
COLD = "--cold" in sys.argv
# WP13.8: --temp T samples every decode request at temperature T (seed per request), so a coupled
# drafter's law is used; 0 (the default) is the greedy bench of work/perf.
TEMP = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 0.0
_SEED = [0]
BASE = f"http://127.0.0.1:{PORT}"
SRC = "/root/work/claude-scratch/hy4flash/src"

CODE_PROMPTS = [
    "Write a complete Python module implementing an LRU cache class with get/put, TTL expiry per entry, thread safety via a lock, type hints and docstrings, followed by a pytest test suite covering eviction order, expiry and concurrent access.",
    "Implement in C++17 a lock-free single-producer single-consumer ring buffer template with power-of-two capacity, std::atomic indices with acquire/release ordering, try_push/try_pop, and a small benchmark main(). Explain each memory ordering choice in comments.",
    "Write a Rust command-line tool that recursively scans a directory, groups files by SHA-256 hash to find duplicates, prints the groups sorted by wasted bytes, and supports --min-size and --json flags. Use clap and walkdir; include error handling and unit tests.",
    "Write a TypeScript Express REST API for a todo list with routes for create, list (with pagination and filtering by status), update and delete, input validation with zod, an in-memory repository behind an interface, and Jest tests using supertest.",
    "Here is a slow Python function:\n\n```python\ndef pairs(nums, target):\n    out = []\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] + nums[j] == target and (nums[j], nums[i]) not in out:\n                out.append((nums[i], nums[j]))\n    return out\n```\n\nRewrite it to run in O(n), keep the exact output semantics, explain the differences, and add property-based tests with hypothesis comparing it to the original.",
    "Write a Go HTTP reverse proxy with round-robin load balancing over a configurable list of backends, active health checks every 5 seconds, request timeouts, graceful shutdown on SIGTERM and Prometheus metrics for request counts and latencies. Include the full main.go and a README section on usage.",
    "Implement Dijkstra's algorithm and A* search in Python over a weighted grid read from a text file (digits are cell costs, # is a wall), with a CLI that prints the path overlaid on the grid, plus unit tests comparing both algorithms on random grids.",
    "Write a bash script that backs up a PostgreSQL database with pg_dump, compresses it with zstd, uploads it to an S3 bucket with the aws cli, keeps the last 14 daily and 8 weekly backups, logs to syslog and exits non-zero with a clear message on any failure. Then explain every section.",
    "Write a C function library for a growable string builder (create, append, appendf with printf-style formatting, insert, free) with careful overflow checks, then a test program using assert that exercises edge cases such as empty strings and very large appends.",
    "Write a SQL schema for a library system (books, authors, members, loans, reservations) with constraints and indexes, then ten example queries in PostgreSQL covering joins, window functions, CTEs and an upsert, and explain each query's plan considerations.",
    "Write a Python asyncio web crawler that respects robots.txt, limits concurrency per host with semaphores, deduplicates URLs, follows links up to a configurable depth, stores pages in SQLite and exposes a progress report. Include type hints and tests with aiohttp test utilities.",
    "Implement a simple tokenizer, recursive-descent parser and tree-walking interpreter in Python for arithmetic expressions with variables, assignment, if/else and while loops, plus a REPL and tests for operator precedence and error messages.",
]
GREEDY = [CODE_PROMPTS[0], CODE_PROMPTS[4],
          "Fix the bug in this C function and show the corrected code:\n\nint sum(int *a, int n) {\n  int s;\n  for (int i = 0; i <= n; i++) s += a[i];\n  return s;\n}"]


def code_corpus():
    files = []
    for d in ("core", "arch/hyv4flash", "libavx", "libr4d", "core/server", "core/sched"):
        p = os.path.join(SRC, d)
        for f in sorted(os.listdir(p)):
            if f.endswith((".cpp", ".h", ".hip")):
                files.append(os.path.join(p, f))
    return files


CORPUS = code_corpus()


def long_prompt(chars, seed):
    rnd = random.Random(seed)
    fs = CORPUS[:]
    rnd.shuffle(fs)
    parts, n = [], 0
    for f in fs:
        t = open(f, errors="replace").read()
        t = t[: max(0, chars - n)]
        parts.append(f"// ===== {os.path.relpath(f, SRC)} =====\n{t}")
        n += len(t)
        if n >= chars:
            break
    return (f"[review {seed}] Review the following C++ code for bugs, then summarise the design.\n\n"
            + "\n".join(parts))


def metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    m = {}
    for ln in txt.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        k, _, v = ln.rpartition(" ")
        name = k.split("{")[0]
        if name in ("radiance:decode_tokens_total", "radiance:engine_steps_decode_total",
                    "radiance:engine_steps_total", "radiance:draft_tokens_total",
                    "radiance:draft_accepted_total", "radiance:prefill_tokens_total",
                    "radiance:engine_steps_mixed_total"):
            try:
                m[name] = m.get(name, 0.0) + float(v)
            except ValueError:
                pass
    return m


def stats():
    try:
        o = json.loads(urllib.request.urlopen(BASE + "/stats", timeout=10).read().decode())
    except Exception as e:
        return {"err": str(e)}
    def find(d, k):
        if isinstance(d, dict):
            if k in d:
                return d[k]
            for v in d.values():
                r = find(v, k)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for v in d:
                r = find(v, k)
                if r is not None:
                    return r
        return None
    e = find(o, "experts") or {}
    l = find(o, "link") or {}
    out = {k: e.get(k) for k in ("routed", "routed_res", "promotions", "demotions", "dispatches")}
    out.update({"link_" + k: v for k, v in l.items() if isinstance(v, (int, float))})
    out["tiers"] = e.get("tiers")
    return out


def sdelta(a, b):
    d = {}
    for k, v in b.items():
        if isinstance(v, (int, float)) and isinstance(a.get(k), (int, float)):
            d[k] = v - a[k]
    if d.get("routed"):
        d["resident_hit"] = d.get("routed_res", 0) / d["routed"]
    d["tiers_after"] = b.get("tiers")
    return d


def mdelta(a, b):
    d = {k.split(":")[1]: b.get(k, 0) - a.get(k, 0) for k in b}
    st = d.get("engine_steps_decode_total", 0)
    if st:
        d["decode_tok_per_step"] = d.get("decode_tokens_total", 0) / st
    if d.get("draft_tokens_total"):
        d["accept_ratio"] = d["draft_accepted_total"] / d["draft_tokens_total"]
    return d


def request(content, max_tokens, ignore_eos=True, keep_text=False, temp=None):
    temp = TEMP if temp is None else temp
    _SEED[0] += 1
    body = {"model": "hy4-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": temp, "top_p": 1.0, "reasoning_effort": "none",
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
    if temp > 0:
        body["seed"] = 7919 * _SEED[0]
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    tfirst = None
    usage, text = {}, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:"):
                continue
            d = ln[5:].strip()
            if d == "[DONE]":
                break
            ev = json.loads(d)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if tfirst is None:
                        tfirst = time.time()
                    if keep_text:
                        text.append(delta.get("content") or "")
    t1 = time.time()
    gen = usage.get("completion_tokens", 0)
    res = {"t0": t0, "t1": t1, "ttft": (tfirst or t1) - t0, "e2e": t1 - t0, "gen": gen,
           "prompt": usage.get("prompt_tokens", 0)}
    if tfirst and gen > 1:
        res["decode_tps"] = (gen - 1) / (t1 - tfirst)
    if keep_text:
        res["text"] = "".join(text)
    return res


def decode_phase(nclients, per_client, seed, ident=False):
    rnd = random.Random(seed)
    results, lock = [], threading.Lock()

    def worker(ci):
        for k in range(per_client):
            p = CODE_PROMPTS[(ci + k * nclients + seed) % len(CODE_PROMPTS)]
            r = request(CODE_PROMPTS[0] if ident else f"(task {seed}-{ci}-{k}) {p}", 256)
            with lock:
                results.append(r)

    m0 = metrics()
    s0 = stats()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(nclients)]
    t0 = time.time()
    for t in th:
        t.start()
        time.sleep(0.02)
    for t in th:
        t.join()
    wall = time.time() - t0
    md = mdelta(m0, metrics())
    sd = sdelta(s0, stats())
    gen = sum(r["gen"] for r in results)
    out = {"clients": nclients, "requests": len(results), "gen": gen, "wall": wall,
           "agg_tps": gen / wall, "mean_req_decode_tps": sum(r.get("decode_tps", 0) for r in results) / len(results),
           "mean_ttft": sum(r["ttft"] for r in results) / len(results), "metrics": md, "stats": sd, "reqs": results}
    print(f"[{TAG}] c{nclients}: {gen} tok in {wall:.1f}s -> {out['agg_tps']:.1f} tok/s aggregate, "
          f"per-request decode {out['mean_req_decode_tps']:.1f}, ttft {out['mean_ttft']:.2f}s, "
          f"tok/step {md.get('decode_tok_per_step', 0):.2f}, accept {md.get('accept_ratio', 0):.3f}, expert VRAM hit {sd.get('resident_hit', -1):.3f}", flush=True)
    return out


def prefill_phase(chars, seed):
    m0 = metrics()
    r = request(long_prompt(chars, seed), 1)
    r["prefill_tps"] = r["prompt"] / r["ttft"]
    r["metrics"] = mdelta(m0, metrics())
    print(f"[{TAG}] prefill {r['prompt']} tok: ttft {r['ttft']:.2f}s -> {r['prefill_tps']:.0f} tok/s", flush=True)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"tag": TAG, "start": time.time()}
    # chars per token, calibrated on the first long prompt
    if COLD:
        res["cold_c1"] = decode_phase(1, 1, 900)
        res["cold_prefill"] = prefill_phase(19000, 901)
    cpt = 3.3
    r = prefill_phase(19000, 902)        # warm-up + calibration
    cpt = 19000 / max(1, r["prompt"]) * 1.0
    res["chars_per_token"] = cpt
    res["warmup"] = decode_phase(4, 1, 903)
    res["reps"] = []
    for rep in range(REPS):
        s = 1000 * (rep + 1)
        rr = {"c1": decode_phase(1, 3, s + 1), "c4": decode_phase(4, 2, s + 2), "c8": decode_phase(8, 2, s + 3), "c8id": decode_phase(8, 1, s + 4, True),
              "p5k": [prefill_phase(int(5700 * cpt), s + 10 + i) for i in range(2)],
              "p18k": [prefill_phase(int(18000 * cpt), s + 20 + i) for i in range(2)]}
        res["reps"].append(rr)
        json.dump(res, open(os.path.join(OUT, f"{TAG}.json"), "w"), indent=1)
    res["end"] = time.time()
    json.dump(res, open(os.path.join(OUT, f"{TAG}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
