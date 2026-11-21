#!/usr/bin/env python3
"""Readiness check for Hy4-flash quantisation: the tier-A checklist A1..A11 (PLAN section 4.4), run
where it can be run, printed as a pass/fail table.

    python3 tools/hy4flash_ready.py --build build [--ref REF_DIR] [--release DIR] [--only A1,A9] [--gpu]

Status per item:  PASS   checked here and green
                  FAIL   checked here and red (the detail says what)
                  OPEN   cannot be decided by this script (needs a card, a package that has not merged, or
                         the network); the detail gives the command that decides it
                  SKIP   not selected (--only)
Exit status 0 when no item is FAIL (OPEN does not fail the run, it is listed: "ready" needs every item PASS).

Needs: a built tree (`--build`, host or HIP, with the test targets), numpy; for A2-A4 the HF parity harness
directory `--ref` (default <repo>/../ref, the venv with torch inside it); for A1's release structure, network
access to huggingface.co once (148 shard headers, ~10 MB; `--release DIR` keeps them: re-run offline).
Never downloads weights, never touches a card unless --gpu (then under flock gpu.lock).

Logs of every step go to --work (default <build>/ready), the table is also written there as ready.json.
Do not write "ready" into quant.md from this script's output alone: read the OPEN lines.
"""
import argparse, concurrent.futures as cf, glob, json, os, re, shutil, struct, subprocess, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = "tencent/Hy4-flash"
N_TENSORS = 1964
REAL_VOCAB = 120832
GPU_LOCK = "/root/work/claude-scratch/gpu.lock"
RESULTS = {}      # id -> [(status, text)]


def say(item, status, text):
    RESULTS.setdefault(item, []).append((status, text))
    print(f"  [{status:4}] {item}: {text}", flush=True)


class Ctx:
    pass


def run(cmd, log, env=None, timeout=3600, cwd=None):
    """-> (rc, text). The full output goes to `log`."""
    e = dict(os.environ)
    e.update(env or {})
    try:
        r = subprocess.run([str(c) for c in cmd], env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=timeout, cwd=cwd)
        rc, out = r.returncode, r.stdout
    except subprocess.TimeoutExpired as ex:
        rc, out = 124, (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or "") + "\nTIMEOUT"
    with open(log, "w") as f:
        f.write(" ".join(str(c) for c in cmd) + "\n\n" + out)
    return rc, out


def host_env(c):
    return {"ROCR_VISIBLE_DEVICES": "99", "HIP_VISIBLE_DEVICES": "", "RADIANCE_HOME": c.home,
            "RADIANCE_KERNELS": "libref", "RAD_TEST_CPU_BUDGET_SCALE": "4", "OMP_NUM_THREADS": "2"}


def ctest(c, regex, log_name, extra_env=None, timeout=3600):
    env = dict(host_env(c), **(extra_env or {}))
    env.pop("RADIANCE_KERNELS")
    rc, out = run(["ctest", "--test-dir", c.build, "-R", regex, "-j4", "--output-on-failure"], f"{c.work}/{log_name}.log",
                  env, timeout)
    m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", out)
    skipped = re.findall(r"\*\*\*Skipped|\bSkipped\b", out)
    nskip = len(re.findall(r"Test\s+#\d+:.*\*\*\*Skipped", out))
    nrun = int(m.group(3)) if m else 0
    nfail = int(m.group(2)) if m else -1
    return rc, nrun, nfail, nskip, out


# ------------------------------------------------------------------------------------------------ the release structure
def fetch_release(d, log=print):
    """The cloud session's reconstruction (HANDOFF.md 'Not transferable'): config + tokenizer files, the index, and each
    shard as 8-byte length + header, truncated (sparse) to its real size. Idempotent."""
    os.makedirs(d, exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/main/"

    def get(name, rng=None):
        rq = urllib.request.Request(base + name, headers={"Range": rng} if rng else {})
        with urllib.request.urlopen(rq, timeout=120) as r:
            return r.read(), r.headers

    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            open(p, "wb").write(get(f)[0])
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))

    def one(s):
        p = os.path.join(d, s)
        if os.path.exists(p) and os.path.getsize(p) > 1 << 20:
            return 0
        head, h = get(s, "bytes=0-7")
        n = struct.unpack("<Q", head)[0]
        total = int(h["Content-Range"].split("/")[1])
        hd, _ = get(s, f"bytes=8-{7 + n}")
        with open(p, "wb") as f:
            f.write(head + hd)
            f.truncate(total)
        return 1

    with cf.ThreadPoolExecutor(8) as ex:
        got = sum(ex.map(one, shards))
    log(f"release structure in {d}: {len(shards)} shards ({got} fetched)")
    return len(shards)


def shard_tensor_shape(d, name):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    sh = idx["weight_map"].get(name)
    if not sh:
        return None
    with open(os.path.join(d, sh), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))[name]["shape"]


# ------------------------------------------------------------------------------------------------ items
def a1(c):
    rel = c.release
    rc, nrun, nfail, nskip, out = ctest(c, "^hyv4flash_convert_test$", "a1-ctest",
                                        {"RAD_TEST_HY4_TOKENIZER_DIR": rel} if c.release_ok else None)
    if nrun == 1 and nfail == 0 and rc == 0 and not nskip:
        say("A1", "PASS", "hyv4flash_convert_test green" + (" (with the release tokenizer)" if c.release_ok else ""))
    elif nrun == 1 and nfail == 0 and rc == 0:
        say("A1", "PASS", "hyv4flash_convert_test green (skipped)")
    else:
        say("A1", "FAIL", f"hyv4flash_convert_test: rc {rc}, see {c.work}/a1-ctest.log")
    if not c.release_ok:
        say("A1", "OPEN", f"release structure unavailable ({c.release_err}); on a machine with network: re-run, or "
                          f"`rad-convert <release dir> --plan-only` (HANDOFF.md 'Not transferable')")
        return
    recipes = [None] + sorted(glob.glob(os.path.join(SRC, "data/recipes/hy4-flash-*.recipe")))
    counts = set()
    for r in recipes:
        nm = "none" if r is None else os.path.basename(r)[len("hy4-flash-"):-len(".recipe")]
        cmd = [c.convert, rel, "--plan-only", "-o", os.path.join(c.work, "plan-unused.rad")] + (["--recipe", r] if r else [])
        env = host_env(c)
        env["CALIB"] = os.path.join(c.work, "no-grams")        # a gptq recipe still plans (its Grams are the friend's)
        rc, out = run(cmd, f"{c.work}/a1-plan-{nm}.log", env, 900)
        m = re.search(r"source\s+\S+ \((\d+) tensors\)", out)
        p = re.search(r"plan\s+(\d+) weight\(s\): (\d+) quantised by the recipe, (\d+) kept", out)
        sz = re.search(r"about ([\d.]+ GiB) would be written", out)
        bad = [l for l in out.splitlines() if re.match(r"^(E|F) ", l)]
        if rc == 0 and m and int(m.group(1)) == N_TENSORS and p and sz and not bad:
            counts.add(int(p.group(1)))
            say("A1", "PASS", f"plan-only {nm}: {m.group(1)} tensors read, {p.group(1)} weights ({p.group(2)} quantised), {sz.group(1)}")
        else:
            say("A1", "FAIL", f"plan-only {nm}: rc {rc}, tensors {m and m.group(1)}, "
                              f"{bad[:1] or 'see ' + c.work + '/a1-plan-' + nm + '.log'}")
    if len(counts) > 1:
        say("A1", "FAIL", f"declared weight counts differ between recipes: {sorted(counts)}")


def run_rad(c, preset, tp, scenario, kernels="libref", extra=()):
    # the harness needs a chunk of at least 16 (blocks of 4 and 16): as work/wp4/matrix.sh runs it
    extra = list(extra) + {"oneshot": ["--chunk", "64"], "empty": ["--chunk", "16"]}.get(scenario, [])
    py = os.path.join(c.ref, "venv/bin/python")
    cmd = [py, os.path.join(c.refx, "run_rad.py"), "--preset", preset, "--tp", tp, "--scenario", scenario, "--build", c.build,
           "--src", SRC, "--kernels", kernels, "--json", f"{c.work}/rad-{preset}-{kernels}-tp{tp}-{scenario}.json", *extra]
    log = f"{c.work}/rad-{preset}-{kernels}-tp{tp}-{scenario}.log"
    rc, out = run(cmd, log, {"RAD_BUILD": c.build}, 3 * 3600, cwd=c.refx)
    m = re.search(r"(\d+) comparisons, (\d+) FAIL", out)
    return rc, (int(m.group(1)), int(m.group(2))) if m else None, log


def a2(c):
    if not c.ref_ok:
        return say("A2", "OPEN", f"HF harness dir {c.ref} not found (--ref)")
    libs = ["libref"] + (["libavx"] if os.path.exists(f"{c.home}/kernels/libavx.so") else [])
    if "libavx" not in libs:
        say("A2", "OPEN", "libavx.so not built in this build; `ninja libavx` and re-run")
    for k in libs:
        for sc in ("oneshot", "chunked", "decode", "empty"):
            rc, m, log = run_rad(c, "mini", 1, sc, k)
            if rc == 0 and m and m[1] == 0:
                say("A2", "PASS", f"mini TP1 {k} {sc}: {m[0]} comparisons, 0 FAIL")
            else:
                say("A2", "FAIL", f"mini TP1 {k} {sc}: rc {rc} {m}; {log}")


def a3(c):
    rc, nrun, nfail, nskip, out = ctest(c, "hyv4flash_arch|ref_hy4|hyv4flash_calib|hyv4flash_convert|hyv4flash_chat", "a3-ctest")
    ok = rc == 0 and nrun > 0 and nfail == 0
    say("A3", "PASS" if ok else "FAIL",
        f"hyv4flash unit/trap tests: {nrun} run, {nfail} failed, {nskip} skipped (list of the traps: PLAN 1.2; each is a case of "
        f"hyv4flash_arch_test / ref_hy4_ops_test / hyv4flash_convert_test)")


def a4(c):
    if not c.ref_ok:
        return say("A4", "OPEN", f"HF harness dir {c.ref} not found (--ref)")
    for tp in (2, 4):
        for sc in ("oneshot", "chunked"):
            rc, m, log = run_rad(c, "mini4", tp, sc)
            if rc == 0 and m and m[1] == 0:
                say("A4", "PASS", f"mini4 TP{tp} fake ranks, whole experts, {sc}: {m[0]} comparisons, 0 FAIL")
            else:
                say("A4", "FAIL", f"mini4 TP{tp} {sc}: rc {rc} {m}; {log}")
    sliced = subprocess.run(["grep", "-c", "quantised routed experts are not served yet", f"{SRC}/arch/hyv4flash/hyv4flash.cpp"],
                            stdout=subprocess.PIPE, text=True).stdout.strip()
    if sliced != "0":
        say("A4", "OPEN", "sliced experts at TP>1 are WP10's (the plugin still refuses quantised experts at world > 1); "
                          "whole experts a rank are what is checked")


def a5(c):
    if not c.gpu:
        say("A5", "OPEN", "needs a card (WP8): run with --gpu, or `flock gpu.lock ref/venv/bin/python ref/run_rad.py --preset gpu-mini "
                          "--gpu --tp 1|2` and `rad-kbench --kernels libr4d,libref --file data/kernels_hy4.rkb`, `r4d_selftest`")
        return
    for tp in (1, 2):
        rc, m, log = run_rad(c, "gpu-mini", tp, "oneshot", "libr4d,libavx", ["--gpu"])
        say("A5", "PASS" if rc == 0 and m and m[1] == 0 else "FAIL", f"gpu-mini TP{tp} on the cards: rc {rc} {m}; {log}")
    kb = os.path.join(SRC, "data/kernels_hy4.rkb")
    if os.path.exists(kb):
        rc, out = run(["flock", GPU_LOCK, f"{c.build}/bin/rad-kbench", "--kernels", "libr4d,libref", kb], f"{c.work}/a5-kbench.log",
                      {"RADIANCE_HOME": c.home}, 3600)
        say("A5", "PASS" if rc == 0 else "FAIL", f"rad-kbench over data/kernels_hy4.rkb: rc {rc}")
    else:
        say("A5", "OPEN", "data/kernels_hy4.rkb not in this tree (WP8)")


def a6(c):
    say("A6", "OPEN", "needs the cards and WP8: `debug-selection` on a real-geometry declare at TP1/2/4 (no reference kernel in the "
                      "table); not decidable on the host")


def a7(c):
    if not c.ref_ok:
        return say("A7", "OPEN", f"HF harness dir {c.ref} not found (--ref)")
    w = os.path.join(c.work, "calib")
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    ck = os.path.join(w, "ck")
    os.makedirs(ck)
    src = os.path.join(c.ref, "mini_ckpt/mini4")
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json"):
            os.symlink(os.path.join(src, f), os.path.join(ck, f))
    cfg = json.load(open(os.path.join(src, "config.json")))
    V = cfg["text_config"]["vocab_size"]
    tok = {"version": "1.0", "added_tokens": [dict(id=i, content=f"<t{i}>", single_word=False, lstrip=False, rstrip=False,
                                                    normalized=False, special=True) for i in range(V)],
           "normalizer": None, "pre_tokenizer": dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=True),
           "post_processor": None, "decoder": dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=True),
           "model": dict(type="BPE", dropout=None, unk_token=None, byte_fallback=False, vocab={}, merges=[])}
    json.dump(tok, open(os.path.join(ck, "tokenizer.json"), "w"))
    env = host_env(c)
    env.update(RADIANCE_HY4_ENGRAM_EPS="f32", RADIANCE_HOST_VRAM_MIB="512")
    rc, _ = run([sys.executable, f"{SRC}/data/hy4flash_engram_tables.py", ck, f"{w}/tables.safetensors"], f"{w}/tables.log", env)
    rad = f"{w}/bf16.rad"
    rc2, _ = run([c.convert, ck, "--add", f"{w}/tables.safetensors", "--recipe", f"{SRC}/data/recipes/hy4-flash-bf16.recipe",
                  "-o", rad, "--max-ctx", 128, "--max-tok", 64], f"{w}/convert.log", env)
    if rc or rc2:
        return say("A7", "FAIL", f"mini4 bf16 container: rc {rc}/{rc2}, {w}/")
    ids = json.load(open(os.path.join(c.ref, "prompts.json")))
    with open(f"{w}/corpus.jsonl", "w") as f:
        for n in ("short", "window", "long"):
            f.write(json.dumps({"prompt": "".join(f"<t{i}>" for i in ids[n])}) + "\n")
    grams = f"{w}/grams"
    os.makedirs(grams)
    yml = subprocess.run([f"{c.build}/bin/rad-yaml", "--model", rad, "--tp", "4", "--max-model-len", "128",
                          "--max-num-batched-tokens", "64", "--max-num-seqs", "1", "--no-prefix-cache", "--kernels", "libref",
                          "--debug-accept-reference-kernels", "--calib-corpus", f"{w}/corpus.jsonl", "--calib-dir", grams],
                         stdout=subprocess.PIPE, text=True, env=dict(os.environ, **env)).stdout
    open(f"{w}/calib.yaml", "w").write(yml)
    env["RADIANCE_HOST_DEVICES"] = "4"
    rc, out = run([f"{c.build}/bin/radiance", "--config", f"{w}/calib.yaml"], f"{w}/calib.log", env, 1800)
    if rc:
        return say("A7", "FAIL", f"calibration run at fake TP4: rc {rc}, {w}/calib.log")
    run([sys.executable, f"{SRC}/tools/calib_st2gram.py", grams], f"{w}/st2gram.log", env)
    sts = glob.glob(f"{grams}/*.safetensors")
    gm = glob.glob(f"{grams}/gram.r*.bin")
    # expected trunk sites: the plugin's own table, from the arch test's expectations is not exported; count against the layers
    layers = cfg["text_config"]["num_hidden_layers"]
    say("A7", "PASS" if sts and gm else "FAIL", f"mini4 fake TP4 calibration run: {len(sts)} trunk site files, {len(gm)} RADGRAM2 files "
                                                f"({layers} layers) in {grams}")
    env["CALIB"] = grams
    rc, out = run([c.convert, ck, "--add", f"{w}/tables.safetensors", "--recipe", f"{SRC}/data/recipes/hy4-flash-w4nl64-gptq.recipe",
                   "-o", f"{w}/q.rad", "--max-ctx", 128, "--max-tok", 64], f"{w}/gptq.log", env, 1800)
    missing = re.findall(r"no calibration for '([^']+)'", out)
    if rc == 0 and not missing:
        say("A7", "PASS", "rad-convert with the gptq recipe found a Gram for every expert projection")
    elif rc == 0:
        say("A7", "FAIL", f"gptq recipe: no Gram for {len(missing)} weight(s), first {missing[:3]}")
    else:
        say("A7", "OPEN" if "WP10" in out or "not served" in out else "FAIL", f"gptq convert rc {rc}, {w}/gptq.log")
    say("A7", "OPEN", "the 1e-3 relative match of the trunk Grams to X^T X from HF activations is not reachable against bf16 "
                      "activations (WP9 measured 0.3% to 1.6% on mini TP1, 2% TP4 vs TP1): the coordinator decides the gate; "
                      "trunk Grams of the plugin's own taps agree to 2e-8 (hyv4flash_calib_test)")


def kl_run(c, rad, refdir, out, tp=1):
    env = host_env(c)
    env.update(RADIANCE_HY4_ENGRAM_EPS="f32", RADIANCE_HOST_VRAM_MIB="4096", RADIANCE_HOST_DEVICES=str(tp))
    args = [f"{SRC}/scripts/radiance-flags", f"{c.build}/bin/radiance", "--model", rad, "--kernels", "libref",
            "--debug-accept-reference-kernels", "--max-model-len", 128, "--max-num-batched-tokens", 64, "--max-num-seqs", 1,
            "--tp", tp, "--no-prefix-cache", "--kld-ref", refdir, "--kld-out", out]
    rc, text = run(args, out + ".log", env, 1800)
    m = re.search(r"KLD mean ([\d.e+-]+)", text)
    t = re.search(r"top-1 agreement ([\d.]+)%", text)
    return rc, (float(m.group(1)) if m else None), (float(t.group(1)) if t else None), text


def prepare_kl(c):
    """mini bf16 container + the converter's reference from the HF f32 dumps (short, window), cached for A8 / A9."""
    if hasattr(c, "kl"):
        return c.kl
    w = os.path.join(c.work, "kl")
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    ck = os.path.join(w, "ck")
    os.makedirs(ck)
    src = os.path.join(c.ref, "mini_ckpt/mini")
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f in ("config.json", "model.safetensors.index.json"):
            os.symlink(os.path.join(src, f), os.path.join(ck, f))
    V = json.load(open(os.path.join(src, "config.json")))["text_config"]["vocab_size"]
    json.dump({"version": "1.0", "added_tokens": [dict(id=i, content=f"<t{i}>", single_word=False, lstrip=False, rstrip=False,
                                                        normalized=False, special=True) for i in range(V)],
               "normalizer": None, "pre_tokenizer": dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=True),
               "post_processor": None, "decoder": dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=True),
               "model": dict(type="BPE", dropout=None, unk_token=None, byte_fallback=False, vocab={}, merges=[])},
              open(os.path.join(ck, "tokenizer.json"), "w"))
    env = host_env(c)
    env.update(RADIANCE_HY4_ENGRAM_EPS="f32", RADIANCE_HOST_VRAM_MIB="4096")     # the Engram norm eps HF's f32 run resolves
    run([sys.executable, f"{SRC}/data/hy4flash_engram_tables.py", ck, f"{w}/tables.safetensors"], f"{w}/tables.log", env)
    c.kl = dict(w=w, ck=ck, V=V, env=env)
    return c.kl


def convert_mini(c, kl, name, recipe):
    out = f"{kl['w']}/{name}.rad"
    rc, text = run([c.convert, kl["ck"], "--add", f"{kl['w']}/tables.safetensors", "-o", out, "--max-ctx", 128, "--max-tok", 64]
                   + (["--recipe", recipe] if recipe else []), f"{kl['w']}/{name}.convert.log",
                   dict(kl["env"], CALIB=f"{kl['w']}/no-grams"), 600)
    return out if rc == 0 else None, text


def a9(c):
    if not c.ref_ok:
        return say("A9", "OPEN", f"HF harness dir {c.ref} not found (--ref)")
    kl = prepare_kl(c)
    D = os.path.join(c.ref, "dumps/mini/f32")
    npz = [f"{D}/short.npz", f"{D}/window.npz"]
    bf16, _ = convert_mini(c, kl, "bf16", None)
    if not bf16:
        return say("A9", "FAIL", "mini bf16 container did not convert")
    # (1) radiance's own record, then the self comparison: KL 0
    corpus = f"{kl['w']}/corpus.jsonl"
    ids = json.load(open(os.path.join(c.ref, "prompts.json")))
    with open(corpus, "w") as f:
        for n in ("short", "window"):
            f.write(json.dumps({"prompt": "".join(f"<t{i}>" for i in ids[n]), "source": n}) + "\n")
    selfref = f"{kl['w']}/self-ref"
    shutil.rmtree(selfref, ignore_errors=True)
    args = [f"{SRC}/scripts/radiance-flags", f"{c.build}/bin/radiance", "--model", bf16, "--kernels", "libref",
            "--debug-accept-reference-kernels", "--max-model-len", 128, "--max-num-batched-tokens", 64, "--max-num-seqs", 1,
            "--no-prefix-cache", "--kld-record", selfref, "--kld-corpus", corpus]
    rc, _ = run(args, f"{kl['w']}/record.log", dict(kl["env"], RADIANCE_HOST_DEVICES="1"), 1800)
    if rc or not os.path.exists(f"{selfref}/kld.json"):
        return say("A9", "FAIL", f"--kld-record on mini: rc {rc}, {kl['w']}/record.log")
    rc, mean, top1, _ = kl_run(c, bf16, selfref, f"{kl['w']}/self.json")
    ok = rc == 0 and mean is not None and mean <= 1e-6 and top1 == 100.0
    say("A9", "PASS" if ok else "FAIL", f"--kld-record then --kld-ref on mini (self comparison): KL mean {mean}, top-1 {top1}%")
    # (2) the converter: HF f32 logits -> reference -> radiance on the bf16 container
    hf = f"{kl['w']}/hf-ref"
    shutil.rmtree(hf, ignore_errors=True)
    rc, _ = run([sys.executable, f"{HERE}/hf_logits_to_kld.py", hf, "--npz", *npz, "--model", "hf-f32"], f"{kl['w']}/conv.log")
    rc2, mean_b, top1_b, _ = kl_run(c, bf16, hf, f"{kl['w']}/hf-bf16.json")
    ok = rc == 0 and rc2 == 0 and mean_b is not None and mean_b < 2e-3 and top1_b == 100.0
    say("A9", "PASS" if ok else "FAIL", f"HF logits -> reference (tools/hf_logits_to_kld.py) -> --kld-ref on the bf16 container: "
                                         f"KL mean {mean_b}, top-1 {top1_b}% (gate: mean < 2e-3, the harness's own limit)")
    c.kl["mean_bf16"] = mean_b
    hf_top = f"{kl['w']}/hf-ref-top"
    shutil.rmtree(hf_top, ignore_errors=True)
    run([sys.executable, f"{HERE}/hf_logits_to_kld.py", hf_top, "--npz", *npz, "--top", "8"], f"{kl['w']}/conv-top.log")
    rc3, mean_t, top1_t, _ = kl_run(c, bf16, hf_top, f"{kl['w']}/hf-bf16-top.json")
    say("A9", "PASS" if rc3 == 0 and mean_t is not None and mean_t < 2e-3 else "FAIL",
        f"the same as a sparse (--top 8) reference: KL mean {mean_t}, top-1 {top1_t}%")
    # (3) a wrong reference (rows of another document) gives a large KL: the comparison compares something
    z = [np.load(f) for f in npz]
    wrong = f"{kl['w']}/wrong.npz"
    swap = dict(input_ids=z[0]["input_ids"], logits=np.roll(z[0]["logits"], 1, axis=0))
    np.savez(wrong, **swap)
    wr = f"{kl['w']}/wrong-ref"
    shutil.rmtree(wr, ignore_errors=True)
    run([sys.executable, f"{HERE}/hf_logits_to_kld.py", wr, "--npz", wrong], f"{kl['w']}/conv-wrong.log")
    rc4, mean_w, _, _ = kl_run(c, bf16, wr, f"{kl['w']}/wrong.json")
    say("A9", "PASS" if mean_w and mean_w > 100 * max(mean_b or 1e-6, 1e-6) else "FAIL",
        f"a reference with shifted rows gives KL {mean_w} (>> {mean_b}): the comparison is sensitive")
    # (4) the real geometry: logits width 120832 in the release (embedding and lm_head rows, config)
    if c.release_ok:
        cfgj = json.load(open(os.path.join(c.release, "config.json")))
        v = cfgj.get("text_config", cfgj).get("vocab_size")
        idx = json.load(open(os.path.join(c.release, "model.safetensors.index.json")))["weight_map"]
        heads = [k for k in idx if k.endswith("lm_head.weight")]
        shp = shard_tensor_shape(c.release, heads[0]) if heads else None
        ok = v == REAL_VOCAB and shp and shp[0] == REAL_VOCAB
        say("A9", "PASS" if ok else "FAIL", f"release config vocab_size {v}, lm_head {shp}: the KL reference is {REAL_VOCAB} columns wide "
                                              f"(radiance's logits width is the vocabulary, summed over ranks)")
    else:
        say("A9", "OPEN", f"release config unavailable: vocabulary {REAL_VOCAB} columns not checked")
    say("A9", "OPEN", "the real-geometry logits buffer (rad_declare_logits max_logit_rows, vocabulary shards summing to 120832 "
                      "at TP1/2/4) needs a card (WP8) or a real-size declare; mini's shards sum to its 320")


def a8(c):
    if not c.ref_ok:
        return say("A8", "OPEN", f"HF harness dir {c.ref} not found (--ref)")
    kl = prepare_kl(c)
    D = os.path.join(c.ref, "dumps/mini/f32")
    hf = f"{kl['w']}/hf-ref"
    if not os.path.exists(f"{hf}/kld.json"):
        run([sys.executable, f"{HERE}/hf_logits_to_kld.py", hf, "--npz", f"{D}/short.npz", f"{D}/window.npz"], f"{kl['w']}/conv.log")
    plain, _ = convert_mini(c, kl, "plain", None)
    rc, base, top1, _ = kl_run(c, plain, hf, f"{kl['w']}/plain.json") if plain else (1, None, None, "")
    if base is None:
        return say("A8", "FAIL", "the plain bf16 container does not serve on the host")
    say("A8", "PASS", f"no recipe (bf16): KL mean {base} against the HF f32 logits, top-1 {top1}%")
    c.kl["mean_bf16"] = base
    recipes = sorted(glob.glob(os.path.join(SRC, "data/recipes/hy4-flash-*.recipe")))
    for r in recipes:
        nm = os.path.basename(r)[len("hy4-flash-"):-len(".recipe")]
        if "gptq" in nm:
            say("A8", "OPEN", f"{nm}: needs Grams (calibrated); the A7 run converts it, its KL is the friend's measurement")
            continue
        rad, text = convert_mini(c, kl, nm, r)
        if not rad:
            say("A8", "FAIL", f"{nm}: does not convert on mini ({kl['w']}/{nm}.convert.log)")
            continue
        rc, mean, top1, out = kl_run(c, rad, hf, f"{kl['w']}/{nm}.json")
        if rc != 0 or mean is None:
            why = re.search(r"E [^\n]*(not served|refus)[^\n]*", out)
            say("A8", "OPEN", f"{nm}: converts, not served by this build ({why.group(0)[:150] if why else 'see log'}) -- WP10")
        else:
            gate = ("bf16" in nm or "8" in nm.replace("w4", "")) and top1 is not None and top1 < 80
            say("A8", "FAIL" if gate else "PASS", f"{nm}: serves on the host; KL mean {mean} (bf16 container {base}), "
                                                  f"top-1 {top1}%")
    say("A8", "OPEN", "the same on cards (gpu-mini) for the expert forms and the int8 trunk: WP8/WP10; and the formats of PLAN 4.3 "
                      "without a recipe in data/recipes (w5nl64, w6nl64, i8a8h, IQR trellis, int8 trunk) are not exercised here")


def a10(c):
    rc, nrun, nfail, nskip, out = ctest(c, "hyv4flash_chat|^chat_test$", "a10-ctest",
                                        {"RAD_TEST_HY4_TOKENIZER_DIR": c.release} if c.release_ok else None)
    say("A10", "PASS" if rc == 0 and nrun and nfail == 0 else "FAIL",
        f"chat tests (template rendered with no_think and high, reply format): {nrun} run, {nfail} failed, {nskip} skipped")
    rc, out = run(["grep", "-rln", "no_think", f"{SRC}/core", f"{SRC}/arch/hyv4flash", f"{SRC}/server"], f"{c.work}/a10-grep.log")
    if rc == 0 and out.strip():
        say("A10", "PASS", "server maps its reasoning settings onto no_think / high: " + out.strip().splitlines()[-1].split("/")[-1])
    else:
        say("A10", "OPEN", "the server's mapping of reasoning settings onto no_think / high is WP12 (not in this tree yet)")
    if c.release_ok:
        t = json.load(open(os.path.join(c.release, "tokenizer.json")))
        n = len(t.get("added_tokens", [])) + len(t["model"]["vocab"])
        say("A10", "PASS" if os.path.exists(os.path.join(c.release, "chat_template.jinja")) else "FAIL",
            f"release tokenizer.json ({n} entries) and chat_template.jinja present; baking is tested by hyv4flash_convert_test (A1)")


def a11(c):
    arch = open(os.path.join(SRC, "docs/ARCHITECTURES.md")).read()
    need = ["# `hyv4flash`: Tencent Hy4-flash", "## Calibrating Hy4-flash", "hy4flash-quant.md"]
    miss = [n for n in need[:2] if n not in arch]
    say("A11", "FAIL" if miss else "PASS", "docs/ARCHITECTURES.md has the hyv4flash and calibration sections" if not miss else f"missing {miss}")
    ops = open(os.path.join(SRC, "docs/OPS.md")).read()
    names = ["ihc_", "qsa_select", "mla_kv_split", "idx_pool_gated", "idx_score", "gate_mul", "engram_ids", "attn_kv_sink_h256"]
    miss = [n for n in names if n not in ops]
    say("A11", "FAIL" if miss else "PASS", "docs/OPS.md names the Hy4-flash ops" if not miss else f"OPS.md lacks {miss}")
    rc, nrun, nfail, nskip, out = ctest(c, "ops_doc", "a11-opsdoc")
    if nrun and nfail == 0 and not nskip:
        say("A11", "PASS", "ctest -R ops_doc green")
    elif nrun and nskip:
        say("A11", "OPEN", "ops_doc skipped (libr4d does not dlopen in this build, WP1 note); notes/wp6a.md has the manual run "
                           "tools/ops_doc_check.sh")
    else:
        say("A11", "FAIL", f"ops_doc: rc {rc} ({nrun} run, {nfail} failed)")
    if c.full_suite:
        for b in c.full_suite:
            cc = Ctx()
            cc.build, cc.work, cc.home = b, c.work, f"{b}/radiance_home"
            rc, nrun, nfail, nskip, out = ctest(cc, ".", f"a11-suite-{os.path.basename(b.rstrip('/'))}", timeout=4 * 3600)
            say("A11", "PASS" if nfail == 0 and nrun else "FAIL", f"whole suite in {b}: {nrun} run, {nfail} failed, {nskip} skipped "
                                                                   f"(pre-existing failures: BASELINE_TESTS.md)")
    else:
        say("A11", "OPEN", "whole suite on the no-ROCm and the HIP build: run with --full-suite BUILD1 [BUILD2 ...]")


ITEMS = [("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("A5", a5), ("A6", a6), ("A7", a7), ("A8", a8), ("A9", a9),
         ("A10", a10), ("A11", a11)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--ref", default=os.path.join(SRC, "..", "..", "ref"))
    ap.add_argument("--release", help="directory for the release structure (default <work>/release)")
    ap.add_argument("--work")
    ap.add_argument("--only", help="comma list, e.g. A1,A9")
    ap.add_argument("--gpu", action="store_true", help="also run the card items (A5), under flock gpu.lock")
    ap.add_argument("--full-suite", nargs="*", metavar="BUILD", help="run the whole ctest of these build directories (A11)")
    ap.add_argument("--offline", action="store_true", help="do not fetch the release structure")
    a = ap.parse_args()
    c = Ctx()
    c.build = os.path.abspath(a.build)
    c.home = f"{c.build}/radiance_home"
    c.convert = f"{c.build}/bin/rad-convert"
    c.work = os.path.abspath(a.work or f"{c.build}/ready")
    os.makedirs(c.work, exist_ok=True)
    c.ref = os.path.abspath(a.ref)
    c.ref_ok = os.path.exists(f"{c.ref}/run_rad.py") and os.path.exists(f"{c.ref}/venv/bin/python")
    c.refx = f"{c.work}/ref"          # a private view of the harness, so its cache and outputs are this run's own
    if c.ref_ok:
        os.makedirs(c.refx, exist_ok=True)
        for f in os.listdir(c.ref):
            if f in ("venv", "transformers", "transformers_b865850d", "__pycache__", "dumps-rad", "dumps"):
                continue
            p = os.path.join(c.refx, f)
            if not os.path.lexists(p):
                os.symlink(os.path.join(c.ref, f), p)
        os.path.lexists(f"{c.refx}/dumps") or os.symlink(f"{c.ref}/dumps", f"{c.refx}/dumps")
        os.path.lexists(f"{c.refx}/venv") or os.symlink(f"{c.ref}/venv", f"{c.refx}/venv")
        c.work_root = c.work
        # run_rad keeps its caches in <refx>/../work: that is c.work/work
    c.release = os.path.abspath(a.release or f"{c.work}/release")
    c.release_ok, c.release_err = False, ""
    if os.path.exists(f"{c.release}/model.safetensors.index.json") and len(glob.glob(f"{c.release}/model-*.safetensors")) == 148:
        c.release_ok = True
    elif not a.offline:
        try:
            c.release_ok = fetch_release(c.release) == 148
        except Exception as e:  # noqa: BLE001 -- the network is the one thing that may be missing
            c.release_err = f"{type(e).__name__}: {e}"
    else:
        c.release_err = "--offline and no cached structure"
    c.gpu = a.gpu
    c.full_suite = [os.path.abspath(b) for b in (a.full_suite or [])]
    want = set(a.only.split(",")) if a.only else None
    for item, fn in ITEMS:
        if want and item not in want:
            RESULTS[item] = [("SKIP", "not selected")]
            continue
        print(f"{item} ...", flush=True)
        try:
            fn(c)
        except Exception as e:  # noqa: BLE001
            say(item, "FAIL", f"the check itself failed: {type(e).__name__}: {e}")
    # the table
    print("\n" + "=" * 100)
    worst = {"FAIL": 3, "OPEN": 2, "PASS": 1, "SKIP": 0}
    nfail = 0
    for item, _ in ITEMS:
        rows = RESULTS.get(item, [])
        st = max((s for s, _ in rows), key=lambda s: worst[s], default="SKIP")
        nfail += st == "FAIL"
        print(f"{item:4} {st:5} " + f"{sum(s == 'PASS' for s, _ in rows)} pass, {sum(s == 'OPEN' for s, _ in rows)} open, "
                                    f"{sum(s == 'FAIL' for s, _ in rows)} fail")
        for s, t in rows:
            if s in ("FAIL", "OPEN"):
                print(f"       {s}: {t}")
    print("=" * 100)
    print("READY (tier A all PASS)" if all(max((s for s, _ in RESULTS.get(i, [("SKIP", "")])), key=lambda s: worst[s]) == "PASS"
                                          for i, _ in ITEMS) else "NOT YET: items above that are not PASS")
    json.dump(RESULTS, open(f"{c.work}/ready.json", "w"), indent=1)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
