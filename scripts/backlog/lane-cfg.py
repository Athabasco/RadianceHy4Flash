#!/usr/bin/env python3
"""lane-cfg.py <cfg.yaml> <out.yaml>: a backlog config with the lane's settings put in.

The configs in scripts/backlog/*.yaml are written for the served (local) lane. A lane run rewrites
the keys that differ per lane, from the environment lane.sh exports, and leaves every other
line of the file alone, so a run's exact config sits next to its numbers:

    RAD_PORT               server.port
    RAD_MODEL              models.*.model
    RAD_HOST_POOL_MIB      models.*.host-pool-mib (empty: keep the file's)
    RAD_NGRAM              models.*.ngram-placement (empty: keep the file's)
    RAD_PREFIX_CACHE_MIB   models.*.prefix-cache-mib (empty: keep the file's)
    RAD_CHUNK_SLICES       models.*.chunk-slices (the S-02 lever: a mixed step's chunk in m
                           sub-steps; empty: keep the file's). Inserted under each model's
                           `model:` line where the file does not name it.
    RAD_CHUNK_SLICES_PREFILL
                           models.*.chunk-slices-prefill (the S-02 test hook; empty: keep the
                           file's). Inserted the same way, so a run sets the hook without a second
                           config file.
    RAD_LOG                server.log (empty: keep the file's). Inserted under `server:` where the
                           file does not name it; radiance refuses flags beside --config, so a
                           run that needs Trace (the buffer probe) sets it here.
"""
import os, re, sys

KEYS = {"port": "RAD_PORT", "model": "RAD_MODEL",
        "host-pool-mib": "RAD_HOST_POOL_MIB", "ngram-placement": "RAD_NGRAM",
        "prefix-cache-mib": "RAD_PREFIX_CACHE_MIB"}

# Model keys a run sets without a second config file: inserted under each model's `model:` line
# where the file does not name them, rewritten in place where it does. chunk-slices is the lever
# (S-02's mixed-step slice); chunk-slices-prefill is the S-02b test hook.
PER_MODEL = {"chunk-slices": "RAD_CHUNK_SLICES",
             "chunk-slices-prefill": "RAD_CHUNK_SLICES_PREFILL"}
LEVEL = "RAD_LOG"


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: lane-cfg.py <cfg.yaml> <out.yaml>")
    src, out = sys.argv[1], sys.argv[2]
    want = {k: os.environ.get(v, "") for k, v in KEYS.items()}
    per_model = {k: os.environ.get(v, "") for k, v in PER_MODEL.items()}
    src_lines = open(src, errors="replace").read().splitlines(True)
    # A file that names a per-model key somewhere is rewritten in place; one that does not gets the
    # key added under each model's `model:` line.
    add = {k: bool(v) and not any(re.match(r"\s+" + re.escape(k) + r":", ln) for ln in src_lines)
           for k, v in per_model.items()}
    level = os.environ.get(LEVEL, "")
    add_level = bool(level) and not any(re.match(r"\s+log:", ln) for ln in src_lines)
    seen, lines = set(), []
    for ln in src_lines:
        m = re.match(r"(\s*)([a-z0-9-]+):(\s*)(.*)$", ln.rstrip("\n"))
        key = m.group(2) if m else ""
        # The key's own value first, then the two keys handled here; a line either branch rewrites
        # still gets its neighbours inserted below it, which is how the hook reaches a model whose
        # `model:` line the lane rewrites (RAD_MODEL always does).
        wrote = None
        if m and key in want and want[key]:
            wrote = f"{m.group(1)}{key}: {want[key]}\n"
            seen.add(key)
        elif m and key in per_model and per_model[key]:
            wrote = f"{m.group(1)}{key}: {per_model[key]}\n"
        elif m and key == "log" and level:
            wrote = f"{m.group(1)}log: {level}\n"
        lines.append(ln if wrote is None else wrote)
        indent = m.group(1) if m else ""
        if key == "model" and len(indent) >= 2:
            for k in PER_MODEL:
                if add[k]:
                    lines.append(f"{indent}{k}: {per_model[k]}\n")
        if add_level and key == "server":
            lines.append(f"{indent}  log: {level}\n")
    missing = sorted(k for k, v in want.items() if v and k not in seen)
    if missing:
        sys.exit(f"lane-cfg.py: {src} has no key(s) {', '.join(missing)} to set")
    with open(out, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
