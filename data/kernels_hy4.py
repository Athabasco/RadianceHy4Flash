#!/usr/bin/env python3
"""data/kernels_hy4.rkb, the rad-kbench fixture of Hy4-flash's ops at real per-head geometry (WP8).

No container declares every shape: the gpu-mini container (ref/ harness, PLAN 1.1: the real heads,
latent, rope, window, indexer and Engram dims at n = 512) gives the graph's points, M 1..1024;
WP6b's hand-written MLA-with-sink cases give the real contexts; clones of the gpu-mini points give
what gpu-mini cannot declare: the TP2 / TP4 head shares (32 / 64 -> 16 / 8 heads of the absorbed and
non-absorbed attention, the gate and the kv_b projection), the real stream n = 3072 (iHC, the Engram
gate) and M = 6 (a verify of 1 + 5 drafts). Rebuild (host only, libref answers):

  rad-kbench -m gpu-mini.rad --kernels libref --max-ctx 4096 --max-tok 1024 --record gpumini.rkb
  python3 data/kernels_hy4.py raw gpumini.rkb raw_extra.rkb
  rad-kbench --fixture raw_extra.rkb --kernels libref --rerecord extra.rkb
  python3 data/kernels_hy4.py merge gpumini.rkb hy4_mla_sink.rkb extra.rkb data/kernels_hy4.rkb

hy4_mla_sink.rkb is WP6b's (notes/wp6b-mkfix.py, rerecorded). The vision rows (WP15):

  rad-kbench -m gpu-mini-vis.rad --kernels libref --max-ctx 4096 --max-tok 1024 --mm-max-patches 4096 --record vis.rkb
  python3 data/kernels_hy4.py vis data/kernels_hy4.rkb vis.rkb data/kernels_hy4.rkb

Check a card with
  rad-kbench --fixture data/kernels_hy4.rkb --kernels libr4d:libref [--bench]
"""
import copy, struct, sys
ENC = 568   # sizeof(RadEncoding)
class R:
    def __init__(s, b): s.b, s.p = b, 0
    def take(s, n): v = s.b[s.p:s.p + n]; s.p += n; return v
    def u8(s): return s.take(1)[0]
    def u32(s): return struct.unpack('<I', s.take(4))[0]
    def u64(s): return struct.unpack('<Q', s.take(8))[0]
    def i64(s): return struct.unpack('<q', s.take(8))[0]
    def f64(s): return struct.unpack('<d', s.take(8))[0]
    def str(s): return s.take(s.u32()).decode()
def load(path):
    r = R(open(path, 'rb').read())
    assert r.take(4) == b'RKB1'
    fx = dict(version=r.u32(), samples=r.u32(), source=r.str(), oracle=r.str(), recorded=r.str(), cases=[])
    for _ in range(r.u32()):
        c = dict(op=r.str(), seed=r.u64(), m=r.i64(), band=r.str(), params=[], opd=[])
        for _ in range(r.u32()):
            c['params'].append([r.str(), r.u32(), r.u64(), r.f64(), r.str()])
        for _ in range(r.u32()):
            o = dict(role=r.u32(), dtype=r.u32())
            o['shape'] = [r.i64() for _ in range(r.u32())]
            o['absent'] = r.u8(); o['fill'] = r.u32(); o['flags'] = r.u8()
            o['idx_max'] = r.i64(); o['idx_const'] = r.i64(); o['fill_chunk'] = r.i64()
            o['has_enc'] = r.u8()
            if o['has_enc']:
                o['enc'] = r.take(ENC); o['sel'] = [(r.u32(), r.i64(), r.i64()) for _ in range(r.u32())]
            o['have_ref'] = r.u8()
            if o['have_ref']:
                o['ref'] = (r.i64(), r.f64(), r.i64(), r.take(4 * r.u32()), None)
                n = r.u32(); o['sketch'] = r.take(8 * n); o['nsk'] = n
            c['opd'].append(o)
        fx['cases'].append(c)
    assert r.p == len(r.b)
    return fx
def save(fx, path):
    out = []
    P = lambda f, v: out.append(struct.pack(f, v))
    def s(x): b = x.encode(); P('<I', len(b)); out.append(b)
    out.append(b'RKB1'); P('<I', 6); P('<I', 1024); s(fx['source']); s(fx['oracle']); s(fx['recorded'])
    P('<I', len(fx['cases']))
    for c in fx['cases']:
        s(c['op']); P('<Q', c['seed']); P('<q', c['m']); s(c['band'])
        P('<I', len(c['params']))
        for k, kind, i, d, st in c['params']:
            s(k); P('<I', kind); P('<Q', i & 0xffffffffffffffff); P('<d', d); s(st)
        P('<I', len(c['opd']))
        for o in c['opd']:
            P('<I', o['role']); P('<I', o['dtype']); P('<I', len(o['shape']))
            for x in o['shape']: P('<q', x)
            P('<B', o['absent']); P('<I', o['fill']); P('<B', o['flags'])
            P('<q', o['idx_max']); P('<q', o['idx_const']); P('<q', o['fill_chunk'])
            P('<B', o['has_enc'])
            if o['has_enc']:
                out.append(o['enc']); P('<I', len(o['sel']))
                for a, b, c2 in o['sel']: P('<I', a); P('<q', b); P('<q', c2)
            hr = o.get('have_ref', 0)
            P('<B', hr)
            if hr:
                n, ss, nf, smp, _ = o['ref']; P('<q', n); P('<d', ss); P('<q', nf)
                P('<I', len(smp) // 4); out.append(smp); P('<I', o['nsk']); out.append(o['sketch'])
    open(path, 'wb').write(b''.join(out))
def par(c, k):
    for p in c['params']:
        if p[0] == k: return p[2] if p[1] == 1 else (p[3] if p[1] == 4 else p[4])
    return None
HY = ['ihc_pre', 'ihc_post', 'gate_mul', 'mla_project_kv', 'mla_kv_split', 'mla_attn_paged_sink', 'mla_attn_sel_sink',
      'attn_kv_sink', 'idx_pool_gated', 'idx_score', 'engram_norm', 'engram_gate', 'embed_lookup_q', 'quant_act_fp8',
      'dequant', 'qsa_select', 'qsa_work', 'qsa_tail_store', 'swiglu_clamped', 'router_sigmoid_topk']
HEADS = ['mla_project_kv', 'mla_kv_split', 'mla_attn_paged_sink', 'mla_attn_sel_sink', 'attn_kv_sink', 'gate_mul']
WIDE = ['ihc_pre', 'ihc_post', 'engram_gate']

def setp(c, k, v):
    for p in c['params']:
        if p[0] == k:
            if p[1] == 1: p[2] = v
            else: p[3] = float(v)
def fresh(c, salt):
    c = copy.deepcopy(c)
    c['seed'] = (c['seed'] * 0x9E3779B97F4A7C15 + salt) & 0xffffffffffffffff
    for o in c['opd']:
        o['have_ref'] = 0; o.pop('ref', None)
        if o['has_enc']: o['has_enc'] = 0; o.pop('enc', None); o.pop('sel', None)   # drawn plain
    return c

def heads_clone(c, H2, salt):
    H = par(c, 'heads') if c['op'] != 'gate_mul' else par(c, 'n') // 256
    d = fresh(c, salt)
    m = {H: H2, H * 448: H2 * 448, H * 256: H2 * 256}
    for o in d['opd']:
        sh = o['shape']
        if c['op'] == 'gate_mul' and len(sh) == 2: sh[1] = m.get(sh[1], sh[1])
        elif len(sh) == 3 and sh[1] == H: sh[1] = H2                              # q / k / v / out [M, H, d]
        elif o['role'] == 3 and len(sh) == 1 and sh[0] == H: sh[0] = H2           # sinks
        elif o['role'] == 3 and len(sh) == 2 and sh[0] == H * 448: sh[0] = H2 * 448   # kv_b
        elif c['op'] == 'mla_kv_split' and len(sh) == 2 and sh[1] == H * 448: sh[1] = H2 * 448
    if c['op'] == 'gate_mul': setp(d, 'n', H2 * 256)
    else: setp(d, 'heads', H2)
    d['band'] = f"{c['band']} heads {H2}"
    return d

def wide_clone(c, n2, salt):
    n = par(c, 'n'); hc = par(c, 'hc')
    d = fresh(c, salt)
    m = {n: n2, n * hc: n2 * hc}
    for o in d['opd']: o['shape'] = [m.get(x, x) if i > 0 or len(o['shape']) == 1 else x for i, x in enumerate(o['shape'])]
    setp(d, 'n', n2); d['band'] = f"{c['band']} n {n2}"
    return d

def m_clone(c, M2, salt):
    M = c['m']; d = fresh(c, salt)
    for o in d['opd']:
        if o['shape'] and o['shape'][0] == M and not (o['flags'] & 4) and o['role'] != 3: o['shape'][0] = M2
        if o['idx_max'] == M and (o['flags'] & 4): o['idx_max'] = M2
    d['m'] = M2; setp(d, 'M', M2); d['band'] = f"{c['band']} M {M2}"
    return d

def raw(src, out):
    fx = load(src)
    base = [c for c in fx['cases'] if c['op'] in HEADS + WIDE]
    new, salt = [], 1
    for c in base:
        if c['m'] not in (1, 8, 64, 1024): continue
        if c['op'] in HEADS:
            H = par(c, 'heads') if c['op'] != 'gate_mul' else par(c, 'n') // 256
            for H2 in (H // 2, H // 4) if H == 32 or H == 64 else ():
                if c['op'] == 'attn_kv_sink' and c['m'] == 1024 and H2 == 8: continue
                new.append(heads_clone(c, H2, salt)); salt += 1
        if c['op'] in WIDE and par(c, 'n') == 512:
            new.append(wide_clone(c, 3072, salt)); salt += 1
        if c['m'] == 8 and c['op'] in ('mla_project_kv', 'mla_attn_paged_sink', 'gate_mul', 'ihc_pre', 'ihc_post'):
            new.append(m_clone(c, 6, salt)); salt += 1
    seen, uniq = set(), []
    for c in new:
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k not in seen: seen.add(k); uniq.append(c)
    new = uniq
    x = dict(fx, source='hy4-flash TP shards / real n / M 6 (WP8, clones of the gpu-mini graph)', cases=new)
    save(x, out); print(len(new), 'cases')

def merge(*paths):
    out = paths[-1]; cases = []
    for i, p in enumerate(paths[:-1]):
        fx = load(p)
        seen = set()
        for c in fx['cases']:
            if i == 0 and c['op'] not in HY: continue
            if not any(o.get('have_ref') for o in c['opd']): continue   # the oracle refused it at record
            k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
            if i == 0 and k in seen: continue      # the same point a layer later: one is enough
            seen.add(k); cases.append(c)
    head = dict(load(paths[0]), source='hy4-flash: gpu-mini graph + real geometry / TP shards (WP8)', cases=cases)
    save(head, out); print(len(cases), 'cases ->', out)

# WP15: the vision rows, from a recording of the gpu-mini-vis container with --mm-max-patches (the tower
# at the real head width 72, its merger, the newline lookup, the media rows' scatter, and the trunk's
# and the indexer's M-RoPE forms).
VIS = ('grid_embed_sf', 'conv2d_unfold', 'attn_dense', 'layernorm', 'gemm_nt_bias', 'scatter_rows')
def vis_case(c):
    if c['op'] in VIS: return True
    if c['op'] == 'embed_lookup': return par(c, 'n_vocab') == 1          # the newline row
    if c['op'] in ('rope', 'idx_pool_gated'):
        return any(p[0] == 'mode' and p[4] == 'imrope_gptj' for p in c['params'])
    return False

def vis(base, rec, out):
    """base fixture + the vision cases of `rec` (rad-kbench -m gpu-mini-vis.rad --mm-max-patches 4096
    --kernels libref --record rec), each distinct point once."""
    fx, r = load(base), load(rec)
    have = {(c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
            for c in fx['cases']}
    add = []
    for c in r['cases']:
        if not vis_case(c) or not any(o.get('have_ref') for o in c['opd']): continue
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k in have: continue
        have.add(k); add.append(c)
    fx = dict(fx, source=fx['source'] + ' + vision (WP15, gpu-mini-vis)', cases=fx['cases'] + add)
    save(fx, out); print(len(add), 'vision cases added ->', out, len(fx['cases']), 'in all')

if __name__ == '__main__':
    {'raw': raw, 'merge': merge, 'vis': vis}[sys.argv[1]](*sys.argv[2:])
