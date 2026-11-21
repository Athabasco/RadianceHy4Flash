# Quantising Hy4-flash: what to run, in this tree

For whoever quantises `tencent/Hy4-flash` (bf16, 148 shards, 440 GB). Everything here is the short
form; the reference is [ARCHITECTURES.md, hyv4flash](ARCHITECTURES.md#hyv4flash-tencent-hy4-flash).
No real Hy4-flash weight has been through this tree: it was developed on random-weight dummies against the
transformers port (PR 50903, commit c2ec8505), so the first real run is a measurement, not a regression.

## What changed in the weight names since the first plugin commit (ce9ae1ff)

Convert once with the current tree; a container made earlier does not load. The declared set grew from
43699 to 43753 weights (`rad-convert --plan-only` over the release's 1964 tensors):

| before | now | why |
|---|---|---|
| `blk.l.attn_q_b` | `blk.l.attn_q_nope` + `blk.l.attn_q_rope` | the rope part of each head is rotated on its own; two linears over rows of the same `q_b_proj` (each head's 192 leading / 64 trailing rows). 42 layers |
| `blk.l.index_query` | `blk.l.index_query_nope` + `blk.l.index_query_rope` | the same split for the indexer's query (64 + 64 per head), the 12 sparse layers |
| `blk.l.ffn_gate_inp.{weight,correction}` | `blk.l.glm_router.{weight,correction}` | the router is read by the tree's GLM router block; f32 |
| (unchanged) | `attn_kv_b` | stays bf16 plain: the plugin projects through it as stored, a recipe naming it is refused by name |

Recipe patterns must use the new names: `blk.*.attn_q_nope.weight`, not `attn_q_b`. The routed experts,
Engram tables, DFly2 and the tower keep their names. Router, iHC, sinks, indexer pooling and the
position parameters are not quantisable (a recipe naming one is refused at declare).

## The commands

```sh
# 0. the declared set and sizes, no weights read (about 10 s; needs only the safetensors headers and config.json)
rad-convert <ckpt> --plan-only --recipe data/recipes/hy4-flash-w4nl64-rtn.recipe

# 1. bf16 container (the Engram hash constants first)
python3 data/hy4flash_engram_tables.py <ckpt> engram-tables.safetensors
rad-convert <ckpt> --add engram-tables.safetensors --recipe data/recipes/hy4-flash-bf16.recipe -o hy4-bf16.rad
#    a checkpoint larger than your disk: --defer / --in-place (TOOLS.md)

# 2. calibrate: ARCHITECTURES.md "Calibrating Hy4-flash" has the whole flow (rad-yaml, --calib-corpus,
#    --calib-dir, --calib-layers windows, tools/calib_st2gram.py, TP, the corpus). Short form:
radiance --config calib.yaml          # at --tp 4, from the bf16 container; then
python3 tools/calib_st2gram.py grams

# 3. quantise against the Grams (a recipe line with calib=$CALIB)
CALIB=$PWD/grams rad-convert <ckpt> --add engram-tables.safetensors \
    --recipe data/recipes/hy4-flash-w4nl64-gptq.recipe -o hy4-q.rad
```

Recipes in `data/recipes/`: `hy4-flash-bf16.recipe` (Engram E4M3 only), `hy4-flash-bf16-f32.recipe`
(the same at an f32 scale a row, code for code the port's f32 round trip), `hy4-flash-w4nl64-rtn.recipe`
(uncalibrated baseline, also the boot container a calibration can run from), `hy4-flash-w4nl64-gptq.recipe`,
`hy4-flash-w5nl64-rtn.recipe`, `hy4-flash-w6nl64-rtn.recipe`, `hy4-flash-i8.recipe` (trunk, experts and
head int8), `hy4-flash-iqr-gptq.recipe` (u4 trellis gate_up, w6nl64 down, u5 trellis trunk, int8 rest).
Write your own from them; ARCHITECTURES.md "Quantised Hy4-flash" lists every form the graph reads, per weight.

What is served (all of it measured only on random-weight dummies):

| weights | forms | note |
|---|---|---|
| routed experts | w4nl64, w5nl64, w6nl64 (rotated E4M3 activations); rotated int8, the IQR trellis, w4nl64 (rotated int8 activations); a w5nl64 / w6nl64 down behind a trellis gate_up; bf16 | SwiGLU clamp (gate <= 10, \|up\| <= 10) in every layer, every form: clamp in bf16, then quantise. TP2 / TP4: a slice of every expert (768 / 384 columns) |
| trunk linears | bf16; int8 rotated (`/fwht128`) or not; IQ8R; the i8a8h trellis (u5 ...); rotated u4/u5 with a zero point | each int8 linear quantises its own input; kv_a not as a trellis on the cards (576 rows) |
| lm_head | bf16, int8 | |
| Engram table | bf16, E4M3 rows at a bf16 or f32 scale | |
| kv_b, router, iHC, sinks, norms, indexer pooling, Engram projections, token embedding | as shipped | refused by name when a recipe touches them |

The IQR trellis (experts or trunk) is decoded by libr4d only: such a container serves on the cards, not
on the host. libavx has no w6nl64 kernel; on the host a w6nl64 container runs on libref.

## KL against your HF reference

`radiance --kld-ref DIR` scores a container against a directory of recorded log-probabilities
(TOOLS.md "The KL mode"; format in `core/kld.h`). radiance records its own with `--kld-record DIR
--kld-corpus corpus.jsonl`; yours comes from HF logits. Run the transformers port (c2ec8505) in bf16
over your token corpus, save each document's logits `[n_tokens, 120832]` as `.npy` (f32), and convert:

```sh
# corpus-ids.jsonl: {"ids": [...], "source": "code", "score_from": 0} a line; the ids HF saw (BOS, template tokens)
python3 tools/hf_logits_to_kld.py ref/ --docs corpus-ids.jsonl --logits-dir hf-logits/    # doc0.npy, doc1.npy ...
python3 tools/hf_logits_to_kld.py ref/ --docs corpus-ids.jsonl --logits-dir hf-logits/ --top 512   # sparse form for big corpora
rad-yaml --model hy4-q.rad --tp 4 ... --kld-ref ref/ --kld-out kl.json > cand.yaml && radiance --config cand.yaml
```

- Row `p` of a document's logits predicts token `p + 1`; positions `[score_from, n - 1)` are scored.
  radiance scores exactly the ids stored in `ref/tokens.i32`, so its tokeniser need not match yours.
- The width must equal radiance's logits width (120832; a wider HF logit matrix needs `--crop-vocab`,
  a narrower one is refused). The tool can also run the HF model itself (`--hf-model DIR`, CPU, slow).
- The full form is 240 KB a position (f16 over the vocabulary); `--top N` keeps the N most likely tokens and the tail
  mass (8 bytes a token) and scores a coarsened KL; numbers of the two forms do not compare.
- Run every candidate with the same flags. Reference and candidate need the same `--tp` only for the speed; the
  KL does not depend on it beyond rounding. Held-out text, not the calibration corpus.
- Tested on the dummy models: HF f32 logits of `mini` against radiance's bf16 container give KL about 1e-4
  (the parity harness's own noise) and 100 % top-1; shifted rows give a KL two orders larger.

## `tools/hy4flash_ready.py`

Checks the readiness list (A1..A11 of the plan) and prints a pass/fail table: build tests, plan-only over the
release structure (fetches the 148 headers once, ~10 MB), the host parity harness, calibration on `mini4`
at four fake ranks, the KL round trip, the chat tests. Items that need a card or a package not yet merged are
listed as OPEN with the command that decides them.

## Known limits

- The quantised forms above serve on the host (libref, libavx) at one, two and four (fake) ranks; on the
  cards only kernel by kernel so far (`r4d_selftest`): `gpu-mini` does not run end to end on a card yet.
  Calibrate from the bf16 container (the boot container with w4nl64 experts works too; its expert Grams
  are then in the rotated domain the GEMMs read).
- The Grams match `X^T X` of the plugin's own taps to 1e-7, of HF activations to 0.3 % to 1.6 % relative
  (bf16 activation rounding), TP4 against TP1 to 2 %.
- The indexer's selection ties (relu scores of 0) and the router's near-ties break differently from HF's `topk`;
  a parity comparison on random weights needs the conditioned reference of `ref/` (README there).
- DFly2 (speculation), the vision tower and the reply format (tool calls, reasoning markers) are not served;
  `--max-spec 0` is a `rad-convert` flag.
- `gpu-mini` does not start on a card in some builds (Engram round trip kernels); the real-geometry card
  numbers are not measured.
- Engram E4M3 rows are not exact against the port in bf16 (2.9 % of codes differ by one step: the port divides in
  bf16 before the cast). `hy4-flash-bf16-f32.recipe` is exact against the port in f32.
