# radiance · Hy4-flash

This is a fork of [radiance](https://codeberg.org/StillDeadcode/radiance), the C++/HIP inference
server for AMD GPUs, with a plugin for **Tencent Hy4-flash** (`hy_v4_flash`, 220B MoE). It adds the
`hyv4flash` architecture plugin and its kernels:

- MLA attention with sinks: 30 sliding-window layers and 12 sparse layers with a top-k indexer
- Engram n-gram tables, iHC hyper-connections, and the latent MoE (512 experts, top-16, plus a
  shared expert)
- DFly2 block-drafter speculative decoding, with coupled proposals, traversal verify and an
  adaptive verify depth
- tensor parallel at TP2 and TP4, calibration taps and the KL mode for quantising, and vision

It is built and tested on two Radeon AI PRO R9700 cards (gfx1201, 32 GB each), and on four of them
at TP4.

Quantised containers are made by a separate quantisation workflow. The one used below is
[`keptn/hy4-flash-iqr-v4`](https://huggingface.co/keptn/hy4-flash-iqr-v4)
(`hy4-flash-iqr-v4-u3.rad`, 128.6 GiB: u3 trellis experts, w4nl64 GPTQ attention); its README
lists the other variants and their KL figures.

For everything that is not Hy4-specific (other models, Docker, the full option list), see the
upstream README and [`docs/GUIDE.md`](docs/GUIDE.md). The Hy4 details and the TP2 recipe are in
[`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) (section "Hy4-flash"), the four-rank one with its
measurements in [`docs/TP4.md`](docs/TP4.md).

## Example config: two R9700 at TP2

```yaml
server:
  host: 127.0.0.1
  port: 8033
  kernels: libr4d,libavx,spechub      # spechub carries the DFly2 coupling / traversal ops
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: /srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad
    tp: 2
    placement: expert_tiered          # experts that do not fit stay in pinned host memory
    host-pool-mib: 39000
    ngram-placement: ram              # Engram tables in host RAM, off the cards
    max-model-len: 1000000
    max-num-batched-tokens: 2048
    decode-headroom: auto             # decode rows on top of the batch: an interval is one step
                                      # beside decoding sessions at the full 2048 rows (arrival
                                      # TTFT -15 % at one decoder, -12 % at four on iks-A) and the
                                      # chunks do not move, so the answers do not either. auto is
                                      # 64 rows here (7 x 1, rounded up): step_tok 2112, the
                                      # activation arena +43 MiB a rank
    checkpoint-interval: 1984         # a quantum under the batch: a chunk sharing its step with
                                      # decoding sessions lands on the interval boundary, so an
                                      # interval is one step instead of two (arrival TTFT -16 %
                                      # at one decoder, -12 % at four on the iks-A lane). Chunks
                                      # are a quantum shorter than 2048 rows, so a long prompt's
                                      # answer may differ from a 2048-chunk step at tie level (no
                                      # dtype, kernel or accumulation changes). Redundant beside
                                      # decode-headroom at 2048, which measured equal or tighter and
                                      # at the full chunk size; kept because S-01 delivered it and
                                      # it is what the notes here were measured on
    max-num-seqs: 8
    mm-max-patches: 0                 # no vision
    prefix-resume: nearest
    prefix-cache-mib: 16384           # host tier: finished conversations' KV moves to host RAM,
                                      # so the VRAM goes back to the experts
    busy-prefill-chunk: auto          # 512 halves decode stalls when long prompts arrive under load,
                                      # at about +60 % time to first token for those prompts
    checkpoint-slots: 128             # for many concurrent agentic sessions (default 64)
    num-speculative-tokens: 5         # DFly2 drafts at most 5 tokens
    draft-window: fixed
    draft-coupling: on
    draft-verify: traversal
    draft-depth: adaptive
    draft-depth-load: on
    draft-depth-load-from: 1          # depth rule priced per row, from one sequence
    draft-suffix: on
```

Run it with:

```sh
radiance --config hy4-tp2.yaml
```

Measured with this config (u3 container, TP2, 210 W power cap, sampled at T 0.9):

| | 1 client | 8 clients |
|---|---|---|
| decode without speculation | about 75 tok/s | about 305 tok/s |
| decode with DFly2 | about 140–150 tok/s | about 380–430 tok/s |
| prefill | 3000–4100 tok/s, depending on prompt length | |

## Example config: four R9700 at TP4

Four ranks cut the experts four ways: each card holds a quarter of them, and each expert's width is
cut in 128-column blocks over the ranks, so a card's share is whole blocks and every block has one
owner. The scheduler keys of the file above (the `draft-*` family, `busy-prefill-chunk`,
`checkpoint-slots`, the Engram placement) are per request and carry over unchanged; the file below
leaves them at their defaults and leaves placement to the planner, which at four ranks fits more of
the experts on the cards than the TP2 recipe can.

```yaml
server:
  host: 127.0.0.1
  port: 8034
  kernels: libr4d,libavx,spechub      # spechub carries the DFly2 coupling / traversal ops
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: /srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4.rad
    tp: 4
    num-speculative-tokens: 5         # DFly2 drafts at most 5 tokens
    prefill-span: 2                   # an 8192-token batch is two intervals, run as two steps
    max-num-batched-tokens: 8192
    checkpoint-interval: 4096         # one interval a step, and the step is what gets cut
```

Run it with the prefill pipelined:

```sh
RADIANCE_HY4_PIPE=1 radiance --config hy4-tp4.yaml
```

`RADIANCE_HY4_PIPE=1` runs a step as two row pieces and issues each piece's junction -- its `add`,
its exact all-reduce and the norm the next layer reads -- on the step's second lane, beside the
other piece's expert GEMMs. It needs a step of exactly one checkpoint interval, which is why the
batch is two intervals wide and the span runs them as two steps; pieces under 1024 rows are not
cut (`RADIANCE_HY4_PIPE_ROWS` moves that floor, `RADIANCE_HY4_PIPE_LANE` the lane).

Measured at these settings on four R9700 (32 GB each, 210 W a card), a 10948-token prompt, three
warm runs a config:

| | |
|---|---|
| prefill, `checkpoint-interval: 4096` and the 8192-token batch, no pipeline | 4978.1 tok/s |
| the same with `RADIANCE_HY4_PIPE=1` | **5577.5 tok/s** (+12.0 %, and +13.5 % over four ranks at the 2048-token intervals it ships with) |
| decode, 1 / 8 clients | 147.2 / 551.8 tok/s, the same greedy tokens with and without the cut |
| the cut's cost in memory | 0.05 GiB of activation arena and 0.32 GiB of the pools |
| prefill with the cut, bit for bit | not the step run whole (`448a9473ec` against `e7001a8285`) |

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+, Ninja and ROCm (`hipcc`).

```sh
git clone https://github.com/Code000-211/RadianceHy4Flash.git && cd RadianceHy4Flash
./build.sh                                   # configure, build for the cards present, run the fast tests
```

Set `RADIANCE_HOME=build/radiance_home` when running from the build tree, so the server loads the
kernel plugins built there.

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from.
