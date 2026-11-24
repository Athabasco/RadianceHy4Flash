/* r4d_mla_sink_fold.h -- the sink's fold, shared by the op that runs it (r4d_mla_sink.hip:
 * mla_sink_rows, merge_sink, the fused form) and by the op that folds it into its own work
 * (r4d_mla_kv_bf16.hip: mla_attn_paged_sink_expand_i8g, which merges a head's partials into its
 * LDS instead of launching merge_sink). One copy of the rule, so the places that apply it cannot
 * drift: a (row, head)'s partials fold in ascending slot order, the first visible slot copied and
 * each later one by acc = acc * exp(acc_m - m') + part * exp(part_m - m'), and the sink joins once
 * after the fold, never per slot. Slot s of (row, head) is at
 * parts + (row * heads + head) * splits * kMlaPart + s * kMlaPart.
 *
 * Host parses of this header (the row tables, a lint) reach only the constants and the struct:
 * the two rules below are the device's. */
#pragma once

#include <cmath>
#include <cstdint>

#include "r4d_args.h"

#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>        /* __device__, __forceinline__ */
#define R4D_MLA_FOLD_HD __device__ __forceinline__
#else
/* A host parse (the emulator, a lint): the same rules as plain inline functions. */
#define R4D_MLA_FOLD_HD inline
#endif

/* A partial: the latent's values, the slot's max, its denominator, padded to 16 bytes (the
 * merge's vector loads). `kMlaPart` is the slot's pitch in floats. */
constexpr int kMlaD = 512, kMlaPart = kMlaD + 4;
/* The slots a launch holds at most: sink_law's ceiling. */
constexpr int kMlaSplitsMax = 20;

/* The law's merge, shared by merge_sink, the fused form and the fold in the expand so all three
 * give the same bytes: slots with a visible key folded in ascending order into (acc_m, acc_d, acc),
 * the first one copied, each later one by acc = acc * exp(acc_m - m') + part * exp(part_m - m'),
 * m' = max(acc_m, part_m); explicit fmaf so no contraction choice of the compiler differs between
 * the kernels. */
struct SinkFold {
    float a, b, m;
};
R4D_MLA_FOLD_HD SinkFold sink_fold(float acc_m, float part_m) {
    const float mn = fmaxf(acc_m, part_m);
    return { expf(acc_m - mn), expf(part_m - mn), mn };
}
/* The sink once at the end: shift to max(m, sink) as libref does, so a sink far above every score
 * cannot overflow. No visible key gives zero, sink or not. */
R4D_MLA_FOLD_HD float sink_inv(float m, float d, float sink) {
    if (!(d > 0)) return 0;
    const float top = fmaxf(m, sink), c = expf(m - top);
    return c / fmaf(d, c, expf(sink - top));
}

/* What mla_attn_paged_sink's launch resolved to. The caller of mla_sink_launch that fuses the
 * launch's next step needs it to know whether the partials are in the op's scratch (splits > 1),
 * whether the launch wrote the latent itself (fused: an unsplit launch that still ran the law),
 * and over how many rows. */
struct MlaSinkForm {
    long ss;    /* the sinks' per-head stride, f32 [>= heads] */
    int rows;   /* the launch's rows */
    int splits; /* slots a (row, head) was split over; > 1: the partials are in the scratch */
    int fused;  /* 1: an unsplit launch that folded the slots itself and wrote the latent */
};

/* Emits mla_attn_paged_sink's kernels for `a` (that op's own operands 0..7 and parameters) and
 * reports the form. With `form` the caller takes the launch over: the sink is emitted WITHOUT
 * merge_sink, and the caller writes the latent itself, from the partials where splits > 1. With
 * `form` null the whole op is emitted, as r4d_rad_mla_attn_paged_sink does. */
int mla_sink_launch(const RadArgs* a, RadStream stream, MlaSinkForm* form);
