// r4d_mla_kv_bf16.hip -- Hy4-flash's kv_b read apart from the sink attention kernels:
// mla_project_kv (absorbed MLA's per-head projections with unequal K and V widths) and
// mla_kv_split (the kv_b output laid out as k and v heads for the non-absorbed prefill).
//
// mla_project_kv is r4d_mla.hip's project_absorb_rows / project_expand_rows with the head's block
// generalised (kernels/mla_sink_attn.md section 5): head h's rows of the checkpoint's kv_b start at
// h * (k_dim + v_dim), its k_dim key rows first and then its v_dim value rows. Compiled for
// (k_dim, v_dim, latent) = (192, 256, 512), up to 64 heads.
//   absorb: y[m,h,j] = sum_{c<192} a[m,h,c] * b[h*448 + c][j]            [M,H,192] -> [M,H,512]
//   expand: y[m,h,c] = sum_{j<512} a[m,h,j] * b[h*448 + 192 + c][j]      [M,H,512] -> [M,H,256]
// The sums run in one fixed order (absorb: c ascending; expand: each lane's share of j ascending,
// then r4d_xor_sum<32>) whatever the group size R is, so a row's bytes do not depend on the batch
// it is in. They use fused multiply-adds where libref does not, so a result is within one bf16 ulp
// of libref's, not byte-equal.
//
// mla_kv_split is a copy: k[i,h,0:kd] = kvb[i, h*(kd+vd) : +kd], k[i,h,kd:kd+r] = k_rope[i],
// v[i,h,:] = kvb[i, h*(kd+vd)+kd : +vd]. One thread a 16 bytes of output.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <cstdint>

namespace {

constexpr int kKd = 192, kVd = 256, kLatent = 512;

struct KvProjectMeta {
    long xr, xh, wp, yr, yh;
    int rows, heads;
};

template <int R>
__global__ __launch_bounds__(128) void kv_absorb_rows(const uint16_t* x, const uint16_t* w,
                                                      uint16_t* y, KvProjectMeta m) {
    const int col = blockIdx.x * 128 + threadIdx.x, head = blockIdx.y, r0 = blockIdx.z * R;
    if (col >= kLatent) return;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = x + (long)min(r0 + r, m.rows - 1) * m.xr + head * m.xh;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = 0; j < kKd; ++j) {
        const float wv = bf16_to_f32(w[((long)head * (kKd + kVd) + j) * m.wp + col]);
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32(xr[r][j]), wv, sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (r0 + r < m.rows) y[(long)(r0 + r) * m.yr + head * m.yh + col] = f32_to_bf16(sum[r]);
}

template <int R>
__global__ __launch_bounds__(128) void kv_expand_rows(const uint16_t* x, const uint16_t* w,
                                                      uint16_t* y, KvProjectMeta m) {
    const int lane = threadIdx.x & 31, col = blockIdx.x * 4 + (threadIdx.x >> 5), head = blockIdx.y,
              r0 = blockIdx.z * R;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = x + (long)min(r0 + r, m.rows - 1) * m.xr + head * m.xh;
    const uint16_t* const wr = w + ((long)head * (kKd + kVd) + kKd + col) * m.wp;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = lane; j < kLatent; j += 32) {
        const float wv = bf16_to_f32(wr[j]);
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32(xr[r][j]), wv, sum[r]);
    }
    float mine = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = r4d_xor_sum<32>(sum[r]);
        if (lane == r) mine = sum[r];
    }
    if (lane < R && r0 + lane < m.rows)
        y[(long)(r0 + lane) * m.yr + head * m.yh + col] = f32_to_bf16(mine);
}

// One thread a chunk of C elements (8 for 16-byte accesses, 1 otherwise) of one (row, head): the
// k row's chunks, kvb's then k_rope's, then the v row's.
template <int C>
__global__ __launch_bounds__(256) void kv_split_kernel(
    const uint16_t* __restrict__ kvb, const uint16_t* __restrict__ kr, uint16_t* __restrict__ k,
    uint16_t* __restrict__ v, uint32_t N, uint32_t heads, uint32_t kd, uint32_t vd, uint32_t rd,
    uint64_t bp, uint64_t rp, uint64_t kp, uint64_t vp) {
    const uint32_t kc = (kd + rd) / C, vc = vd / C, per = kc + vc;
    const uint64_t g = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (g >= (uint64_t)N * heads * per) return;
    const uint64_t rh = g / per;
    const uint32_t c = (uint32_t)(g % per), h = (uint32_t)(rh % heads);
    const uint64_t i = rh / heads;
    const uint16_t* src = kvb + i * bp + (uint64_t)h * (kd + vd);
    uint16_t* dst;
    const uint16_t* from;
    if (c < kc) {
        const uint32_t e = c * C;
        dst = k + i * kp + (uint64_t)h * (kd + rd) + e;
        from = e < kd ? src + e : kr + i * rp + (e - kd);
    } else {
        const uint32_t e = (c - kc) * C;
        dst = v + i * vp + (uint64_t)h * vd + e;
        from = src + kd + e;
    }
    if (C == 8) *(uint4*)dst = *(const uint4*)from;
    else        *dst = *from;
}

}  // namespace

extern "C" int r4d_rad_mla_project_kv(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* w = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const char* mode = r4d_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    const long h = r4d_geti_or(a, "heads", 0);
    if ((!absorb && !expand) || h < 1 || h > 64 || r4d_geti_or(a, "k_dim", 0) != kKd ||
        r4d_geti_or(a, "v_dim", 0) != kVd || r4d_geti_or(a, "latent_dim", 0) != kLatent)
        return R4D_NO(RAD_E_SHAPE);
    const long xw = absorb ? kKd : kLatent, yw = absorb ? kLatent : kVd;
    /* [M, heads * width] or [M, heads, width]: a token's heads are one row, a head a slice. */
    long long M = 0, My = 0;
    const long long xr = r4d_rows_pitch(x, h * xw, &M), yr = r4d_rows_pitch(y, h * yw, &My);
    if (xr < 0 || yr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || My != M) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != h * (kKd + kVd) || w->shape[1] != kLatent ||
        w->stride[1] != 1 || w->stride[0] < kLatent)
        return R4D_NO(RAD_E_SHAPE);
    const KvProjectMeta m{ (long)xr, xw, (long)w->stride[0], (long)yr, yw, (int)M, (int)h };
    const auto* xp = (const uint16_t*)x->data;
    const auto* wp = (const uint16_t*)w->data;
    auto* yp = (uint16_t*)y->data;
    const int R = M >= 8 ? 8 : M >= 4 ? 4 : M >= 2 ? 2 : 1;
#define R4D_KVP(RR)                                                                              \
    do {                                                                                         \
        const unsigned groups = (unsigned)((M + RR - 1) / RR);                                   \
        if (absorb)                                                                              \
            return R4D_EMIT(s, (kv_absorb_rows<RR>), dim3(kLatent / 128, (unsigned)h, groups),   \
                            dim3(128), 0, xp, wp, yp, m);                                        \
        return R4D_EMIT(s, (kv_expand_rows<RR>), dim3(kVd / 4, (unsigned)h, groups), dim3(128), \
                        0, xp, wp, yp, m);                                                       \
    } while (0)
    if (R == 8) R4D_KVP(8);
    if (R == 4) R4D_KVP(4);
    if (R == 2) R4D_KVP(2);
    R4D_KVP(1);
#undef R4D_KVP
}

extern "C" int r4d_rad_mla_kv_split(const RadArgs* a, RadStream s) {
    const RadTensor *kvb = r4d_opt(a, 0), *kr = r4d_opt(a, 1), *k = r4d_opt(a, 2),
                    *v = r4d_opt(a, 3);
    if (!kvb || !kr || !k || !v) return R4D_NO(RAD_E_INVAL);
    if (kvb->dtype != RAD_BF16 || kr->dtype != RAD_BF16 || k->dtype != RAD_BF16 ||
        v->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long long H = r4d_geti_or(a, "heads", 0), kd = r4d_geti_or(a, "k_dim", 0),
                    vd = r4d_geti_or(a, "v_dim", 0), rd = r4d_geti_or(a, "rope_dim", 0);
    if (H < 1 || kd < 1 || vd < 1 || rd < 1) return R4D_NO(RAD_E_SHAPE);
    long long N = 0, Nr = 0, Nk = 0, Nv = 0;
    const long long bp = r4d_rows_pitch(kvb, H * (kd + vd), &N), rp = r4d_rows_pitch(kr, rd, &Nr),
                    kp = r4d_rows_pitch(k, H * (kd + rd), &Nk),
                    vp = r4d_rows_pitch(v, H * vd, &Nv);
    if (bp < 0 || rp < 0 || kp < 0 || vp < 0) return R4D_NO(RAD_E_STRIDE);
    if (N < 1 || Nr != N || Nk != N || Nv != N) return R4D_NO(RAD_E_SHAPE);
    const bool vec = kd % 8 == 0 && vd % 8 == 0 && rd % 8 == 0 && bp % 8 == 0 && rp % 8 == 0 &&
                     kp % 8 == 0 && vp % 8 == 0 && r4d_aligned(kvb->data, 16) &&
                     r4d_aligned(kr->data, 16) && r4d_aligned(k->data, 16) &&
                     r4d_aligned(v->data, 16);
    const long long per = (kd + rd + vd) / (vec ? 8 : 1);
    const unsigned long long total = (unsigned long long)N * H * per;
    const unsigned long long blocks = (total + 255) / 256;
    if (blocks > 0x7fffffffull) return R4D_NO(RAD_E_SHAPE);
#define R4D_KVS(C)                                                                               \
    R4D_EMIT(s, (kv_split_kernel<C>), dim3((uint32_t)blocks), dim3(256), 0,                      \
             (const uint16_t*)kvb->data, (const uint16_t*)kr->data, (uint16_t*)k->data,         \
             (uint16_t*)v->data, (uint32_t)N, (uint32_t)H, (uint32_t)kd, (uint32_t)vd,           \
             (uint32_t)rd, (uint64_t)bp, (uint64_t)rp, (uint64_t)kp, (uint64_t)vp)
    return vec ? R4D_KVS(8) : R4D_KVS(1);
#undef R4D_KVS
}
