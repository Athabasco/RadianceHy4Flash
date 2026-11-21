// r4d_ihc_bf16.hip -- Hy4-flash's independent hyper-connections and the sigmoid-gated product.
//
//   ihc_pre   f = f32(h); r = 1/sqrt(sum f^2 / (hc n) + eps); c[j] = (fn[j] . f) r
//             pre[j]  = sigmoid(c[j] scale[0] + base[j]) + hc_eps               j < hc
//             x       = bf16(sum_j pre[j] f[j])
//             post[j] = post_mag sigmoid(c[hc+j] scale[1] + base[hc+j]) + hc_eps  (when `post` is
//                       present: rows == 2 hc and scale has two entries)
//   ihc_post  h[j][i] = bf16(post[j] f32(y[i]) + f32(h[j][i])), in place on the hc streams
//   gate_mul  y = bf16(f32(a) sigmoid(f32(g))), one rounding
//
// libref/ref_hy4.cpp is the oracle, kernels/ihc.md the design. Written for this repository; the
// semantics are HYV4FlashHyperConnection / HyperHead as the design note states them.
//
// ihc_pre: one workgroup of 8 waves a token row. Each thread strides the row's hc*n columns (t,
// t+256, ...) keeping rows+1 f32 partial sums (the projections and the sum of squares); lanes then
// reduce through r4d_xor_sum, the 8 waves through LDS in wave order. One thread forms r, the gates
// and `post`; every thread then forms x for its columns. libref sums in ascending order (the sum
// of squares in double), so the kernel agrees with it to rounding and not in bytes: relative
// 1e-5 on c, and x within one bf16 ulp except where a sum straddles a rounding boundary. The
// reduction order does not depend on M, so a row's bytes do not depend on the batch.
// LDS: (8 * (rows + 1) + 2 hc) floats, under 1 KiB. No scratch.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace {

enum { kIhcThreads = 256, kIhcWaves = kIhcThreads / 32 };

__device__ __forceinline__ float ihc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

template <int HC, bool POST>
__global__ __launch_bounds__(kIhcThreads) void ihc_pre_kernel(
    const uint16_t* __restrict__ h, const float* __restrict__ fn, const float* __restrict__ scale,
    const float* __restrict__ base, uint16_t* __restrict__ x, float* __restrict__ post,
    uint32_t n, float eps, float heps, float pmag, uint64_t hp, uint64_t fp, uint64_t xp,
    uint64_t pp) {
    constexpr int R = POST ? 2 * HC : HC;
    __shared__ float part[kIhcWaves][R + 1];
    __shared__ float gate[HC];
    const uint32_t t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const uint64_t row = blockIdx.x;
    const uint32_t width = (uint32_t)HC * n;
    const uint16_t* __restrict__ hr = h + row * hp;

    float acc[R + 1];
#pragma unroll
    for (int k = 0; k <= R; ++k) acc[k] = 0.0f;
    for (uint32_t c = t; c < width; c += kIhcThreads) {
        const float f = bf16_to_f32(hr[c]);
        acc[R] += f * f;
#pragma unroll
        for (int j = 0; j < R; ++j) acc[j] += fn[(uint64_t)j * fp + c] * f;
    }
#pragma unroll
    for (int k = 0; k <= R; ++k) {
        const float s = r4d_xor_sum<32>(acc[k]);
        if (lane == 0) part[wave][k] = s;
    }
    __syncthreads();
    if (t == 0) {
        float s[R + 1];
#pragma unroll
        for (int k = 0; k <= R; ++k) {
            float v = 0.0f;
            for (int w = 0; w < kIhcWaves; ++w) v += part[w][k];
            s[k] = v;
        }
        const float inv = 1.0f / sqrtf(s[R] / (float)width + eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) gate[j] = ihc_sigmoid(s[j] * inv * scale[0] + base[j]) + heps;
        if (POST) {
#pragma unroll
            for (int j = 0; j < HC; ++j) {
                const int k = POST ? HC + j : 0;
                post[row * pp + j] = pmag * ihc_sigmoid(s[k] * inv * scale[POST ? 1 : 0] + base[k]) + heps;
            }
        }
    }
    __syncthreads();
    uint16_t* __restrict__ xr = x + row * xp;
    for (uint32_t i = t; i < n; i += kIhcThreads) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) v += gate[j] * bf16_to_f32(hr[(uint32_t)j * n + i]);
        xr[i] = f32_to_bf16(v);
    }
}

// h[j][i] = bf16(post[j] * y[i] + h[j][i]). The product is its own f32 rounding, as libref's.
// Eight columns of every stream a thread when rows and pointers allow 16-byte accesses.
template <bool VEC>
__global__ __launch_bounds__(256) void ihc_post_kernel(
    const uint16_t* __restrict__ y, const float* __restrict__ post, uint16_t* __restrict__ h,
    uint32_t n, uint32_t hc, uint64_t yp, uint64_t pp, uint64_t hp) {
    const uint64_t row = blockIdx.y;
    const uint16_t* __restrict__ yr = y + row * yp;
    const float* __restrict__ pr = post + row * pp;
    uint16_t* __restrict__ hr = h + row * hp;
    if (VEC) {
        const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
        if (i >= n) return;
        const uint4 yv = *(const uint4*)(yr + i);
        float yf[8];
        const uint32_t yw[4] = { yv.x, yv.y, yv.z, yv.w };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            yf[2 * k]     = __uint_as_float(yw[k] << 16);
            yf[2 * k + 1] = __uint_as_float(yw[k] & 0xffff0000u);
        }
        for (uint32_t j = 0; j < hc; ++j) {
            const float p = pr[j];
            uint4 hv = *(const uint4*)(hr + (uint64_t)j * n + i);
            uint32_t hw[4] = { hv.x, hv.y, hv.z, hv.w };
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float lo = __uint_as_float(hw[k] << 16), hi = __uint_as_float(hw[k] & 0xffff0000u);
                const float a = __fmul_rn(p, yf[2 * k]), b = __fmul_rn(p, yf[2 * k + 1]);
                hw[k] = (uint32_t)f32_to_bf16(__fadd_rn(a, lo)) |
                        ((uint32_t)f32_to_bf16(__fadd_rn(b, hi)) << 16);
            }
            *(uint4*)(hr + (uint64_t)j * n + i) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
        }
    } else {
        const uint32_t i = blockIdx.x * 256u + threadIdx.x;
        if (i >= n) return;
        const float yf = bf16_to_f32(yr[i]);
        for (uint32_t j = 0; j < hc; ++j) {
            uint16_t* hp_ = hr + (uint64_t)j * n + i;
            *hp_ = f32_to_bf16(__fadd_rn(__fmul_rn(pr[j], yf), bf16_to_f32(*hp_)));
        }
    }
}

// y = bf16(a * sigmoid(g)).
template <bool VEC>
__global__ __launch_bounds__(256) void gate_mul_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ g, uint16_t* __restrict__ y,
    uint32_t n, uint64_t ap, uint64_t gp, uint64_t yp) {
    const uint64_t row = blockIdx.y;
    if (VEC) {
        const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
        if (i >= n) return;
        const uint4 av = *(const uint4*)(a + row * ap + i), gv = *(const uint4*)(g + row * gp + i);
        const uint32_t aw[4] = { av.x, av.y, av.z, av.w }, gw[4] = { gv.x, gv.y, gv.z, gv.w };
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a0 = __uint_as_float(aw[k] << 16), a1 = __uint_as_float(aw[k] & 0xffff0000u);
            const float g0 = __uint_as_float(gw[k] << 16), g1 = __uint_as_float(gw[k] & 0xffff0000u);
            o[k] = (uint32_t)f32_to_bf16(a0 * ihc_sigmoid(g0)) |
                   ((uint32_t)f32_to_bf16(a1 * ihc_sigmoid(g1)) << 16);
        }
        *(uint4*)(y + row * yp + i) = make_uint4(o[0], o[1], o[2], o[3]);
    } else {
        const uint32_t i = blockIdx.x * 256u + threadIdx.x;
        if (i >= n) return;
        y[row * yp + i] = f32_to_bf16(bf16_to_f32(a[row * ap + i]) *
                                      ihc_sigmoid(bf16_to_f32(g[row * gp + i])));
    }
}

bool ihc_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
bool ihc_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
bool ihc_vec(const void* p, uint64_t pitch) { return ((uintptr_t)p & 15) == 0 && pitch % 8 == 0; }

}  // namespace

enum { IHC_H = 0, IHC_FN, IHC_SCALE, IHC_BASE, IHC_X, IHC_POST };

extern "C" int r4d_rad_ihc_pre(const RadArgs* a, RadStream s) {
    const RadTensor* h = r4d_opt(a, IHC_H);
    const RadTensor* fn = r4d_opt(a, IHC_FN);
    const RadTensor* sc = r4d_opt(a, IHC_SCALE);
    const RadTensor* bs = r4d_opt(a, IHC_BASE);
    const RadTensor* x = r4d_opt(a, IHC_X);
    const RadTensor* post = r4d_opt(a, IHC_POST);
    if (!h || !fn || !sc || !bs || !x) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(h) || !ihc_bf16(x) || !ihc_f32(fn) || !ihc_f32(sc) || !ihc_f32(bs) ||
        (post && !ihc_f32(post)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0),
                    rows = r4d_geti_or(a, "rows", 0);
    const float eps = (float)r4d_getf(a, "eps", 0), heps = (float)r4d_getf(a, "hc_eps", 0);
    if (n < 1 || n > 16384 || hc < 2 || hc > 8 || !std::isfinite(eps) || eps <= 0 ||
        !std::isfinite(heps) || heps < 0)
        return R4D_NO(RAD_E_SHAPE);
    const bool has_post = post != nullptr;
    if (rows != (has_post ? 2 * hc : hc)) return R4D_NO(RAD_E_SHAPE);
    float pmag = 0.0f;
    if (has_post) {
        if (r4d_getf(a, "post_mag", -1e30) == -1e30) return R4D_NO(RAD_E_INVAL);
        pmag = (float)r4d_getf(a, "post_mag", 0);
        if (!std::isfinite(pmag)) return R4D_NO(RAD_E_SHAPE);
    }
    const long long W = hc * n;
    long long M = 0, Mx = 0, Mp = 0;
    const long long hp = r4d_rows_pitch(h, W, &M), xp = r4d_rows_pitch(x, n, &Mx);
    if (hp < 0 || xp < 0) return R4D_NO(RAD_E_STRIDE);
    long long pp = 0;
    if (has_post) {
        pp = r4d_rows_pitch(post, hc, &Mp);
        if (pp < 0) return R4D_NO(RAD_E_STRIDE);
    }
    if (M < 1 || M > 65535 || Mx != M || (has_post && Mp != M)) return R4D_NO(RAD_E_SHAPE);
    if (fn->rank != 2 || fn->shape[0] != rows || fn->shape[1] != W || fn->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sc) != (has_post ? 2 : 1) || r4d_numel(bs) != rows) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(sc) || !r4d_contig(bs)) return R4D_NO(RAD_E_STRIDE);

#define R4D_IHC(HC_, POST_)                                                                       \
    R4D_EMIT(s, (ihc_pre_kernel<HC_, POST_>), dim3((uint32_t)M), dim3(kIhcThreads), 0,            \
             (const uint16_t*)h->data, (const float*)fn->data, (const float*)sc->data,            \
             (const float*)bs->data, (uint16_t*)x->data, (float*)(has_post ? post->data : nullptr), \
             (uint32_t)n, eps, heps, pmag, (uint64_t)hp, (uint64_t)fn->stride[0], (uint64_t)xp,   \
             (uint64_t)pp)
#define R4D_IHC_HC(HC_) return has_post ? R4D_IHC(HC_, true) : R4D_IHC(HC_, false)
    switch (hc) {
        case 2: R4D_IHC_HC(2);
        case 3: R4D_IHC_HC(3);
        case 4: R4D_IHC_HC(4);
        case 5: R4D_IHC_HC(5);
        case 6: R4D_IHC_HC(6);
        case 7: R4D_IHC_HC(7);
        default: R4D_IHC_HC(8);
    }
#undef R4D_IHC_HC
#undef R4D_IHC
}

extern "C" int r4d_rad_ihc_post(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    const RadTensor* post = r4d_opt(a, 1);
    const RadTensor* h = r4d_opt(a, 2);
    if (!y || !post || !h) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(y) || !ihc_bf16(h) || !ihc_f32(post)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    if (n < 1 || n > 16384 || hc < 2 || hc > 8) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0, Mp = 0;
    const long long hp = r4d_rows_pitch(h, hc * n, &M), yp = r4d_rows_pitch(y, n, &My),
                    pp = r4d_rows_pitch(post, hc, &Mp);
    if (hp < 0 || yp < 0 || pp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || My != M || Mp != M) return R4D_NO(RAD_E_SHAPE);
    const bool vec = n % 8 == 0 && ihc_vec(h->data, hp) && ihc_vec(y->data, yp);
    const dim3 grid(vec ? (uint32_t)((n / 8 + 255) / 256) : (uint32_t)((n + 255) / 256), (uint32_t)M);
    return vec ? R4D_EMIT(s, ihc_post_kernel<true>, grid, dim3(256), 0, (const uint16_t*)y->data,
                          (const float*)post->data, (uint16_t*)h->data, (uint32_t)n, (uint32_t)hc,
                          (uint64_t)yp, (uint64_t)pp, (uint64_t)hp)
               : R4D_EMIT(s, ihc_post_kernel<false>, grid, dim3(256), 0, (const uint16_t*)y->data,
                          (const float*)post->data, (uint16_t*)h->data, (uint32_t)n, (uint32_t)hc,
                          (uint64_t)yp, (uint64_t)pp, (uint64_t)hp);
}

extern "C" int r4d_rad_gate_mul(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* g = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !g || !y) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(x) || !ihc_bf16(g) || !ihc_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n < 1) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0, My = 0;
    const long long ap = r4d_rows_pitch(x, n, &M), gp = r4d_rows_pitch(g, n, &Mg),
                    yp = r4d_rows_pitch(y, n, &My);
    if (ap < 0 || gp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || Mg != M || My != M) return R4D_NO(RAD_E_SHAPE);
    const bool vec = n % 8 == 0 && ihc_vec(x->data, ap) && ihc_vec(g->data, gp) && ihc_vec(y->data, yp);
    const dim3 grid(vec ? (uint32_t)((n / 8 + 255) / 256) : (uint32_t)((n + 255) / 256), (uint32_t)M);
    return vec ? R4D_EMIT(s, gate_mul_kernel<true>, grid, dim3(256), 0, (const uint16_t*)x->data,
                          (const uint16_t*)g->data, (uint16_t*)y->data, (uint32_t)n, (uint64_t)ap,
                          (uint64_t)gp, (uint64_t)yp)
               : R4D_EMIT(s, gate_mul_kernel<false>, grid, dim3(256), 0, (const uint16_t*)x->data,
                          (const uint16_t*)g->data, (uint16_t*)y->data, (uint32_t)n, (uint64_t)ap,
                          (uint64_t)gp, (uint64_t)yp);
}
