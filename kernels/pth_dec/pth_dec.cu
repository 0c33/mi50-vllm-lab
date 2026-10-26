// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sami (github.com/0c33)
// gfx906 split-K decode attention over vLLM's int8 per-token-head KV cache.
//
// Decode steps of Qwen3.8-27B (TP=2 rank: 12 q heads, 2 kv heads, head 256) carry up to
// QMAX query tokens per sequence (4 for MTP k=3 verification, 1 for the drafter). With
// GQA 6 that is up to 24 query rows per kv head, so the step is compute-bound (~47 FLOP
// per KV byte); Triton's unified attention reaches only ~1.1 TFLOPS there.
//
// One workgroup = (split, kv head, sequence); 256 threads = 4 waves; key tiles of 256.
//  QK: lane = key. Each lane converts its key's K row (int8 -> fp16, exact) once and
//      computes all R = QMAX*G scores with v_dot2_f32_f16 (fp32 accumulation). Q is read
//      with wave-uniform loads (scalar unit), so it needs neither LDS nor VGPRs.
//  softmax: online, base 2; tile max / sum reduced across the 4 waves through LDS.
//  PV: lane = output dim. P' = p * v_scale * 256 (fp16; the factor 256 keeps it out of
//      the fp16 subnormal range and is removed exactly at the end) is staged in LDS as
//      key pairs; V bytes are read straight from the cache (coalesced); dot2 over pairs.
// Partials (unnormalised O, running max m in log2 units, sum l) go to the buffers of
// vLLM's 3D Triton path: segm_output [T, Hq, NSEG, 256], segm_max/expsum [T, Hq, NSEG].
// pth_dec_combine merges the active splits into the fp16 output.
//
// HIP-graph safe: no host syncs, no allocation; split ranges and the number of active
// splits are derived on the device from seq_lens.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

#define PD_THREADS 256
#define PD_TILE 256
#define PD_D 256
#define PD_NEG_INF (-__builtin_inff())
#define PD_VOFF 260

// DPP butterfly within rows of 16, then row_bcast15/31; the total ends up in lane 63 and
// is returned wave-uniform via readlane.
template <int CTRL, int ROW_MASK>
static __device__ __forceinline__ float dpp_f(float old, float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old),
                                                                 __builtin_bit_cast(int, v), CTRL, ROW_MASK, 0xF, false));
}
static __device__ __forceinline__ float wave_max(float v) {
    const float ninf = -__builtin_inff();
    v = fmaxf(v, dpp_f<0xB1, 0xF>(ninf, v));   // quad_perm [1,0,3,2]
    v = fmaxf(v, dpp_f<0x4E, 0xF>(ninf, v));   // quad_perm [2,3,0,1]
    v = fmaxf(v, dpp_f<0x141, 0xF>(ninf, v));  // row_half_mirror
    v = fmaxf(v, dpp_f<0x140, 0xF>(ninf, v));  // row_mirror
    v = fmaxf(v, dpp_f<0x142, 0xA>(ninf, v));  // row_bcast15
    v = fmaxf(v, dpp_f<0x143, 0xC>(ninf, v));  // row_bcast31
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}
static __device__ __forceinline__ float wave_sum(float v) {
    v += dpp_f<0xB1, 0xF>(0.f, v);
    v += dpp_f<0x4E, 0xF>(0.f, v);
    v += dpp_f<0x141, 0xF>(0.f, v);
    v += dpp_f<0x140, 0xF>(0.f, v);
    v += dpp_f<0x142, 0xA>(0.f, v);
    v += dpp_f<0x143, 0xC>(0.f, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}

// 4 signed int8 (one dword) -> 2 half2, exact: fp16(0x6400 | (b ^ 0x80)) - 1152 == (int8)b
static __device__ __forceinline__ void cvt4(uint32_t w, h2& lo, h2& hi) {
    const uint32_t x = w ^ 0x80808080u;
    const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u);  // b0,64,b1,64
    const uint32_t b = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u);  // b2,64,b3,64
    const h2 off = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    lo = __builtin_bit_cast(h2, a) + off;
    hi = __builtin_bit_cast(h2, b) + off;
}

// low bytes of a and b (signed int8) -> half2(a, b), exact
static __device__ __forceinline__ h2 cvt2(uint32_t a, uint32_t b) {
    const uint32_t x = __builtin_amdgcn_perm(b, a, 0x0c040c00u) ^ 0x64806480u;  // a,0,b,0
    const h2 off = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    return __builtin_bit_cast(h2, x) + off;
}

static __device__ __forceinline__ int num_active_splits(int seq_len, int nsplit, int* chunk_out) {
    int chunk = (seq_len + nsplit - 1) / nsplit;
    chunk = (chunk + PD_TILE - 1) / PD_TILE * PD_TILE;
    *chunk_out = chunk;
    return (seq_len + chunk - 1) / chunk;
}

template <int QMAX, int G>
__global__ void __launch_bounds__(PD_THREADS, 2)  // 2 workgroups per CU
pth_dec_partial(const _Float16* __restrict__ q, const int q_st,
                const int8_t* __restrict__ kc, const int k_sb, const int k_ss, const int k_sh,
                const int* __restrict__ block_table, const int bt_stride,
                const int* __restrict__ seq_lens, const int* __restrict__ qsl,
                const int block_size, const float scale_log2,
                float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                const int nseg, const int nsplit) {
    // vLLM int8 per-token-head row (block, head, slot): [K 256 | K scale f32 | V 256 | V scale f32];
    // K row at kc + off, K scale at +256, V row at +PD_VOFF, V scale at +PD_VOFF+256 (host-checked).
    constexpr int R = QMAX * G;
    const int s = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
    const int Hq = gridDim.y * G;
    const int tid = threadIdx.x, w = tid >> 6, lane = tid & 63;

    const int q0 = qsl[b];
    const int q_len = qsl[b + 1] - q0;
    const int seq_len = seq_lens[b];
    if (q_len <= 0 || seq_len <= 0) return;
    const int ql = q_len < QMAX ? q_len : QMAX;  // the host guarantees q_len <= QMAX
    int chunk;
    num_active_splits(seq_len, nsplit, &chunk);
    const int k_begin = s * chunk;
    if (k_begin >= seq_len) return;  // inactive split; the combine derives the same count
    const int k_end = min(seq_len, k_begin + chunk);
    const int ctx0 = seq_len - q_len;  // absolute position of query token 0

    __shared__ __attribute__((aligned(16))) _Float16 sP[PD_TILE * R];  // [pair][row][2]
    __shared__ long sVoff[PD_TILE];                                    // V row offset per key
    __shared__ float sRedMax[4][R], sRedSum[4][R];                     // per-wave tile max / sum
    __shared__ float sM[R], sL[R];                                     // running max (log2) / sum

    if (tid < R) {
        sM[tid] = PD_NEG_INF;
        sL[tid] = 0.f;
    }

    // Q heads are contiguous (q_sh == 256, checked by the host): one base pointer per
    // query token, every (head, dim) offset is an immediate of the scalar loads.
    const h2* __restrict__ qt[QMAX];
#pragma unroll
    for (int i = 0; i < QMAX; ++i)
        qt[i] = reinterpret_cast<const h2*>(q) + (long)(q0 + (i < ql ? i : ql - 1)) * (q_st >> 1) + (long)(g * G) * (PD_D / 2);

    float acc_o[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc_o[r] = 0.f;
    const int d = w * 64 + lane;  // output dim owned in the PV phase
    const h2* sP2 = reinterpret_cast<const h2*>(sP);

    __syncthreads();
    for (int t0 = k_begin; t0 < k_end; t0 += PD_TILE) {
        // ---------------- QK: lane = key ----------------
        const int t = t0 + tid;
        const bool kv_ok = t < k_end;
        // Lanes past k_end read the last valid key (always in bounds, no branches around
        // the loads); their scores are masked to -inf below.
        const int tc = kv_ok ? t : k_end - 1;
        const int lb = tc / block_size;
        const int sl = tc - lb * block_size;
        const long blk = block_table[(long)b * bt_stride + lb];
        const long krow = blk * k_sb + (long)sl * k_ss + (long)g * k_sh;
        const uint2* krow8 = reinterpret_cast<const uint2*>(kc + krow);
        uint2 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) raw[j] = krow8[j];
        const float ks = *reinterpret_cast<const float*>(kc + krow + PD_D);
        const float vs = *reinterpret_cast<const float*>(kc + krow + PD_VOFF + PD_D);
        sVoff[tid] = krow + PD_VOFF;

        float acc_s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc_s[r] = 0.f;
        for (int c = 0; c < PD_D / 32; ++c) {  // 32 dims per step; the next step's K bytes are in flight
            uint2 nxt[4];
            const int cn = c + 1 < PD_D / 32 ? c + 1 : c;
#pragma unroll
            for (int j = 0; j < 4; ++j) nxt[j] = krow8[cn * 4 + j];
            h2 kh[16];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                cvt4(raw[j].x, kh[4 * j + 0], kh[4 * j + 1]);
                cvt4(raw[j].y, kh[4 * j + 2], kh[4 * j + 3]);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const h2* qp = qt[r / G] + (r % G) * (PD_D / 2) + c * 16;
#pragma unroll
                for (int k = 0; k < 16; ++k) acc_s[r] = __builtin_amdgcn_fdot2(qp[k], kh[k], acc_s[r], false);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) raw[j] = nxt[j];
        }

        // ---------------- online softmax ----------------
        const float kscale = ks * scale_log2;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = r / G;
            const bool ok = kv_ok && (i < ql) && (t <= ctx0 + i);
            acc_s[r] = ok ? acc_s[r] * kscale : PD_NEG_INF;
            const float mx = wave_max(acc_s[r]);
            if (lane == 0) sRedMax[w][r] = mx;
        }
        __syncthreads();  // (1) tile maxima visible
        float my_alpha = 1.f, my_mnew = PD_NEG_INF;
        const float vsc256 = vs * 256.f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float mt = fmaxf(fmaxf(sRedMax[0][r], sRedMax[1][r]), fmaxf(sRedMax[2][r], sRedMax[3][r]));
            const float m_old = sM[r];
            const float m_new = fmaxf(m_old, mt);
            const float alpha = (m_new == PD_NEG_INF) ? 1.f : exp2f(m_old - m_new);
            const float p = (acc_s[r] == PD_NEG_INF) ? 0.f : exp2f(acc_s[r] - m_new);
            acc_o[r] *= alpha;
            if (tid == r) {
                my_alpha = alpha;
                my_mnew = m_new;
            }
            const float ps = wave_sum(p);
            if (lane == 0) sRedSum[w][r] = ps;
            sP[((tid >> 1) * R + r) * 2 + (tid & 1)] = (_Float16)(p * vsc256);
        }
        __syncthreads();  // (2) P', sums visible; every thread is done reading sM
        if (tid < R) {
            sM[tid] = my_mnew;
            sL[tid] = sL[tid] * my_alpha + (sRedSum[0][tid] + sRedSum[1][tid] + sRedSum[2][tid] + sRedSum[3][tid]);
        }

        // ---------------- PV: lane = output dim ----------------
        const int nk = min(PD_TILE, k_end - t0);
#pragma unroll 4
        for (int k = 0; k < nk; k += 2) {
            const uint32_t va = (uint32_t)(uint8_t)kc[sVoff[k] + d];
            const uint32_t vb = (k + 1 < nk) ? (uint32_t)(uint8_t)kc[sVoff[k + 1] + d] : 0u;
            const h2 vh = cvt2(va, vb);
            const h2* pp = sP2 + (k >> 1) * R;
#pragma unroll
            for (int r = 0; r < R; ++r) acc_o[r] = __builtin_amdgcn_fdot2(pp[r], vh, acc_o[r], false);
        }
        __syncthreads();  // (3) sP / sVoff / sRed* are rewritten by the next tile
    }

    // ---------------- partials ----------------
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int i = r / G, h = r % G;
        if (i < ql) {
            const long row = ((long)(q0 + i) * Hq + (g * G + h)) * nseg + s;
            part_o[row * PD_D + d] = acc_o[r] * (1.f / 256.f);
            if (tid == 0) {
                part_m[row] = sM[r];
                part_l[row] = sL[r];
            }
        }
    }
}

// One workgroup of 64 threads per (token, q head); 4 dims per thread.
__global__ void __launch_bounds__(64)
pth_dec_combine(const float* __restrict__ part_o, const float* __restrict__ part_m,
                const float* __restrict__ part_l, const int* __restrict__ seq_lens,
                const int* __restrict__ qsl, const int num_seqs, _Float16* __restrict__ out,
                const long o_st, const long o_sh, const int nseg, const int nsplit) {
    const int tok = blockIdx.x, head = blockIdx.y, Hq = gridDim.y;
    int b = 0;
    while (b < num_seqs && qsl[b + 1] <= tok) ++b;
    if (b >= num_seqs || tok < qsl[b]) return;  // padding token
    const int seq_len = seq_lens[b];
    if (seq_len <= 0) return;
    int chunk;
    const int nact = num_active_splits(seq_len, nsplit, &chunk);
    const long base = ((long)tok * Hq + head) * nseg;
    float M = PD_NEG_INF;
    for (int p = 0; p < nact; ++p) M = fmaxf(M, part_m[base + p]);
    float L = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (M != PD_NEG_INF) {
        for (int p = 0; p < nact; ++p) {
            const float mp = part_m[base + p];
            if (mp == PD_NEG_INF) continue;
            const float wgt = exp2f(mp - M);
            L += wgt * part_l[base + p];
            const float4 o = reinterpret_cast<const float4*>(part_o + (base + p) * PD_D)[threadIdx.x];
            acc.x += wgt * o.x;
            acc.y += wgt * o.y;
            acc.z += wgt * o.z;
            acc.w += wgt * o.w;
        }
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    _Float16* o = out + (long)tok * o_st + (long)head * o_sh + 4 * threadIdx.x;
    o[0] = (_Float16)(acc.x * inv);
    o[1] = (_Float16)(acc.y * inv);
    o[2] = (_Float16)(acc.z * inv);
    o[3] = (_Float16)(acc.w * inv);
}

extern "C" hipError_t pth_dec_launch(
    const void* q, int q_st, const void* kc, int k_sb, int k_ss, int k_sh,
    const int* block_table, int bt_stride, const int* seq_lens, const int* qsl,
    int num_seqs, int num_tokens, int heads_q, int heads_kv, int block_size, float scale,
    int qmax, float* part_o, float* part_m, float* part_l, int nseg, int nsplit,
    void* out, long o_st, long o_sh, hipStream_t stream) {
    const int G = heads_q / heads_kv;
    if (G != 6 || heads_q % heads_kv != 0) return hipErrorInvalidValue;
    if (nsplit < 1 || nsplit > nseg || num_seqs < 1) return hipErrorInvalidValue;
    const float scale_log2 = scale * 1.4426950408889634f;
    dim3 grid(nsplit, heads_kv, num_seqs);
    const _Float16* qh = static_cast<const _Float16*>(q);
    const int8_t* k8 = static_cast<const int8_t*>(kc);
#define PD_ARGS qh, q_st, k8, k_sb, k_ss, k_sh, block_table, bt_stride, seq_lens, qsl, block_size, \
                scale_log2, part_o, part_m, part_l, nseg, nsplit
    if (qmax == 1) {
        hipLaunchKernelGGL((pth_dec_partial<1, 6>), grid, dim3(PD_THREADS), 0, stream, PD_ARGS);
    } else if (qmax <= 4) {
        hipLaunchKernelGGL((pth_dec_partial<4, 6>), grid, dim3(PD_THREADS), 0, stream, PD_ARGS);
    } else {
        return hipErrorInvalidValue;
    }
#undef PD_ARGS
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return e;
    hipLaunchKernelGGL(pth_dec_combine, dim3(num_tokens, heads_q), dim3(64), 0, stream,
                       part_o, part_m, part_l, seq_lens, qsl, num_seqs,
                       static_cast<_Float16*>(out), o_st, o_sh, nseg, nsplit);
    return hipGetLastError();
}
