// W4A16 GEMM for gfx906 at M <= 8 rows (GPTQ/Exllama-shuffled weights, v2 zeros).
//
// Replaces ops.gptq_gemm for the spec-decode verify shapes. Differences from the
// Exllama kernel it mirrors (same packed layout, same dequant bit tricks):
//  - each weight is dequantized once, scaled once (fp16), then used for all M
//    rows via v_dot2_f32_f16 with fp32 accumulation;
//  - no fp16 CAS atomics: the 4 waves of a block split the block's K range and
//    fold through LDS; K-blocks write fp32 partials that a small kernel sums.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

union H2U {
  uint32_t u;
  h2_t h;
};

// (q - z) * s for the 8 shuffled nibbles of one word -> 4 half2 in k order.
__device__ __forceinline__ void dequant8(uint32_t q, h2_t z1, h2_t z16, h2_t s2, h2_t (&dq)[4]) {
  const uint32_t c0 = 0x64006400;
  const h2_t inv16 = {(_Float16)0.0625f, (_Float16)0.0625f};
  H2U q0, q1, q2, q3;
  q0.u = (q & 0x000f000f) | c0;          // (q0 + 1024, q1 + 1024)
  q1.u = (q & 0x00f000f0) | c0;          // (q2 * 16 + 1024, q3 * 16 + 1024)
  q >>= 8;
  q2.u = (q & 0x000f000f) | c0;
  q3.u = (q & 0x00f000f0) | c0;
  dq[0] = (q0.h + z1) * s2;
  dq[1] = (q1.h * inv16 + z16) * s2;
  dq[2] = (q2.h + z1) * s2;
  dq[3] = (q3.h * inv16 + z16) * s2;
}

template <int M, int WAVES>
__global__ void __launch_bounds__(64 * WAVES)
w4_skinny_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ QW,
                 const uint32_t* __restrict__ QZ, const __half* __restrict__ SC,
                 __half* __restrict__ Y, float* __restrict__ P, int N, int K, int KW) {
  // KW: k values per wave (multiple of 32); block K range = WAVES * KW.
  extern __shared__ __align__(16) unsigned char smem[];
  const int KB = WAVES * KW;
  __half* as = reinterpret_cast<__half*>(smem);   // [M][KB] while the K loop runs
  float* red = reinterpret_cast<float*>(smem);    // [WAVES-1][M][256] afterwards

  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int n = blockIdx.x * 256 + lane * 4;
  const int kb0 = blockIdx.y * KB;

  for (int i = threadIdx.x * 8; i < M * KB; i += 64 * WAVES * 8) {
    const int m = i / KB, kk = i - m * KB;
    *reinterpret_cast<u32x4*>(&as[m * KB + kk]) =
        *reinterpret_cast<const u32x4*>(&A[(size_t)m * K + kb0 + kk]);
  }
  __syncthreads();

  float acc[M][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[m][c] = 0.f;

  const bool active = n < N;
  const int k_begin = kb0 + wave * KW;
  const int groups_n8 = N / 8;
  if (active) {
    for (int k = k_begin; k < k_begin + KW; k += 32) {
      const int g = k >> 5;
      // zeros for columns n..n+3 (v2 format: no +1 offset)
      const uint32_t zw = QZ[(size_t)g * groups_n8 + (n >> 3)] >> ((n & 7) * 4);
      const u32x2 sraw = *reinterpret_cast<const u32x2*>(&SC[(size_t)g * N + n]);
      H2U s01, s23;
      s01.u = sraw.x;
      s23.u = sraw.y;
      h2_t z1[4], z16[4], s2[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int z = (zw >> (4 * c)) & 0xf;
        const _Float16 zf1 = (_Float16)(-1024 - z);
        const _Float16 zf16 = (_Float16)(-64 - z);
        z1[c] = {zf1, zf1};
        z16[c] = {zf16, zf16};
        const _Float16 s = c < 2 ? s01.h[c] : s23.h[c - 2];
        s2[c] = {s, s};
      }
      u32x4 qv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        qv[j] = __builtin_nontemporal_load(
            reinterpret_cast<const u32x4*>(&QW[(size_t)((k >> 3) + j) * N + n]));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        h2_t dq[4][4];
        dequant8(qv[j].x, z1[0], z16[0], s2[0], dq[0]);
        dequant8(qv[j].y, z1[1], z16[1], s2[1], dq[1]);
        dequant8(qv[j].z, z1[2], z16[2], s2[2], dq[2]);
        dequant8(qv[j].w, z1[3], z16[3], s2[3], dq[3]);
        const int ka = k - kb0 + j * 8;
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const u32x4 av = *reinterpret_cast<const u32x4*>(&as[m * KB + ka]);
          const h2_t* a2 = reinterpret_cast<const h2_t*>(&av);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float r = acc[m][c];
#pragma unroll
            for (int i = 0; i < 4; ++i) r = __builtin_amdgcn_fdot2(dq[c][i], a2[i], r, false);
            acc[m][c] = r;
          }
        }
      }
    }
  }

  // Fold the WAVES partials of this block's K range through LDS (reusing the
  // activation buffer, hence the barrier first).
  __syncthreads();
  if (wave > 0) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < 4; ++c) red[((wave - 1) * M + m) * 256 + lane * 4 + c] = acc[m][c];
  }
  __syncthreads();
  if (wave == 0 && active) {
#pragma unroll
    for (int w = 1; w < WAVES; ++w)
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[m][c] += red[((w - 1) * M + m) * 256 + lane * 4 + c];
    if (gridDim.y == 1) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        __half2* out = reinterpret_cast<__half2*>(&Y[(size_t)m * N + n]);
        out[0] = __floats2half2_rn(acc[m][0], acc[m][1]);
        out[1] = __floats2half2_rn(acc[m][2], acc[m][3]);
      }
    } else {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float4 v = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        *reinterpret_cast<float4*>(&P[((size_t)blockIdx.y * M + m) * N + n]) = v;
      }
    }
  }
}

__global__ void w4_reduce_kernel(const float* __restrict__ P, __half* __restrict__ Y, int MN,
                                 int KS) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= MN) return;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < KS; ++k) {
    const float4 v = *reinterpret_cast<const float4*>(&P[(size_t)k * MN + i]);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  __half2* out = reinterpret_cast<__half2*>(&Y[i]);
  out[0] = __floats2half2_rn(s.x, s.y);
  out[1] = __floats2half2_rn(s.z, s.w);
}

template <int M, int WAVES>
static void launch_w4(const at::Tensor& a, const at::Tensor& qw, const at::Tensor& qz,
                      const at::Tensor& sc, at::Tensor& y, int KW) {
  const int K = a.size(1), N = qw.size(1);
  const int KB = WAVES * KW;
  TORCH_CHECK(K % KB == 0, "K=", K, " not a multiple of the block K range ", KB);
  TORCH_CHECK(N % 256 == 0, "N must be a multiple of 256");
  const int KS = K / KB;
  auto stream = at::hip::getCurrentHIPStream();
  at::Tensor p;
  if (KS > 1) p = at::empty({KS, M, N}, a.options().dtype(at::kFloat));
  const size_t smem = std::max((size_t)M * KB * 2, (size_t)(WAVES - 1) * M * 256 * 4);
  hipLaunchKernelGGL((w4_skinny_kernel<M, WAVES>), dim3(N / 256, KS), dim3(64 * WAVES), smem,
                     stream, reinterpret_cast<const __half*>(a.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qw.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qz.data_ptr()),
                     reinterpret_cast<const __half*>(sc.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()),
                     KS > 1 ? p.data_ptr<float>() : nullptr, N, K, KW);
  if (KS > 1) {
    const int MN = M * N;
    hipLaunchKernelGGL(w4_reduce_kernel, dim3((MN / 4 + 255) / 256), dim3(256), 0, stream,
                       p.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr()), MN, KS);
  }
}


// ---------------------------------------------------------------------------
// v2: activations through the scalar unit. All lanes of a wave read the same A
// values, so the wave index is made explicitly uniform (readfirstlane) and A
// is loaded with wave-uniform addresses -> s_load into SGPRs, consumed directly
// as the v_dot2 scalar operand. No LDS staging/barrier in the K loop, fewer
// VGPRs (higher occupancy), and the next group's weights are prefetched.
template <int M, int WAVES, int ABLATE = 0>
__global__ void __launch_bounds__(64 * WAVES)
w4_skinny_s_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ QW,
                   const uint32_t* __restrict__ QZ, const __half* __restrict__ SC,
                   __half* __restrict__ Y, float* __restrict__ P, int N, int K, int KW) {
  __shared__ float red[(WAVES > 1 ? WAVES - 1 : 1) * M * 256];
  const int lane = threadIdx.x & 63;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int n = blockIdx.x * 256 + lane * 4;
  const int KB = WAVES * KW;
  const int k_begin = blockIdx.y * KB + wave * KW;
  const int groups_n8 = N / 8;
  const bool active = n < N;
  const int nc = active ? n : 0;

  float acc[M][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[m][c] = 0.f;

  u32x4 qn[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    qn[j] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4*>(&QW[(size_t)((k_begin >> 3) + j) * N + nc]));
  for (int k = k_begin; k < k_begin + KW; k += 32) {
    u32x4 qv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) qv[j] = qn[j];
    if (k + 32 < k_begin + KW) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        qn[j] = __builtin_nontemporal_load(
            reinterpret_cast<const u32x4*>(&QW[(size_t)(((k + 32) >> 3) + j) * N + nc]));
    }
    const int g = k >> 5;
    const uint32_t zw = QZ[(size_t)g * groups_n8 + (nc >> 3)] >> ((nc & 7) * 4);
    const u32x2 sraw = *reinterpret_cast<const u32x2*>(&SC[(size_t)g * N + nc]);
    H2U s01, s23;
    s01.u = sraw.x;
    s23.u = sraw.y;
    h2_t z1[4], z16[4], s2[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int z = (zw >> (4 * c)) & 0xf;
      const _Float16 zf1 = (_Float16)(-1024 - z);
      const _Float16 zf16 = (_Float16)(-64 - z);
      z1[c] = {zf1, zf1};
      z16[c] = {zf16, zf16};
      const _Float16 s = c < 2 ? s01.h[c] : s23.h[c - 2];
      s2[c] = {s, s};
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h2_t dq[4][4];
      if (ABLATE == 1) {
        const uint32_t qq[4] = {qv[j].x, qv[j].y, qv[j].z, qv[j].w};
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
          for (int i = 0; i < 4; ++i) { H2U t; t.u = qq[c] ^ (i * 0x01010101u); dq[c][i] = t.h; }
      } else {
        dequant8(qv[j].x, z1[0], z16[0], s2[0], dq[0]);
        dequant8(qv[j].y, z1[1], z16[1], s2[1], dq[1]);
        dequant8(qv[j].z, z1[2], z16[2], s2[2], dq[2]);
        dequant8(qv[j].w, z1[3], z16[3], s2[3], dq[3]);
      }
      if (ABLATE == 2) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          acc[0][c] += (float)(dq[c][0].x + dq[c][1].y + dq[c][2].x + dq[c][3].y);
        continue;
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 av = *reinterpret_cast<const u32x4*>(&A[(size_t)m * K + k + j * 8]);
        const h2_t* a2 = reinterpret_cast<const h2_t*>(&av);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float r = acc[m][c];
#pragma unroll
          for (int i = 0; i < 4; ++i) r = __builtin_amdgcn_fdot2(dq[c][i], a2[i], r, false);
          acc[m][c] = r;
        }
      }
    }
  }

  if (WAVES > 1) {
    if (wave > 0) {
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int c = 0; c < 4; ++c) red[((wave - 1) * M + m) * 256 + lane * 4 + c] = acc[m][c];
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 1; w < WAVES; ++w)
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[m][c] += red[((w - 1) * M + m) * 256 + lane * 4 + c];
  }
  if (!active) return;
  if (gridDim.y == 1) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      __half2* out = reinterpret_cast<__half2*>(&Y[(size_t)m * N + n]);
      out[0] = __floats2half2_rn(acc[m][0], acc[m][1]);
      out[1] = __floats2half2_rn(acc[m][2], acc[m][3]);
    }
  } else {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float4 v = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
      *reinterpret_cast<float4*>(&P[((size_t)blockIdx.y * M + m) * N + n]) = v;
    }
  }
}

template <int M, int WAVES, int ABLATE = 0>
static void launch_w4s(const at::Tensor& a, const at::Tensor& qw, const at::Tensor& qz,
                       const at::Tensor& sc, at::Tensor& y, int KW) {
  const int K = a.size(1), N = qw.size(1);
  const int KB = WAVES * KW;
  TORCH_CHECK(K % KB == 0, "K=", K, " not a multiple of the block K range ", KB);
  TORCH_CHECK(N % 256 == 0, "N must be a multiple of 256");
  const int KS = K / KB;
  auto stream = at::hip::getCurrentHIPStream();
  at::Tensor p;
  if (KS > 1) p = at::empty({KS, M, N}, a.options().dtype(at::kFloat));
  hipLaunchKernelGGL((w4_skinny_s_kernel<M, WAVES, ABLATE>), dim3(N / 256, KS), dim3(64 * WAVES), 0,
                     stream, reinterpret_cast<const __half*>(a.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qw.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qz.data_ptr()),
                     reinterpret_cast<const __half*>(sc.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()),
                     KS > 1 ? p.data_ptr<float>() : nullptr, N, K, KW);
  if (KS > 1) {
    const int MN = M * N;
    hipLaunchKernelGGL(w4_reduce_kernel, dim3((MN / 4 + 255) / 256), dim3(256), 0, stream,
                       p.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr()), MN, KS);
  }
}


// ---------------------------------------------------------------------------
// v3: like v2 (scalar-unit activations, prefetch) but each thread owns COLS=2
// output columns: half the accumulators/dequant registers -> ~2x occupancy,
// and twice the threads for latency hiding. Loads are 8 B per lane.
template <int M, int WAVES>
__global__ void __launch_bounds__(64 * WAVES)
w4_skinny_c2_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ QW,
                    const uint32_t* __restrict__ QZ, const __half* __restrict__ SC,
                    __half* __restrict__ Y, float* __restrict__ P, int N, int K, int KW) {
  __shared__ float red[(WAVES > 1 ? WAVES - 1 : 1) * M * 128];
  const int lane = threadIdx.x & 63;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int n = blockIdx.x * 128 + lane * 2;
  const int KB = WAVES * KW;
  const int k_begin = blockIdx.y * KB + wave * KW;
  const int groups_n8 = N / 8;
  const bool active = n < N;
  const int nc = active ? n : 0;

  float acc[M][2];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m][0] = acc[m][1] = 0.f;

  u32x2 qn[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    qn[j] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x2*>(&QW[(size_t)((k_begin >> 3) + j) * N + nc]));
  for (int k = k_begin; k < k_begin + KW; k += 32) {
    u32x2 qv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) qv[j] = qn[j];
    if (k + 32 < k_begin + KW) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        qn[j] = __builtin_nontemporal_load(
            reinterpret_cast<const u32x2*>(&QW[(size_t)(((k + 32) >> 3) + j) * N + nc]));
    }
    const int g = k >> 5;
    const uint32_t zw = QZ[(size_t)g * groups_n8 + (nc >> 3)] >> ((nc & 7) * 4);
    H2U s01;
    s01.u = *reinterpret_cast<const uint32_t*>(&SC[(size_t)g * N + nc]);
    h2_t z1[2], z16[2], s2[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int z = (zw >> (4 * c)) & 0xf;
      const _Float16 zf1 = (_Float16)(-1024 - z);
      const _Float16 zf16 = (_Float16)(-64 - z);
      z1[c] = {zf1, zf1};
      z16[c] = {zf16, zf16};
      s2[c] = {s01.h[c], s01.h[c]};
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h2_t dq[2][4];
      dequant8(qv[j].x, z1[0], z16[0], s2[0], dq[0]);
      dequant8(qv[j].y, z1[1], z16[1], s2[1], dq[1]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 av = *reinterpret_cast<const u32x4*>(&A[(size_t)m * K + k + j * 8]);
        const h2_t* a2 = reinterpret_cast<const h2_t*>(&av);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          float r = acc[m][c];
#pragma unroll
          for (int i = 0; i < 4; ++i) r = __builtin_amdgcn_fdot2(dq[c][i], a2[i], r, false);
          acc[m][c] = r;
        }
      }
    }
  }

  if (WAVES > 1) {
    if (wave > 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        red[((wave - 1) * M + m) * 128 + lane * 2] = acc[m][0];
        red[((wave - 1) * M + m) * 128 + lane * 2 + 1] = acc[m][1];
      }
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 1; w < WAVES; ++w)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m][0] += red[((w - 1) * M + m) * 128 + lane * 2];
        acc[m][1] += red[((w - 1) * M + m) * 128 + lane * 2 + 1];
      }
  }
  if (!active) return;
  if (gridDim.y == 1) {
#pragma unroll
    for (int m = 0; m < M; ++m)
      *reinterpret_cast<__half2*>(&Y[(size_t)m * N + n]) = __floats2half2_rn(acc[m][0], acc[m][1]);
  } else {
#pragma unroll
    for (int m = 0; m < M; ++m)
      *reinterpret_cast<float2*>(&P[((size_t)blockIdx.y * M + m) * N + n]) =
          make_float2(acc[m][0], acc[m][1]);
  }
}

template <int M, int WAVES>
static void launch_w4c2(const at::Tensor& a, const at::Tensor& qw, const at::Tensor& qz,
                        const at::Tensor& sc, at::Tensor& y, int KW) {
  const int K = a.size(1), N = qw.size(1);
  const int KB = WAVES * KW;
  TORCH_CHECK(K % KB == 0, "K=", K, " not a multiple of the block K range ", KB);
  TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
  const int KS = K / KB;
  auto stream = at::hip::getCurrentHIPStream();
  at::Tensor p;
  if (KS > 1) p = at::empty({KS, M, N}, a.options().dtype(at::kFloat));
  hipLaunchKernelGGL((w4_skinny_c2_kernel<M, WAVES>), dim3(N / 128, KS), dim3(64 * WAVES), 0,
                     stream, reinterpret_cast<const __half*>(a.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qw.data_ptr()),
                     reinterpret_cast<const uint32_t*>(qz.data_ptr()),
                     reinterpret_cast<const __half*>(sc.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()),
                     KS > 1 ? p.data_ptr<float>() : nullptr, N, K, KW);
  if (KS > 1) {
    const int MN = M * N;
    hipLaunchKernelGGL(w4_reduce_kernel, dim3((MN / 4 + 255) / 256), dim3(256), 0, stream,
                       p.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr()), MN, KS);
  }
}

// a: [M, K] fp16; qw: [K/8, N] int32 (Exllama-shuffled); qz: [K/32, N/8] int32;
// sc: [K/32, N] fp16. Group size 32, v2 zero points.
at::Tensor w4_gemm(const at::Tensor& a, const at::Tensor& qw, const at::Tensor& qz,
                   const at::Tensor& sc, int64_t kw, int64_t waves) {
  TORCH_CHECK(a.dtype() == at::kHalf && sc.dtype() == at::kHalf);
  TORCH_CHECK(a.is_contiguous() && qw.is_contiguous() && qz.is_contiguous() && sc.is_contiguous());
  const int M = a.size(0), K = a.size(1), N = qw.size(1);
  TORCH_CHECK(qw.size(0) * 8 == K && sc.size(0) * 32 == K && M >= 1 && M <= 8);
  auto y = at::empty({M, N}, a.options());
#define CASE(MM)                                              \
  case MM:                                                    \
    if (waves == 2) launch_w4<MM, 2>(a, qw, qz, sc, y, kw);   \
    else if (waves == 4) launch_w4<MM, 4>(a, qw, qz, sc, y, kw); \
    else if (waves == 12) launch_w4s<MM, 2>(a, qw, qz, sc, y, kw); \
    else if (waves == 14) launch_w4s<MM, 4>(a, qw, qz, sc, y, kw); \
    else if (waves == 22) launch_w4c2<MM, 2>(a, qw, qz, sc, y, kw); \
    else if (waves == 41) launch_w4s<MM, 4, 1>(a, qw, qz, sc, y, kw); \
    else if (waves == 42) launch_w4s<MM, 4, 2>(a, qw, qz, sc, y, kw); \
    else if (waves == 24) launch_w4c2<MM, 4>(a, qw, qz, sc, y, kw); \
    else launch_w4s<MM, 1>(a, qw, qz, sc, y, kw);             \
    break;
  switch (M) { CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) }
#undef CASE
  return y;
}

TORCH_LIBRARY_FRAGMENT(gfx906_skinny, m) {
  m.def("w4gemm(Tensor a, Tensor qw, Tensor qz, Tensor sc, int kw, int waves) -> Tensor");
}
TORCH_LIBRARY_IMPL(gfx906_skinny, CUDA, m) { m.impl("w4gemm", &w4_gemm); }
