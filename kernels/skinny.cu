// Skinny fp16 GEMM for gfx906: Y[M, N] = X[M, K] @ W[N, K]^T with M <= 8.
//
// Built for the shapes a DFlash2 k=7 step runs at M=7/8 rows (target lm_head
// verify, draft lm_head, draft layers). Each wave streams YT weight rows with
// coalesced 16-byte loads (lanes along K); X is staged per K-chunk in LDS once
// per workgroup and read back per lane, so X is not re-read from L2 for every
// weight row. Partial sums (YT * MP per lane) are folded with a
// recursive-halving butterfly: log2(64) exchanges move a shrinking set of
// values instead of reducing each value separately.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float dot8(const u32x4& w, const u32x4& x, float acc) {
  const h2_t* wh = reinterpret_cast<const h2_t*>(&w);
  const h2_t* xh = reinterpret_cast<const h2_t*>(&x);
#pragma unroll
  for (int i = 0; i < 4; ++i) acc = __builtin_amdgcn_fdot2(wh[i], xh[i], acc, false);
  return acc;
}

// Sum V values across the 64 lanes of a wave by recursive halving: each step
// pairs lanes differing in one bit and each keeps half its values, so V values
// cost V-1 exchanges (not 6*V). Afterwards lanes whose top log2(V) bits equal
// j hold the full sum of value j.
template <int V, int N, int BIT>
__device__ __forceinline__ void butterfly(float (&v)[V], int lane) {
  if constexpr (BIT >= 0) {
    if constexpr (N > 1) {
      constexpr int H = N / 2;
      const bool upper = (lane >> BIT) & 1;
#pragma unroll
      for (int i = 0; i < H; ++i) {
        const float send = upper ? v[i] : v[i + H];
        const float keep = upper ? v[i + H] : v[i];
        v[i] = keep + __shfl_xor(send, 1 << BIT);
      }
      butterfly<V, H, BIT - 1>(v, lane);
    } else {
      v[0] += __shfl_xor(v[0], 1 << BIT);
      butterfly<V, 1, BIT - 1>(v, lane);
    }
  }
}

template <int M, int MP, int YT, int KC, int WAVES>
__global__ void __launch_bounds__(64 * WAVES)
skinny_gemm_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                   __half* __restrict__ Y, int N, int K) {
  constexpr int V = YT * MP;
  static_assert((V & (V - 1)) == 0 && V <= 64, "YT*MP must be a power of two <= 64");
  __shared__ __align__(16) __half xs[M * KC];
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int row0 = (blockIdx.x * WAVES + wave) * YT;

  float acc[V];
#pragma unroll
  for (int i = 0; i < V; ++i) acc[i] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    const int kc = min(KC, K - k0);
    __syncthreads();
    for (int i = threadIdx.x * 8; i < M * kc; i += 64 * WAVES * 8) {
      const int m = i / kc, kk = i - m * kc;
      *reinterpret_cast<u32x4*>(&xs[m * KC + kk]) =
          *reinterpret_cast<const u32x4*>(&X[(size_t)m * K + k0 + kk]);
    }
    __syncthreads();
    for (int kk = lane * 8; kk < kc; kk += 512) {
      u32x4 w[YT];
#pragma unroll
      for (int r = 0; r < YT; ++r) {
        const int row = row0 + r;
        w[r] = row < N ? __builtin_nontemporal_load(
                             reinterpret_cast<const u32x4*>(&W[(size_t)row * K + k0 + kk]))
                       : u32x4{0, 0, 0, 0};
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 xv = *reinterpret_cast<const u32x4*>(&xs[m * KC + kk]);
#pragma unroll
        for (int r = 0; r < YT; ++r) acc[r * MP + m] = dot8(w[r], xv, acc[r * MP + m]);
      }
    }
  }

  butterfly<V, V, 5>(acc, lane);
  // After the butterfly, value index j = r * MP + m lives (fully summed) in
  // lanes whose top log2(V) bits equal j.
  constexpr int LOGV = __builtin_ctz(V);
  const int j = lane >> (6 - LOGV);
  if ((lane & ((1 << (6 - LOGV)) - 1)) == 0) {
    const int r = j / MP, m = j - r * MP;
    const int row = row0 + r;
    if (m < M && row < N) Y[(size_t)m * N + row] = __float2half(acc[0]);
  }
}

template <int M, int MP, int YT, int KC, int WAVES>
static void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& y) {
  const int N = w.size(0), K = w.size(1);
  const int rows_per_block = YT * WAVES;
  dim3 grid((N + rows_per_block - 1) / rows_per_block);
  hipLaunchKernelGGL((skinny_gemm_kernel<M, MP, YT, KC, WAVES>), grid, dim3(64 * WAVES), 0,
                     at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const __half*>(x.data_ptr()),
                     reinterpret_cast<const __half*>(w.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()), N, K);
}


// ---------------------------------------------------------------------------
// Split-K variant: grid (row_groups, KS). Each workgroup stages X[:, kslice]
// (M x KCH halves) into LDS once, then its waves stream many row tiles of
// W[:, kslice] with no barrier in the loop. Partials go to an fp32 workspace
// [KS][M][N] that a second kernel folds into fp16 (KS == 1 writes fp16
// directly).
template <int M, int MP, int YT, int J, int WAVES>
__global__ void __launch_bounds__(64 * WAVES)
skinny_splitk_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                     __half* __restrict__ Y, float* __restrict__ P, int N, int K,
                     int tiles_per_wave) {
  constexpr int KCH = J * 512;
  constexpr int V = YT * MP;
  __shared__ __align__(16) __half xs[M * KCH];
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int ks = blockIdx.y;
  const int k0 = ks * KCH;
  for (int i = threadIdx.x * 8; i < M * KCH; i += 64 * WAVES * 8) {
    const int m = i / KCH, kk = i - m * KCH;
    *reinterpret_cast<u32x4*>(&xs[m * KCH + kk]) =
        *reinterpret_cast<const u32x4*>(&X[(size_t)m * K + k0 + kk]);
  }
  __syncthreads();
  const int rows_per_block = tiles_per_wave * WAVES * YT;
  const int row_base = blockIdx.x * rows_per_block;
  constexpr int LOGV = __builtin_ctz(V);
  for (int t = 0; t < tiles_per_wave; ++t) {
    const int row0 = row_base + (t * WAVES + wave) * YT;
    if (row0 >= N) break;
    float acc[V];
#pragma unroll
    for (int i = 0; i < V; ++i) acc[i] = 0.f;
    u32x4 w[YT][J];
#pragma unroll
    for (int r = 0; r < YT; ++r) {
      const int row = min(row0 + r, N - 1);
#pragma unroll
      for (int j = 0; j < J; ++j)
        w[r][j] = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(
            &W[(size_t)row * K + k0 + j * 512 + lane * 8]));
    }
#pragma unroll
    for (int j = 0; j < J; ++j) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 xv = *reinterpret_cast<const u32x4*>(&xs[m * KCH + j * 512 + lane * 8]);
#pragma unroll
        for (int r = 0; r < YT; ++r) acc[r * MP + m] = dot8(w[r][j], xv, acc[r * MP + m]);
      }
    }
    butterfly<V, V, 5>(acc, lane);
    if ((lane & ((1 << (6 - LOGV)) - 1)) == 0) {
      const int jv = lane >> (6 - LOGV);
      const int r = jv / MP, m = jv - r * MP;
      const int row = row0 + r;
      if (m < M && row < N) {
        if (gridDim.y == 1) Y[(size_t)m * N + row] = __float2half(acc[0]);
        else P[((size_t)ks * M + m) * N + row] = acc[0];
      }
    }
  }
}

__global__ void splitk_reduce_kernel(const float* __restrict__ P, __half* __restrict__ Y,
                                     int MN, int KS) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= MN) return;
  float s = 0.f;
  for (int k = 0; k < KS; ++k) s += P[(size_t)k * MN + i];
  Y[i] = __float2half(s);
}

template <int M, int MP, int YT, int J, int WAVES>
static void launch_splitk(const at::Tensor& x, const at::Tensor& w, at::Tensor& y,
                          int target_blocks) {
  const int N = w.size(0), K = w.size(1);
  constexpr int KCH = J * 512;
  TORCH_CHECK(K % KCH == 0, "K must be a multiple of ", KCH);
  const int KS = K / KCH;
  const int rows_per_tile_group = WAVES * YT;
  const int groups = (N + rows_per_tile_group - 1) / rows_per_tile_group;
  // pick tiles_per_wave so grid.x * KS ~ target_blocks
  int tiles = max(1, (groups * KS + target_blocks - 1) / target_blocks);
  const int gx = (groups + tiles - 1) / tiles;
  auto stream = at::hip::getCurrentHIPStream();
  at::Tensor p;
  if (KS > 1) p = at::empty({KS, M, N}, x.options().dtype(at::kFloat));
  hipLaunchKernelGGL((skinny_splitk_kernel<M, MP, YT, J, WAVES>), dim3(gx, KS),
                     dim3(64 * WAVES), 0, stream,
                     reinterpret_cast<const __half*>(x.data_ptr()),
                     reinterpret_cast<const __half*>(w.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()),
                     KS > 1 ? p.data_ptr<float>() : nullptr, N, K, tiles);
  if (KS > 1) {
    const int MN = M * N;
    hipLaunchKernelGGL(splitk_reduce_kernel, dim3((MN + 255) / 256), dim3(256), 0, stream,
                       p.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr()), MN, KS);
  }
}


// ---------------------------------------------------------------------------
// Pipelined variant: same LDS-staged X as skinny_gemm_kernel, but the weight
// stream is software-pipelined across the whole K loop (including X-chunk
// boundaries): the next 512-half step's YT loads are issued before the
// current step's dot products, so memory stays busy during the math.
template <int M, int MP, int YT, int KC, int WAVES>
__global__ void __launch_bounds__(64 * WAVES)
skinny_pipe_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                   __half* __restrict__ Y, int N, int K) {
  constexpr int V = YT * MP;
  constexpr int STEPS_PER_CHUNK = KC / 512;
  __shared__ __align__(16) __half xs[M * KC];
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int row0 = (blockIdx.x * WAVES + wave) * YT;
  const __half* wrow[YT];
#pragma unroll
  for (int r = 0; r < YT; ++r) wrow[r] = W + (size_t)min(row0 + r, N - 1) * K + lane * 8;

  float acc[V];
#pragma unroll
  for (int i = 0; i < V; ++i) acc[i] = 0.f;

  const int steps = K / 512;
  u32x4 wc[YT], wn[YT];
#pragma unroll
  for (int r = 0; r < YT; ++r)
    wc[r] = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(wrow[r]));
  for (int s = 0; s < steps; ++s) {
    const int sc = s % STEPS_PER_CHUNK;
    if (sc == 0) {
      const int k0 = s * 512;
      const int kc = min(KC, K - k0);
      __syncthreads();
      for (int i = threadIdx.x * 8; i < M * kc; i += 64 * WAVES * 8) {
        const int m = i / kc, kk = i - m * kc;
        *reinterpret_cast<u32x4*>(&xs[m * KC + kk]) =
            *reinterpret_cast<const u32x4*>(&X[(size_t)m * K + k0 + kk]);
      }
      __syncthreads();
    }
    if (s + 1 < steps) {
#pragma unroll
      for (int r = 0; r < YT; ++r)
        wn[r] = __builtin_nontemporal_load(
            reinterpret_cast<const u32x4*>(wrow[r] + (s + 1) * 512));
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const u32x4 xv = *reinterpret_cast<const u32x4*>(&xs[m * KC + sc * 512 + lane * 8]);
#pragma unroll
      for (int r = 0; r < YT; ++r) acc[r * MP + m] = dot8(wc[r], xv, acc[r * MP + m]);
    }
#pragma unroll
    for (int r = 0; r < YT; ++r) wc[r] = wn[r];
  }

  butterfly<V, V, 5>(acc, lane);
  constexpr int LOGV = __builtin_ctz(V);
  if ((lane & ((1 << (6 - LOGV)) - 1)) == 0) {
    const int j = lane >> (6 - LOGV);
    const int r = j / MP, m = j - r * MP;
    const int row = row0 + r;
    if (m < M && row < N) Y[(size_t)m * N + row] = __float2half(acc[0]);
  }
}

template <int M, int MP, int YT, int KC, int WAVES>
static void launch_pipe(const at::Tensor& x, const at::Tensor& w, at::Tensor& y) {
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(K % 512 == 0, "pipelined kernel needs K % 512 == 0");
  const int rows_per_block = YT * WAVES;
  dim3 grid((N + rows_per_block - 1) / rows_per_block);
  hipLaunchKernelGGL((skinny_pipe_kernel<M, MP, YT, KC, WAVES>), grid, dim3(64 * WAVES), 0,
                     at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const __half*>(x.data_ptr()),
                     reinterpret_cast<const __half*>(w.data_ptr()),
                     reinterpret_cast<__half*>(y.data_ptr()), N, K);
}

// variant: 0 = YT4/KC2048/4 waves, 1 = YT2, 2 = YT8 (M<=4), 3 = KC1024
at::Tensor skinny_gemm(const at::Tensor& x, const at::Tensor& w, int64_t variant) {
  TORCH_CHECK(x.dtype() == at::kHalf && w.dtype() == at::kHalf);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous());
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(w.size(1) == K && K % 8 == 0 && M >= 1 && M <= 8);
  auto y = at::empty({M, w.size(0)}, x.options());
#define CASE(MM, MP_)                                                     \
  case MM:                                                                \
    if (variant == 1) launch<MM, MP_, 2, 2048, 4>(x, w, y);               \
    else if (variant == 3) launch<MM, MP_, 4, 1024, 4>(x, w, y);          \
    else if (variant == 4) launch<MM, MP_, 4, 2048, 8>(x, w, y);          \
    else if (variant == 20) launch_pipe<MM, MP_, 4, 2048, 8>(x, w, y);   \
    else if (variant == 21) launch_pipe<MM, MP_, 2, 2048, 8>(x, w, y);   \
    else if (variant == 22) launch_pipe<MM, MP_, 4, 1024, 8>(x, w, y);   \
    else if (variant == 23) launch_pipe<MM, MP_, 2, 1024, 4>(x, w, y);   \
    else if (variant == 24) launch_pipe<MM, MP_, 4, 2048, 4>(x, w, y);   \
    else if (variant == 10) launch_splitk<MM, MP_, 4, 2, 4>(x, w, y, 960);  \
    else if (variant == 11) launch_splitk<MM, MP_, 4, 1, 4>(x, w, y, 960);  \
    else if (variant == 12) launch_splitk<MM, MP_, 2, 2, 4>(x, w, y, 960);  \
    else if (variant == 13) launch_splitk<MM, MP_, 4, 2, 8>(x, w, y, 480);  \
    else if (variant == 14) launch_splitk<MM, MP_, 4, 2, 4>(x, w, y, 1920); \
    else launch<MM, MP_, 4, 2048, 4>(x, w, y);                            \
    break;
  switch (M) {
    CASE(1, 1) CASE(2, 2) CASE(3, 4) CASE(4, 4) CASE(5, 8) CASE(6, 8) CASE(7, 8) CASE(8, 8)
  }
#undef CASE
  return y;
}

TORCH_LIBRARY(gfx906_skinny, m) {
  m.def("gemm(Tensor x, Tensor w, int variant=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(gfx906_skinny, CUDA, m) { m.impl("gemm", &skinny_gemm); }
