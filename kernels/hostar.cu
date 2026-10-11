// Two-GPU all-reduce through pinned host memory, for TP=2 boxes without GPU P2P.
//
// RCCL on this host routes TP=2 traffic through host shared memory with a proxy
// thread and costs ~75-80 us per call regardless of size; spec-decode verify
// steps issue ~140 small (80 KB) all-reduces. Here each GPU writes its slice
// straight into a shared, hipHostRegister'ed buffer over PCIe, publishes a
// generation flag, waits for the peer's flag and adds the peer's slice.
//
// Protocol (one workgroup per call):
//  - generation g comes from a device-side counter (CUDA-graph replay safe);
//  - data slot parity = g & 1. A rank reuses a slot only at g+2, which it can
//    reach only after seeing the peer's flag for g+1, which the peer publishes
//    only after it finished reading slot g -> no overwrite race;
//  - sums are a + b in the same order on both ranks -> bit-identical results.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

namespace {

constexpr size_t kFlagBytes = 4096;
constexpr int kThreads = 1024;
constexpr int kMaxVecPerThread = 8;  // 1024 * 8 * 16 B = 128 KB per call
constexpr size_t kMaxBytes = (size_t)kThreads * kMaxVecPerThread * 16;

struct HostAR {
  void* host;
  char* dev;
  size_t total;
  size_t slot;
  int64_t* counter;
};
std::vector<HostAR> g_ars;

__device__ __forceinline__ u32x4 add_vec(const u32x4& a, const u32x4& b, bool is_half) {
  u32x4 r;
  if (is_half) {
    const __half2* ah = reinterpret_cast<const __half2*>(&a);
    const __half2* bh = reinterpret_cast<const __half2*>(&b);
    __half2* rh = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) rh[i] = __hadd2(ah[i], bh[i]);
  } else {
    const float* af = reinterpret_cast<const float*>(&a);
    const float* bf = reinterpret_cast<const float*>(&b);
    float* rf = reinterpret_cast<float*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) rf[i] = af[i] + bf[i];
  }
  return r;
}

__global__ void __launch_bounds__(kThreads)
hostar_kernel(const u32x4* __restrict__ in, u32x4* __restrict__ out, int nvec, char* base,
              size_t slot, int64_t* counter, int rank, bool is_half) {
  __shared__ int64_t s_gen;
  if (threadIdx.x == 0) {
    const int64_t g = *counter + 1;
    *counter = g;
    s_gen = g;
  }
  __syncthreads();
  const int64_t gen = s_gen;
  const int par = (int)(gen & 1);
  u32x4* mine = reinterpret_cast<u32x4*>(base + kFlagBytes + (size_t)(par * 2 + rank) * slot);
  const u32x4* peer =
      reinterpret_cast<const u32x4*>(base + kFlagBytes + (size_t)(par * 2 + (rank ^ 1)) * slot);
  int64_t* my_flag = reinterpret_cast<int64_t*>(base + rank * 256);
  int64_t* peer_flag = reinterpret_cast<int64_t*>(base + (rank ^ 1) * 256);

  u32x4 v[kMaxVecPerThread];
#pragma unroll
  for (int j = 0; j < kMaxVecPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < nvec) {
      v[j] = in[i];
      __builtin_nontemporal_store(v[j], &mine[i]);
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_store(my_flag, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    long spins = 0;
    while (__hip_atomic_load(peer_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < gen) {
      __builtin_amdgcn_s_sleep(1);
      if (++spins > (1L << 32)) __builtin_trap();  // peer never arrived: fail loudly
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kMaxVecPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < nvec) {
      const u32x4 o = __builtin_nontemporal_load(&peer[i]);
      out[i] = rank == 0 ? add_vec(v[j], o, is_half) : add_vec(o, v[j], is_half);
    }
  }
}

}  // namespace

int64_t hostar_create(std::string name, int64_t slot_bytes, bool create) {
  const size_t total = kFlagBytes + 4 * (size_t)slot_bytes;
  const int fd = shm_open(name.c_str(), create ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
  TORCH_CHECK(fd >= 0, "hostar: shm_open(", name, ") failed");
  if (create) TORCH_CHECK(ftruncate(fd, total) == 0, "hostar: ftruncate failed");
  void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  TORCH_CHECK(p != MAP_FAILED, "hostar: mmap failed");
  if (create) memset(p, 0, total);
  TORCH_CHECK(hipHostRegister(p, total, hipHostRegisterMapped | hipHostRegisterPortable) ==
                  hipSuccess,
              "hostar: hipHostRegister failed");
  void* d = nullptr;
  TORCH_CHECK(hipHostGetDevicePointer(&d, p, 0) == hipSuccess, "hostar: no device pointer");
  int64_t* ctr = nullptr;
  TORCH_CHECK(hipMalloc(&ctr, 256) == hipSuccess);
  TORCH_CHECK(hipMemset(ctr, 0, 256) == hipSuccess);
  TORCH_CHECK(hipDeviceSynchronize() == hipSuccess);
  g_ars.push_back({p, reinterpret_cast<char*>(d), total, (size_t)slot_bytes, ctr});
  return (int64_t)g_ars.size() - 1;
}

void hostar_unlink(std::string name) { shm_unlink(name.c_str()); }

int64_t hostar_max_bytes() { return (int64_t)kMaxBytes; }

at::Tensor hostar_allreduce(const at::Tensor& inp, int64_t handle, int64_t rank) {
  TORCH_CHECK(handle >= 0 && handle < (int64_t)g_ars.size());
  const HostAR& h = g_ars[handle];
  TORCH_CHECK(inp.is_contiguous());
  TORCH_CHECK(inp.dtype() == at::kHalf || inp.dtype() == at::kFloat);
  const size_t bytes = inp.numel() * inp.element_size();
  TORCH_CHECK(bytes % 16 == 0 && bytes <= kMaxBytes && bytes <= h.slot);
  auto out = at::empty_like(inp);
  hipLaunchKernelGGL(hostar_kernel, dim3(1), dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const u32x4*>(inp.data_ptr()),
                     reinterpret_cast<u32x4*>(out.data_ptr()), (int)(bytes / 16), h.dev, h.slot,
                     h.counter, (int)rank, inp.dtype() == at::kHalf);
  return out;
}

TORCH_LIBRARY_FRAGMENT(gfx906_skinny, m) {
  m.def("hostar_create(str name, int slot_bytes, bool create) -> int");
  m.def("hostar_unlink(str name) -> ()");
  m.def("hostar_max_bytes() -> int");
  m.def("hostar_allreduce(Tensor inp, int handle, int rank) -> Tensor");
}
TORCH_LIBRARY_IMPL(gfx906_skinny, CompositeExplicitAutograd, m) {
  m.impl("hostar_create", &hostar_create);
  m.impl("hostar_unlink", &hostar_unlink);
  m.impl("hostar_max_bytes", &hostar_max_bytes);
}
TORCH_LIBRARY_IMPL(gfx906_skinny, CUDA, m) { m.impl("hostar_allreduce", &hostar_allreduce); }
