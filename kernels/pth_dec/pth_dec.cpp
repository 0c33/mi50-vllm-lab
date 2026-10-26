// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sami (github.com/0c33)
// Torch binding for the gfx906 int8 per-token-head split-K decode attention (pth_dec.cu).
// No allocation and no host synchronisation: safe inside HIP graph capture.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

extern "C" hipError_t pth_dec_launch(
    const void* q, int q_st, const void* kc, int k_sb, int k_ss, int k_sh,
    const int* block_table, int bt_stride, const int* seq_lens, const int* qsl,
    int num_seqs, int num_tokens, int heads_q, int heads_kv, int block_size, float scale,
    int qmax, float* part_o, float* part_m, float* part_l, int nseg, int nsplit,
    void* out, long o_st, long o_sh, hipStream_t stream);

static bool same_rows(const torch::Tensor& scale, const torch::Tensor& cache, int64_t byte_off) {
    // fp32 scale view [blocks, block_size, Hkv] must sit byte_off bytes into each cache row
    const auto base = reinterpret_cast<uintptr_t>(cache.data_ptr());
    return scale.dim() == 3 && scale.dtype() == torch::kFloat32 &&
           reinterpret_cast<uintptr_t>(scale.data_ptr()) == base + byte_off &&
           scale.stride(0) * 4 == cache.stride(0) && scale.stride(1) * 4 == cache.stride(1) &&
           scale.stride(2) * 4 == cache.stride(2);
}

// True if the cache views have vLLM's int8 per-token-head row layout that the kernel assumes:
// K row [256 int8 | f32 scale] followed by the V row [256 int8 | f32 scale] (V at +260 bytes).
bool supported_layout(torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor k_scale,
                      torch::Tensor v_scale) {
    if (key_cache.dtype() != torch::kInt8 || value_cache.dtype() != torch::kInt8 || key_cache.dim() != 4 ||
        value_cache.dim() != 4 || key_cache.size(3) != 260 || value_cache.size(3) != 260 ||
        key_cache.stride(3) != 1 || value_cache.stride(3) != 1)
        return false;
    for (int i = 0; i < 3; ++i)
        if (key_cache.stride(i) != value_cache.stride(i) || key_cache.size(i) != value_cache.size(i)) return false;
    const auto kb = reinterpret_cast<uintptr_t>(key_cache.data_ptr());
    if (reinterpret_cast<uintptr_t>(value_cache.data_ptr()) != kb + 260) return false;
    if (kb % 8 || key_cache.stride(0) % 8 || key_cache.stride(1) % 8 || key_cache.stride(2) % 8) return false;
    if (key_cache.stride(0) > INT32_MAX || key_cache.stride(1) > INT32_MAX || key_cache.stride(2) > INT32_MAX)
        return false;
    return same_rows(k_scale, key_cache, 256) && same_rows(v_scale, key_cache, 516);
}

// q:            fp16 [T, Hq, 256], head stride 256
// key_cache / value_cache / k_scale / v_scale: see supported_layout()
// block_table:  int32 [>= num_seqs, max_blocks]; seq_lens: int32 [>= num_seqs]
// query_start_loc: int32 [>= num_seqs + 1]; every sequence has <= qmax (1..4) query tokens
// out:          fp16 [T, Hq, 256] (dim stride 1)
// part_o:       fp32 [>= T, Hq, NSEG, 256] contiguous; part_m / part_l: fp32 [>= T, Hq, NSEG]
void decode(torch::Tensor q, torch::Tensor key_cache, torch::Tensor value_cache,
            torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor block_table,
            torch::Tensor seq_lens, torch::Tensor query_start_loc, torch::Tensor out,
            double scale, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
            int64_t num_seqs, int64_t qmax, int64_t nsplit) {
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kFloat16 && q.dim() == 3 && q.size(2) == 256 &&
                q.stride(2) == 1 && q.stride(1) == 256 && q.stride(0) % 2 == 0 && q.stride(0) <= INT32_MAX &&
                (reinterpret_cast<uintptr_t>(q.data_ptr()) % 4) == 0,
                "q must be fp16 [T, Hq, 256] with contiguous heads");
    TORCH_CHECK(out.dtype() == torch::kFloat16 && out.dim() == 3 && out.size(0) == q.size(0) &&
                out.size(1) == q.size(1) && out.size(2) == 256 && out.stride(2) == 1,
                "out must be fp16 [T, Hq, 256] with unit dim stride");
    TORCH_CHECK(supported_layout(key_cache, value_cache, k_scale, v_scale),
                "KV cache views do not have the int8 per-token-head row layout");
    TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 &&
                query_start_loc.dtype() == torch::kInt32 && block_table.dim() == 2 &&
                block_table.stride(1) == 1 && block_table.stride(0) <= INT32_MAX && seq_lens.is_contiguous() &&
                query_start_loc.is_contiguous(), "block_table / seq_lens / query_start_loc must be int32");
    TORCH_CHECK(part_o.dtype() == torch::kFloat32 && part_o.is_contiguous() && part_o.dim() == 4 &&
                part_o.size(3) == 256 && part_m.dtype() == torch::kFloat32 && part_l.dtype() == torch::kFloat32 &&
                part_m.is_contiguous() && part_l.is_contiguous() && part_m.dim() == 3 && part_l.dim() == 3,
                "partial buffers must be contiguous fp32");
    const int T = q.size(0), Hq = q.size(1), Hkv = key_cache.size(2);
    const int nseg = part_o.size(2);
    TORCH_CHECK(part_o.size(0) >= T && part_o.size(1) == Hq && part_m.size(0) >= T &&
                part_m.size(1) == Hq && part_m.size(2) == nseg && part_l.size(0) >= T &&
                part_l.size(1) == Hq && part_l.size(2) == nseg, "partial buffers too small");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0 && Hq / Hkv == 6, "only GQA 6 is built");
    TORCH_CHECK(nsplit >= 1 && nsplit <= nseg, "nsplit must be in [1, NSEG]");
    TORCH_CHECK(qmax >= 1 && qmax <= 4, "qmax must be 1..4");
    TORCH_CHECK(num_seqs >= 1 && seq_lens.size(0) >= num_seqs && query_start_loc.size(0) >= num_seqs + 1 &&
                block_table.size(0) >= num_seqs, "per-sequence tensors too small");
    if (T == 0) return;
    const hipStream_t stream = c10::hip::getCurrentHIPStream();
    const hipError_t e = pth_dec_launch(
        q.data_ptr(), (int)q.stride(0), key_cache.data_ptr(), (int)key_cache.stride(0),
        (int)key_cache.stride(1), (int)key_cache.stride(2), block_table.data_ptr<int>(),
        (int)block_table.stride(0), seq_lens.data_ptr<int>(), query_start_loc.data_ptr<int>(),
        (int)num_seqs, T, Hq, Hkv, (int)key_cache.size(1), (float)scale, (int)qmax,
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), nseg, (int)nsplit,
        out.data_ptr(), out.stride(0), out.stride(1), stream);
    TORCH_CHECK(e == hipSuccess, "pth_dec launch failed: ", hipGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode, "gfx906 split-K decode attention over the int8 per-token-head KV cache");
    m.def("supported_layout", &supported_layout, "True if the KV cache views have the expected row layout");
}
