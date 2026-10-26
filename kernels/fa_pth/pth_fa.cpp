// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sami (github.com/0c33)
// gfx906 FlashAttention (llama.cpp-derived Q8 tile kernel, vendored in the
// KIntegrated fork) reading vLLM's int8 per-token-head KV cache in place.
//
// Each (block, slot, kv-head) row of that cache is [K: D int8 | fp32 scale |
// V: D int8 | fp32 scale]. The direct-paged kernel was taught to read such rows
// (pth=1, any block size): K quants feed the dp4a Q.K path with the row scale
// as the q8_0 block scale, V is dequantized to fp16 while it fills the tile.
// No gather buffers, so no extra VRAM beyond the fp32 Q and output.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" hipError_t gfx906_fa_launch_paged(
    const float *Q_fp32, const void *K_paged, const __half *V_paged,
    const int32_t *block_table, const int32_t *kv_max_d, float *O_fp32, float2 *O_meta,
    const __half *MASK_f16, int32_t mask_seq_kv_padded, const int32_t *Q_ABS_OFFSET_d,
    int window, const int32_t *KV_START_d, int tile_clip, int batch, int heads_q,
    int heads_kv, int seq_q, int max_seq_kv, int head_dim, int block_size,
    int max_blocks_per_seq, int64_t k_block_stride, int64_t k_token_stride,
    int64_t k_head_stride, int64_t v_block_stride, int64_t v_token_stride,
    int64_t v_head_stride, float scale, hipStream_t stream, int kv_split, int pth);

static int pick_ncols1(int seq_q) {  // LOCKSTEP with the launcher's ladder
    if (seq_q > 32) return 64;
    if (seq_q > 16) return 32;
    if (seq_q > 8) return 16;
    if (seq_q > 4) return 8;
    if (seq_q > 2) return 4;
    return 2;
}

// q:            fp32 [B, Hq, Sq, D] contiguous (Sq padded to the ncols1 ladder)
// key_cache:    int8 [num_blocks, block_size, Hkv, >= D+4] view (K quants + fp32 scale)
// value_cache:  int8 [num_blocks, block_size, Hkv, >= D+4] view (V quants + fp32 scale)
// block_table:  int32 [B, max_blocks]; seq_lens: int32 [B] (kv length per sequence)
// q_abs_offset: int32 [B] absolute position of each sequence's first query row
// returns       fp32 [B, Sq, Hq, D]
torch::Tensor forward_pth(torch::Tensor q, torch::Tensor key_cache, torch::Tensor value_cache,
                          torch::Tensor block_table, torch::Tensor seq_lens, double scale,
                          torch::Tensor q_abs_offset) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dtype() == torch::kFloat32 && q.dim() == 4,
                "q must be contiguous fp32 [B, Hq, Sq, D]");
    TORCH_CHECK(key_cache.dtype() == torch::kInt8 && value_cache.dtype() == torch::kInt8,
                "key/value cache views must be int8");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4, "cache views must be 4D");
    TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1, "cache rows must be contiguous");
    TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 &&
                q_abs_offset.dtype() == torch::kInt32, "block_table/seq_lens/q_abs_offset must be int32");
    const int batch = q.size(0), heads_q = q.size(1), seq_q = q.size(2), head_dim = q.size(3);
    const int block_size = key_cache.size(1), heads_kv = key_cache.size(2);
    TORCH_CHECK(head_dim == 256, "head_dim must be 256 (only instantiation built)");
    TORCH_CHECK(key_cache.size(3) >= head_dim + 4 && value_cache.size(3) >= head_dim + 4,
                "cache rows must hold D int8 values and a fp32 scale");
    TORCH_CHECK(heads_q % heads_kv == 0, "heads_q must be a multiple of heads_kv");
    TORCH_CHECK(seq_q % pick_ncols1(seq_q) == 0, "Sq must be padded to the ncols1 ladder");
    TORCH_CHECK(block_table.size(0) == batch && seq_lens.size(0) == batch && q_abs_offset.size(0) == batch,
                "per-sequence tensors must have B rows");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(key_cache.data_ptr()) % 8) == 0 &&
                (reinterpret_cast<uintptr_t>(value_cache.data_ptr()) % 4) == 0 &&
                key_cache.stride(1) % 8 == 0 && key_cache.stride(2) % 8 == 0 && key_cache.stride(0) % 8 == 0,
                "K rows must be 8-byte aligned and V rows 4-byte aligned");

    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto qo = q_abs_offset.contiguous();
    const int max_blocks = bt.size(1);
    const int grid_x = (seq_q + pick_ncols1(seq_q) - 1) / pick_ncols1(seq_q);
    auto kv_max = sl.unsqueeze(1).expand({batch, grid_x}).contiguous();
    auto out = torch::empty({batch, seq_q, heads_q, head_dim}, q.options());

    const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    hipError_t err = gfx906_fa_launch_paged(
        q.data_ptr<float>(), key_cache.data_ptr(), reinterpret_cast<const __half *>(value_cache.data_ptr()),
        bt.data_ptr<int32_t>(), kv_max.data_ptr<int32_t>(), out.data_ptr<float>(), nullptr,
        nullptr, 0, qo.data_ptr<int32_t>(), /*window=*/0, /*kv_start=*/nullptr, /*tile_clip=*/1,
        batch, heads_q, heads_kv, seq_q, max_blocks * block_size, head_dim, block_size, max_blocks,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        static_cast<float>(scale), stream, /*kv_split=*/1, /*pth=*/1);
    TORCH_CHECK(err == hipSuccess, "gfx906 pth FA launch failed: ", hipGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_pth", &forward_pth, "gfx906 FA over vLLM int8 per-token-head paged KV (prefill)",
          py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("scale"), py::arg("q_abs_offset"));
}
