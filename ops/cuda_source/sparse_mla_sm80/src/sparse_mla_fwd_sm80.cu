// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

// SM80 (Ampere) sparse MLA forward.
//
// out[m, h, :] = softmax_t( sm_scale * q[m, h, :] . kv[idx[m, t], :] ) @ v[idx[m, t], :]
//
// with the top-k gather `idx`, causal masking against `q_start_index_s + m`, and
// `p_out` = the *unscaled* masked qk scores (the indexer KL loss consumes those;
// triton_attn_dist applies sm_scale itself).  lse is natural log, +inf for query
// tokens with no valid key.
//
// This replaces flash_mla_fwd.flash_mla_sparse_fwd on Ampere, which is SM100-only:
// the SM100 kernel keeps its accumulators in TMEM and issues 2-SM cluster
// UMMA, neither of which exists here.  Structure is a standard online-softmax
// flash-attention loop built on mma.sync.m16n8k16.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <vector>

#include "common.cuh"

namespace sparse_mla_sm80 {

constexpr int kSmemQ = kHeadBlock * kQKStride;
constexpr int kSmemKV = kBlockT * kQKStride;
constexpr int kSmemP = kHeadBlock * kPStride;
constexpr int kDVPerWarp = kDV / kNumWarps;
constexpr int kPVTiles = kDVPerWarp / 8;

__device__ __forceinline__ void load_rows(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                          int rows, int cols, int stride_src, int tid,
                                          int nthreads) {
    constexpr int kVec = 8;
    const int vec_per_row = cols / kVec;
    for (int i = tid; i < rows * vec_per_row; i += nthreads) {
        const int r = i / vec_per_row;
        const int c = (i % vec_per_row) * kVec;
        *reinterpret_cast<uint4*>(dst + r * kQKStride + c) =
            *reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * stride_src + c);
    }
}

__global__ __launch_bounds__(kNumThreads) void sparse_mla_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,   // [s_q, h_q, kDQK]
    const __nv_bfloat16* __restrict__ kv,  // [s_kv, 1, kDQK]
    const int* __restrict__ indices,       // [s_q, 1, topk]
    __nv_bfloat16* __restrict__ out,       // [s_q, h_q, kDV]
    float* __restrict__ max_logits,        // [s_q, h_q]
    float* __restrict__ lse,               // [s_q, h_q]
    float* __restrict__ p_out,             // [s_q, h_q, topk] or nullptr
    int s_q, int s_kv, int h_q, int topk, float sm_scale, int q_start_index_s) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKV = sQ + kSmemQ;
    __nv_bfloat16* sP = sKV + kSmemKV;
    float* sRed = reinterpret_cast<float*>(sP + kSmemP);  // [kNumWarps][kHeadBlock]
    float* sMax = sRed + kNumWarps * kHeadBlock;          // [kHeadBlock]
    int* sIdx = reinterpret_cast<int*>(sMax + kHeadBlock);  // [kBlockT]

    const int m = blockIdx.y;
    const int h0 = blockIdx.x * kHeadBlock;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int g = lane >> 2;
    const int p = lane & 3;
    const int causal_limit = q_start_index_s + m;

    load_rows(sQ, q + (static_cast<size_t>(m) * h_q + h0) * kDQK, kHeadBlock, kDQK, kDQK, tid,
              kNumThreads);

    // Online-softmax state, per head row held by this thread (rows g and g + 8).
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.f, 0.f};
    float o_acc[kPVTiles][4] = {};

    for (int t0 = 0; t0 < topk; t0 += kBlockT) {
        __syncthreads();
        for (int i = tid; i < kBlockT; i += kNumThreads) {
            const int t = t0 + i;
            int idx = -1;
            if (t < topk) {
                idx = indices[static_cast<size_t>(m) * topk + t];
                if (idx < 0 || idx >= s_kv || idx > causal_limit) idx = -1;
            }
            sIdx[i] = idx;
        }
        __syncthreads();
        constexpr int kVecPerRow = kDQK / 8;
        for (int i = tid; i < kBlockT * kVecPerRow; i += kNumThreads) {
            const int r = i / kVecPerRow;
            const int c = (i % kVecPerRow) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            const int idx = sIdx[r];
            if (idx >= 0) v = *reinterpret_cast<const uint4*>(kv + static_cast<size_t>(idx) * kDQK + c);
            *reinterpret_cast<uint4*>(sKV + r * kQKStride + c) = v;
        }
        __syncthreads();

        // ---- QK: warp w owns t in [w*8, w*8+8) over all kHeadBlock heads ----
        float qk[4] = {};
        {
            const __nv_bfloat16* brow = sKV + (warp * 8 + g) * kQKStride;
            for (int kk = 0; kk < kDQK / 16; ++kk) {
                const int kc = kk * 16 + 2 * p;
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(brow + kc);
                b[1] = *reinterpret_cast<const uint32_t*>(brow + kc + 8);
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t*>(sQ + g * kQKStride + kc);
                a[1] = *reinterpret_cast<const uint32_t*>(sQ + (g + 8) * kQKStride + kc);
                a[2] = *reinterpret_cast<const uint32_t*>(sQ + g * kQKStride + kc + 8);
                a[3] = *reinterpret_cast<const uint32_t*>(sQ + (g + 8) * kQKStride + kc + 8);
                mma_m16n8k16_bf16(qk, a, b);
            }
        }

        // qk holds rows {g, g+8} x cols {warp*8+2p, +1}.
        const int t_local[2] = {warp * 8 + 2 * p, warp * 8 + 2 * p + 1};
        bool valid[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) valid[c] = (t0 + t_local[c] < topk) && (sIdx[t_local[c]] >= 0);

        if (p_out != nullptr) {
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int t = t0 + t_local[c];
                    if (t >= topk) continue;
                    const int h = h0 + g + r * 8;
                    p_out[(static_cast<size_t>(m) * h_q + h) * topk + t] =
                        valid[c] ? qk[r * 2 + c] : -INFINITY;
                }
        }

        // ---- running max across the whole kBlockT tile --------------------
        float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                if (valid[c]) tile_max[r] = fmaxf(tile_max[r], qk[r * 2 + c] * sm_scale);
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 1));
            tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffu, tile_max[r], 2));
            if (p == 0) sRed[warp * kHeadBlock + g + r * 8] = tile_max[r];
        }
        __syncthreads();
        for (int i = tid; i < kHeadBlock; i += kNumThreads) {
            float v = -INFINITY;
#pragma unroll
            for (int w = 0; w < kNumWarps; ++w) v = fmaxf(v, sRed[w * kHeadBlock + i]);
            sMax[i] = v;
        }
        __syncthreads();

        // ---- rescale, write p, accumulate the denominator ------------------
        float rescale[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int hl = g + r * 8;
            const float new_max = fmaxf(row_max[r], sMax[hl]);
            rescale[r] = (row_max[r] == new_max) ? 1.f
                         : (isinf(new_max) ? 0.f : __expf(row_max[r] - new_max));
            if (isinf(row_max[r]) && row_max[r] < 0.f) rescale[r] = 0.f;
            row_sum[r] *= rescale[r];
            row_max[r] = new_max;

            float local_sum = 0.f;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                float pv = 0.f;
                if (valid[c] && !isinf(new_max)) {
                    pv = __expf(qk[r * 2 + c] * sm_scale - new_max);
                    local_sum += pv;
                }
                sP[hl * kPStride + t_local[c]] = __float2bfloat16(pv);
            }
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, 2);
            if (p == 0) sRed[warp * kHeadBlock + hl] = local_sum;
        }
        __syncthreads();
        {
            float tile_sum[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                tile_sum[r] = 0.f;
#pragma unroll
                for (int w = 0; w < kNumWarps; ++w) tile_sum[r] += sRed[w * kHeadBlock + g + r * 8];
                row_sum[r] += tile_sum[r];
            }
        }

        // ---- PV: warp w owns d in [w*kDVPerWarp, +kDVPerWarp) --------------
#pragma unroll
        for (int nt = 0; nt < kPVTiles; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) o_acc[nt][i] *= rescale[i >> 1];

        for (int kk = 0; kk < kBlockT / 16; ++kk) {
            const int kc = kk * 16 + 2 * p;
            uint32_t a[4];
            a[0] = *reinterpret_cast<const uint32_t*>(sP + g * kPStride + kc);
            a[1] = *reinterpret_cast<const uint32_t*>(sP + (g + 8) * kPStride + kc);
            a[2] = *reinterpret_cast<const uint32_t*>(sP + g * kPStride + kc + 8);
            a[3] = *reinterpret_cast<const uint32_t*>(sP + (g + 8) * kPStride + kc + 8);
#pragma unroll
            for (int nt = 0; nt < kPVTiles; ++nt) {
                const int n = warp * kDVPerWarp + nt * 8 + g;
                uint32_t b[2];
                b[0] = load_pair_transposed(sKV, kQKStride, kc, n);
                b[1] = load_pair_transposed(sKV, kQKStride, kc + 8, n);
                mma_m16n8k16_bf16(o_acc[nt], a, b);
            }
        }
    }

    // ---- epilogue -------------------------------------------------------
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int h = h0 + g + r * 8;
        const bool lonely = (row_sum[r] == 0.f);
        const float inv = lonely ? 0.f : 1.f / row_sum[r];
#pragma unroll
        for (int nt = 0; nt < kPVTiles; ++nt)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int d = warp * kDVPerWarp + nt * 8 + 2 * p + c;
                out[(static_cast<size_t>(m) * h_q + h) * kDV + d] =
                    __float2bfloat16(o_acc[nt][r * 2 + c] * inv);
            }
        if (warp == 0 && p == 0) {
            max_logits[static_cast<size_t>(m) * h_q + h] = row_max[r];
            lse[static_cast<size_t>(m) * h_q + h] =
                lonely ? INFINITY : (__logf(row_sum[r]) + row_max[r]);
        }
    }
}

}  // namespace sparse_mla_sm80

std::vector<at::Tensor> sparse_prefill_fwd_sm80(const at::Tensor& q, const at::Tensor& kv,
                                                const at::Tensor& indices, double sm_scale,
                                                int64_t d_v, int64_t q_start_index_s,
                                                bool write_p_out) {
    using namespace sparse_mla_sm80;
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda());
    TORCH_CHECK(q.dtype() == at::kBFloat16 && kv.dtype() == at::kBFloat16);
    TORCH_CHECK(indices.dtype() == at::kInt);
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3);
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous());
    TORCH_CHECK(d_v == kDV, "sm80 sparse MLA supports d_v=", kDV, " only, got ", d_v);
    TORCH_CHECK(q.size(2) == kDQK, "expected d_qk=", kDQK, ", got ", q.size(2));
    TORCH_CHECK(kv.size(2) == kDQK);
    TORCH_CHECK(kv.size(1) == 1, "h_kv must be 1, got ", kv.size(1));
    TORCH_CHECK(indices.size(1) == 1);
    TORCH_CHECK(q_start_index_s >= 0);

    const int s_q = q.size(0);
    const int h_q = q.size(1);
    const int s_kv = kv.size(0);
    const int topk = indices.size(2);
    TORCH_CHECK(indices.size(0) == s_q);
    TORCH_CHECK(h_q % kHeadBlock == 0, "h_q must be a multiple of ", kHeadBlock, ", got ", h_q);

    const at::cuda::CUDAGuard guard(q.device());
    auto fopts = q.options().dtype(at::kFloat);
    auto out = at::empty({s_q, h_q, kDV}, q.options());
    auto max_logits = at::empty({s_q, h_q}, fopts);
    auto lse = at::empty({s_q, h_q}, fopts);
    at::Tensor p_out;
    if (write_p_out) p_out = at::empty({s_q, h_q, topk}, fopts);

    const size_t smem = (kSmemQ + kSmemKV + kSmemP) * sizeof(__nv_bfloat16) +
                        (kNumWarps * kHeadBlock + kHeadBlock) * sizeof(float) +
                        kBlockT * sizeof(int);
    static bool attr_set = false;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_fwd_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        attr_set = true;
    }

    // Head blocks come first in the grid so the 128 / kHeadBlock CTAs that share a
    // query token -- and therefore share the whole gathered kv -- are launched
    // adjacently and hit in L2.  With the token on x they are h_q/kHeadBlock apart
    // in launch order and each re-reads the gather from HBM.
    const dim3 grid(h_q / kHeadBlock, s_q);
    sparse_mla_fwd_kernel<<<grid, kNumThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), max_logits.data_ptr<float>(),
        lse.data_ptr<float>(), write_p_out ? p_out.data_ptr<float>() : nullptr, s_q, s_kv, h_q,
        topk, static_cast<float>(sm_scale), static_cast<int>(q_start_index_s));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (write_p_out) return {out, max_logits, lse, p_out};
    return {out, max_logits, lse};
}