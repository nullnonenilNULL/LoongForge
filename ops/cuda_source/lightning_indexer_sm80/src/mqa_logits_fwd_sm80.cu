// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>
#include <cmath>

#include "common.cuh"

namespace indexer_sm80 {

// Heads are staged into shared memory in blocks so that a CTA fits twice per SM
// (73 KB each).  Staging all 32 heads at once costs 103 KB and halves occupancy.
constexpr int kHeadBlock = 16;
static_assert(kNumHeads % kHeadBlock == 0, "kNumHeads must be a multiple of kHeadBlock");
static_assert(kBlockN % kNChunk == 0, "kBlockN must be a multiple of kNChunk");
static_assert(kHeadBlock == 16, "epilogue reduction assumes exactly one 16-row mma tile");

constexpr int kSmemK = kBlockN * kSmemStride;               // bf16 elements
constexpr int kSmemQ = kBlockQ * kHeadBlock * kSmemStride;  // bf16 elements

// Cooperative bf16 row copy: `rows` rows of kHeadDim elements from a global
// buffer with row stride `src_stride` into a padded shared buffer.  Rows at or
// past `row_limit` are zero filled so masked-out tiles stay finite.
__device__ __forceinline__ void load_rows_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                               int rows, int src_stride, int row_limit,
                                               int tid, int nthreads) {
    constexpr int kVec = 8;  // 16-byte vector loads
    constexpr int kVecPerRow = kHeadDim / kVec;
    const int total = rows * kVecPerRow;
    for (int i = tid; i < total; i += nthreads) {
        const int r = i / kVecPerRow;
        const int c = (i % kVecPerRow) * kVec;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < row_limit) {
            v = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * src_stride + c);
        }
        *reinterpret_cast<uint4*>(dst + static_cast<size_t>(r) * kSmemStride + c) = v;
    }
}

// Stage kHeadBlock heads of kBlockQ query tokens.  q is [s_q, kNumHeads,
// kHeadDim], so consecutive shared rows within a token are contiguous but the
// step between tokens is kNumHeads rows, not kHeadBlock -- a single uniform
// src_stride cannot express this.
__device__ __forceinline__ void load_q_block(__nv_bfloat16* dst, const __nv_bfloat16* q_base,
                                             int valid_tokens, int tid, int nthreads) {
    constexpr int kVec = 8;
    constexpr int kVecPerRow = kHeadDim / kVec;
    constexpr int total = kBlockQ * kHeadBlock * kVecPerRow;
    for (int i = tid; i < total; i += nthreads) {
        const int r = i / kVecPerRow;
        const int c = (i % kVecPerRow) * kVec;
        const int token = r / kHeadBlock;
        const int head_in_block = r % kHeadBlock;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (token < valid_tokens) {
            const size_t off =
                (static_cast<size_t>(token) * kNumHeads + head_in_block) * kHeadDim + c;
            v = *reinterpret_cast<const uint4*>(q_base + off);
        }
        *reinterpret_cast<uint4*>(dst + static_cast<size_t>(r) * kSmemStride + c) = v;
    }
}


}  // namespace indexer_sm80

namespace indexer_sm80 {

__global__ __launch_bounds__(kNumThreads) void mqa_logits_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,  // [s_q, kNumHeads, kHeadDim]
    const __nv_bfloat16* __restrict__ k,  // [s_kv, kHeadDim]
    const float* __restrict__ weights,    // [s_q, kNumHeads]
    const int* __restrict__ ks,           // [s_q]
    const int* __restrict__ ke,           // [s_q]
    float* __restrict__ logits,           // [s_q, s_kv]
    int s_q, int s_kv) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sQ = sK + kSmemK;
    float* sW = reinterpret_cast<float*>(sQ + kSmemQ);
    float* sOut = sW + kBlockQ * kNumHeads;
    int* sKs = reinterpret_cast<int*>(sOut + kBlockQ * kBlockN);
    int* sKe = sKs + kBlockQ;

    const int q0 = blockIdx.x * kBlockQ;
    const int n0 = blockIdx.y * kBlockN;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;   // one warp per query token
    const int lane = tid % kWarpSize;
    const int g = lane >> 2;            // mma row / n-tile selector, 0..7
    const int p = lane & 3;             // mma column pair, 0..3

    if (tid < kBlockQ) {
        const int qi = q0 + tid;
        sKs[tid] = qi < s_q ? ks[qi] : 0;
        sKe[tid] = qi < s_q ? ke[qi] : 0;
    }
    __syncthreads();

    // Causal early-out: if no query in this tile can reach n0, the whole tile is
    // masked.  logits is uninitialised, so still write -inf before leaving.
    int ke_max = 0;
    int ks_min = INT_MAX;
    for (int i = 0; i < kBlockQ; ++i) {
        ke_max = max(ke_max, sKe[i]);
        ks_min = min(ks_min, sKs[i]);
    }
    if (n0 >= ke_max || n0 + kBlockN <= ks_min) {
        for (int i = tid; i < kBlockQ * kBlockN; i += kNumThreads) {
            const int qi = q0 + i / kBlockN;
            const int ni = n0 + i % kBlockN;
            if (qi < s_q && ni < s_kv) logits[static_cast<size_t>(qi) * s_kv + ni] = -INFINITY;
        }
        return;
    }

    load_rows_bf16(sK, k + static_cast<size_t>(n0) * kHeadDim, kBlockN, kHeadDim,
                   min(kBlockN, s_kv - n0), tid, kNumThreads);
    for (int i = tid; i < kBlockQ * kNumHeads; i += kNumThreads) {
        const int qi = q0 + i / kNumHeads;
        sW[i] = qi < s_q ? weights[static_cast<size_t>(qi) * kNumHeads + i % kNumHeads] : 0.f;
    }
    for (int i = tid; i < kBlockQ * kBlockN; i += kNumThreads) sOut[i] = 0.f;
    __syncthreads();

    for (int hb = 0; hb < kNumHeads / kHeadBlock; ++hb) {
        load_q_block(sQ, q + (static_cast<size_t>(q0) * kNumHeads + hb * kHeadBlock) * kHeadDim,
                     max(0, min(kBlockQ, s_q - q0)), tid, kNumThreads);
        __syncthreads();
        const __nv_bfloat16* sQw = sQ + static_cast<size_t>(warp) * kHeadBlock * kSmemStride;
        const float* sWw = sW + warp * kNumHeads + hb * kHeadBlock;

        for (int nc = 0; nc < kBlockN / kNChunk; ++nc) {
            float acc[kNChunk / 8][4] = {};
            for (int kk = 0; kk < kHeadDim / 16; ++kk) {
                uint32_t a[4];
                const int kcol = kk * 16 + 2 * p;
                a[0] = *reinterpret_cast<const uint32_t*>(sQw + g * kSmemStride + kcol);
                a[1] = *reinterpret_cast<const uint32_t*>(sQw + (g + 8) * kSmemStride + kcol);
                a[2] = *reinterpret_cast<const uint32_t*>(sQw + g * kSmemStride + kcol + 8);
                a[3] = *reinterpret_cast<const uint32_t*>(sQw + (g + 8) * kSmemStride + kcol + 8);
#pragma unroll
                for (int nt = 0; nt < kNChunk / 8; ++nt) {
                    const __nv_bfloat16* srow = sK + (nc * kNChunk + nt * 8 + g) * kSmemStride;
                    uint32_t b[2];
                    b[0] = *reinterpret_cast<const uint32_t*>(srow + kcol);
                    b[1] = *reinterpret_cast<const uint32_t*>(srow + kcol + 8);
                    mma_m16n8k16_bf16(acc[nt], a, b);
                }
            }
            // Epilogue: relu, weight by head, reduce over the 16 head rows of this
            // head block (rows g and g+8 live in-thread, the rest across lanes
            // differing in bits 2..4 of the lane id).
#pragma unroll
            for (int nt = 0; nt < kNChunk / 8; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float v = sWw[g] * fmaxf(acc[nt][j], 0.f) +
                              sWw[g + 8] * fmaxf(acc[nt][2 + j], 0.f);
                    v += __shfl_xor_sync(0xffffffffu, v, 4);
                    v += __shfl_xor_sync(0xffffffffu, v, 8);
                    v += __shfl_xor_sync(0xffffffffu, v, 16);
                    if (g == 0) sOut[warp * kBlockN + nc * kNChunk + nt * 8 + 2 * p + j] += v;
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < kBlockQ * kBlockN; i += kNumThreads) {
        const int qi = q0 + i / kBlockN;
        const int ni = n0 + i % kBlockN;
        if (qi >= s_q || ni >= s_kv) continue;
        const bool keep = ni >= sKs[i / kBlockN] && ni < sKe[i / kBlockN];
        logits[static_cast<size_t>(qi) * s_kv + ni] = keep ? sOut[i] : -INFINITY;
    }
}

}  // namespace indexer_sm80

at::Tensor bf16_mqa_logits(const at::Tensor& q, const at::Tensor& k, const at::Tensor& weights,
                           const at::Tensor& cu_seqlen_ks, const at::Tensor& cu_seqlen_ke) {
    using namespace indexer_sm80;
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && weights.is_cuda());
    TORCH_CHECK(q.dtype() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == at::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(weights.dtype() == at::kFloat, "weights must be float32");
    TORCH_CHECK(cu_seqlen_ks.dtype() == at::kInt && cu_seqlen_ke.dtype() == at::kInt);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 2 && weights.dim() == 2);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && weights.is_contiguous());

    const int s_q = q.size(0);
    const int s_kv = k.size(0);
    TORCH_CHECK(q.size(1) == kNumHeads, "expected ", kNumHeads, " heads, got ", q.size(1));
    TORCH_CHECK(q.size(2) == kHeadDim, "expected head_dim ", kHeadDim, ", got ", q.size(2));
    TORCH_CHECK(k.size(1) == kHeadDim);
    TORCH_CHECK(weights.size(0) == s_q && weights.size(1) == kNumHeads);
    TORCH_CHECK(cu_seqlen_ks.size(0) == s_q && cu_seqlen_ke.size(0) == s_q);

    const at::cuda::CUDAGuard guard(q.device());
    auto logits = at::empty({s_q, s_kv}, q.options().dtype(at::kFloat));

    const size_t smem = (kSmemK + kSmemQ) * sizeof(__nv_bfloat16) +
                        (kBlockQ * kNumHeads + kBlockQ * kBlockN) * sizeof(float) +
                        2 * kBlockQ * sizeof(int);
    static bool attr_set = false;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mqa_logits_fwd_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        attr_set = true;
    }

    const dim3 grid((s_q + kBlockQ - 1) / kBlockQ, (s_kv + kBlockN - 1) / kBlockN);
    mqa_logits_fwd_kernel<<<grid, kNumThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), weights.data_ptr<float>(),
        cu_seqlen_ks.data_ptr<int>(), cu_seqlen_ke.data_ptr<int>(), logits.data_ptr<float>(), s_q,
        s_kv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return logits;
}