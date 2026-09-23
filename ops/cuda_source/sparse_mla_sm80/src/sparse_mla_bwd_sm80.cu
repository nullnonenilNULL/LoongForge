// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

// SM80 (Ampere) sparse MLA backward.
//
// Given the forward's `out` and `lse`, recompute P and produce dQ / dKV.  Because
// K and V are the same tensor -- V is kv[:, :512] and the last 64 dims are the
// positional part used only by QK -- dkv receives two contributions:
//
//   delta[m,h] = sum_d dO[m,h,d] * out[m,h,d]
//   P[m,h,t]   = exp(sm_scale * q[m,h,:] . kv[idx[m,t],:] - lse[m,h])
//   dP[m,h,t]  = sum_{d<512} dO[m,h,d] * kv[idx[m,t],d]
//   dQK[m,h,t] = sm_scale * P[m,h,t] * (dP[m,h,t] - delta[m,h])
//   dQ[m,h,:]        = sum_t dQK[m,h,t] * kv[idx[m,t],:]            (576 dims)
//   dKV[idx[m,t],:] += sum_h dQK[m,h,t] * q[m,h,:]                  (576 dims)
//   dKV[idx[m,t],d] += sum_h P[m,h,t] * dO[m,h,d]      for d < 512
//
// P is exactly zero for masked positions, so dQK vanishes there and no separate
// mask is needed downstream; the scatter still has to skip invalid indices.
//
// This replaces flash_mla_bwd.flash_mla_sparse_bwd on Ampere.  It also replaces the
// TileLang fallback, which cannot run here at all: with h_q=128 it splits the heads
// in two and then requests 216 KB of dynamic shared memory against A100's 164 KB
// limit.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

#include "common.cuh"

namespace sparse_mla_sm80 {

constexpr int kDOStride = kDV + 8;
constexpr int kHStride = kHeadBlock + 8;  // for the [t][h] transposes
constexpr int kDKVSlice = 64;             // d handled per scatter step
constexpr int kDKVPerWarp = kDKVSlice / kNumWarps;

static_assert(kDQK % kDKVSlice == 0, "dkv scatter slices must tile d_qk");
static_assert(kDV % kDKVSlice == 0, "the dO chain must stop on a slice boundary");
static_assert(kDQK % (kNumWarps * 8) == 0, "dq splits d_qk across warps in 8-wide tiles");
static_assert(kDKVPerWarp % 8 == 0, "dkv n-tiles are 8 wide");
static_assert(kHeadBlock == 16, "the dkv GEMMs use a single 16-deep k-step over heads");

constexpr int kDQPerWarp = kDQK / kNumWarps;
constexpr int kDQTiles = kDQPerWarp / 8;
constexpr int kDKVTiles = kDKVPerWarp / 8;

// delta[m, h] = sum_d dO[m, h, d] * out[m, h, d].  One warp per (m, h) row.
__global__ void mla_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ out,
                                          const __nv_bfloat16* __restrict__ grad_out,
                                          float* __restrict__ delta, int rows) {
    const int row = blockIdx.x * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
    if (row >= rows) return;
    const int lane = threadIdx.x % kWarpSize;
    const size_t base = static_cast<size_t>(row) * kDV;

    float acc = 0.f;
    for (int d = lane * 8; d < kDV; d += kWarpSize * 8) {
        const uint4 o = *reinterpret_cast<const uint4*>(out + base + d);
        const uint4 g = *reinterpret_cast<const uint4*>(grad_out + base + d);
        const __nv_bfloat16* ov = reinterpret_cast<const __nv_bfloat16*>(&o);
        const __nv_bfloat16* gv = reinterpret_cast<const __nv_bfloat16*>(&g);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc += __bfloat162float(ov[i]) * __bfloat162float(gv[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) delta[row] = acc;
}

__global__ __launch_bounds__(kNumThreads) void sparse_mla_bwd_kernel(
    const __nv_bfloat16* __restrict__ q,   // [s_q, h_q, kDQK]
    const __nv_bfloat16* __restrict__ kv,  // [s_kv, 1, kDQK]
    const __nv_bfloat16* __restrict__ dO,  // [s_q, h_q, kDV]
    const int* __restrict__ indices,       // [s_q, 1, topk]
    const float* __restrict__ lse,         // [s_q, h_q]
    const float* __restrict__ delta,       // [s_q, h_q]
    __nv_bfloat16* __restrict__ dq,        // [s_q, h_q, kDQK]
    float* __restrict__ dkv,               // [s_kv, kDQK] fp32, pre-zeroed
    int s_q, int s_kv, int h_q, int topk, float sm_scale, int q_start_index_s) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKV = sQ + kHeadBlock * kQKStride;
    __nv_bfloat16* sdO = sKV + kBlockT * kQKStride;
    __nv_bfloat16* sP = sdO + kHeadBlock * kDOStride;        // [h][t]
    __nv_bfloat16* sDS = sP + kHeadBlock * kPStride;         // [h][t]
    __nv_bfloat16* sPT = sDS + kHeadBlock * kPStride;        // [t][h]
    __nv_bfloat16* sDST = sPT + kBlockT * kHStride;          // [t][h]
    int* sIdx = reinterpret_cast<int*>(sDST + kBlockT * kHStride);

    const int m = blockIdx.y;
    const int h0 = blockIdx.x * kHeadBlock;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int g = lane >> 2;
    const int p = lane & 3;
    const int causal_limit = q_start_index_s + m;

    {
        const __nv_bfloat16* qsrc = q + (static_cast<size_t>(m) * h_q + h0) * kDQK;
        for (int i = tid; i < kHeadBlock * (kDQK / 8); i += kNumThreads) {
            const int r = i / (kDQK / 8), c = (i % (kDQK / 8)) * 8;
            *reinterpret_cast<uint4*>(sQ + r * kQKStride + c) =
                *reinterpret_cast<const uint4*>(qsrc + static_cast<size_t>(r) * kDQK + c);
        }
        const __nv_bfloat16* osrc = dO + (static_cast<size_t>(m) * h_q + h0) * kDV;
        for (int i = tid; i < kHeadBlock * (kDV / 8); i += kNumThreads) {
            const int r = i / (kDV / 8), c = (i % (kDV / 8)) * 8;
            *reinterpret_cast<uint4*>(sdO + r * kDOStride + c) =
                *reinterpret_cast<const uint4*>(osrc + static_cast<size_t>(r) * kDV + c);
        }
    }

    // Per-thread rows of the head block: h0 + g and h0 + g + 8.
    float row_lse[2], row_delta[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const size_t o = static_cast<size_t>(m) * h_q + h0 + g + r * 8;
        row_lse[r] = lse[o];
        row_delta[r] = delta[o];
    }

    float dq_acc[kDQTiles][4] = {};

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
        for (int i = tid; i < kBlockT * (kDQK / 8); i += kNumThreads) {
            const int r = i / (kDQK / 8), c = (i % (kDQK / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            const int idx = sIdx[r];
            if (idx >= 0) v = *reinterpret_cast<const uint4*>(kv + static_cast<size_t>(idx) * kDQK + c);
            *reinterpret_cast<uint4*>(sKV + r * kQKStride + c) = v;
        }
        __syncthreads();

        // ---- recompute qk, and dp = dO . V; warp w owns t in [w*8, w*8+8) ----
        float qk[4] = {}, dp[4] = {};
        {
            const __nv_bfloat16* brow = sKV + (warp * 8 + g) * kQKStride;
            for (int kk = 0; kk < kDQK / 16; ++kk) {
                const int kc = kk * 16 + 2 * p;
                uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(brow + kc),
                                 *reinterpret_cast<const uint32_t*>(brow + kc + 8)};
                uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(sQ + g * kQKStride + kc),
                                 *reinterpret_cast<const uint32_t*>(sQ + (g + 8) * kQKStride + kc),
                                 *reinterpret_cast<const uint32_t*>(sQ + g * kQKStride + kc + 8),
                                 *reinterpret_cast<const uint32_t*>(sQ + (g + 8) * kQKStride + kc + 8)};
                mma_m16n8k16_bf16(qk, a, b);
                if (kk < kDV / 16) {
                    uint32_t ao[4] = {
                        *reinterpret_cast<const uint32_t*>(sdO + g * kDOStride + kc),
                        *reinterpret_cast<const uint32_t*>(sdO + (g + 8) * kDOStride + kc),
                        *reinterpret_cast<const uint32_t*>(sdO + g * kDOStride + kc + 8),
                        *reinterpret_cast<const uint32_t*>(sdO + (g + 8) * kDOStride + kc + 8)};
                    mma_m16n8k16_bf16(dp, ao, b);
                }
            }
        }

        // ---- P and dQK, staged in both [h][t] and [t][h] layouts -----------
        const int t_local[2] = {warp * 8 + 2 * p, warp * 8 + 2 * p + 1};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int hl = g + r * 8;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tl = t_local[c];
                const bool valid = (t0 + tl < topk) && (sIdx[tl] >= 0);
                // lse is +inf for query rows with no valid key, so P vanishes there.
                const float pv = valid ? __expf(qk[r * 2 + c] * sm_scale - row_lse[r]) : 0.f;
                const float ds = pv * (dp[r * 2 + c] - row_delta[r]) * sm_scale;
                sP[hl * kPStride + tl] = __float2bfloat16(pv);
                sDS[hl * kPStride + tl] = __float2bfloat16(ds);
                sPT[tl * kHStride + hl] = __float2bfloat16(pv);
                sDST[tl * kHStride + hl] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        // ---- dQ += dQK @ gathered_kv; warp w owns d in [w*kDQPerWarp, +) ---
        for (int kk = 0; kk < kBlockT / 16; ++kk) {
            const int kc = kk * 16 + 2 * p;
            uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(sDS + g * kPStride + kc),
                             *reinterpret_cast<const uint32_t*>(sDS + (g + 8) * kPStride + kc),
                             *reinterpret_cast<const uint32_t*>(sDS + g * kPStride + kc + 8),
                             *reinterpret_cast<const uint32_t*>(sDS + (g + 8) * kPStride + kc + 8)};
#pragma unroll
            for (int nt = 0; nt < kDQTiles; ++nt) {
                const int n = warp * kDQPerWarp + nt * 8 + g;
                uint32_t b[2] = {load_pair_transposed(sKV, kQKStride, kc, n),
                                 load_pair_transposed(sKV, kQKStride, kc + 8, n)};
                mma_m16n8k16_bf16(dq_acc[nt], a, b);
            }
        }

        // ---- dKV scatter, one 64-wide slice of d at a time -----------------
        // Both chains land in the same accumulator: for d < 512 a key row receives
        // dQK^T @ Q and P^T @ dO, and beyond that only the former.
        for (int n0 = 0; n0 < kDQK; n0 += kDKVSlice) {
            float acc[2][kDKVTiles][4] = {};
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r0 = mt * 16 + g;
                uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(sDST + r0 * kHStride + 2 * p),
                                 *reinterpret_cast<const uint32_t*>(sDST + (r0 + 8) * kHStride + 2 * p),
                                 *reinterpret_cast<const uint32_t*>(sDST + r0 * kHStride + 2 * p + 8),
                                 *reinterpret_cast<const uint32_t*>(sDST + (r0 + 8) * kHStride + 2 * p + 8)};
                uint32_t ap[4] = {*reinterpret_cast<const uint32_t*>(sPT + r0 * kHStride + 2 * p),
                                  *reinterpret_cast<const uint32_t*>(sPT + (r0 + 8) * kHStride + 2 * p),
                                  *reinterpret_cast<const uint32_t*>(sPT + r0 * kHStride + 2 * p + 8),
                                  *reinterpret_cast<const uint32_t*>(sPT + (r0 + 8) * kHStride + 2 * p + 8)};
#pragma unroll
                for (int nt = 0; nt < kDKVTiles; ++nt) {
                    const int n = n0 + warp * kDKVPerWarp + nt * 8 + g;
                    uint32_t b[2] = {load_pair_transposed(sQ, kQKStride, 2 * p, n),
                                     load_pair_transposed(sQ, kQKStride, 2 * p + 8, n)};
                    mma_m16n8k16_bf16(acc[mt][nt], a, b);
                    if (n0 < kDV) {
                        uint32_t bo[2] = {load_pair_transposed(sdO, kDOStride, 2 * p, n),
                                          load_pair_transposed(sdO, kDOStride, 2 * p + 8, n)};
                        mma_m16n8k16_bf16(acc[mt][nt], ap, bo);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int tl = mt * 16 + g + r * 8;
                    const int idx = sIdx[tl];
                    if (idx < 0) continue;
#pragma unroll
                    for (int nt = 0; nt < kDKVTiles; ++nt)
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int d = n0 + warp * kDKVPerWarp + nt * 8 + 2 * p + c;
                            atomicAdd(dkv + static_cast<size_t>(idx) * kDQK + d,
                                      acc[mt][nt][r * 2 + c]);
                        }
                }
        }
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int h = h0 + g + r * 8;
#pragma unroll
        for (int nt = 0; nt < kDQTiles; ++nt)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int d = warp * kDQPerWarp + nt * 8 + 2 * p + c;
                dq[(static_cast<size_t>(m) * h_q + h) * kDQK + d] =
                    __float2bfloat16(dq_acc[nt][r * 2 + c]);
            }
    }
}

}  // namespace sparse_mla_sm80

std::vector<at::Tensor> sparse_prefill_bwd_sm80(const at::Tensor& q, const at::Tensor& kv,
                                                const at::Tensor& out, const at::Tensor& grad_out,
                                                const at::Tensor& indices, const at::Tensor& lse,
                                                double sm_scale, int64_t q_start_index_s) {
    using namespace sparse_mla_sm80;
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && out.is_cuda() && grad_out.is_cuda());
    TORCH_CHECK(q.dtype() == at::kBFloat16 && kv.dtype() == at::kBFloat16);
    TORCH_CHECK(out.dtype() == at::kBFloat16 && grad_out.dtype() == at::kBFloat16);
    TORCH_CHECK(indices.dtype() == at::kInt && lse.dtype() == at::kFloat);
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && out.dim() == 3 && grad_out.dim() == 3);
    TORCH_CHECK(q.size(2) == kDQK && kv.size(2) == kDQK);
    TORCH_CHECK(out.size(2) == kDV && grad_out.size(2) == kDV);
    TORCH_CHECK(kv.size(1) == 1, "h_kv must be 1, got ", kv.size(1));
    TORCH_CHECK(q_start_index_s >= 0);

    const int s_q = q.size(0);
    const int h_q = q.size(1);
    const int s_kv = kv.size(0);
    const int topk = indices.size(2);
    TORCH_CHECK(h_q % kHeadBlock == 0, "h_q must be a multiple of ", kHeadBlock, ", got ", h_q);
    TORCH_CHECK(out.size(0) == s_q && out.size(1) == h_q);
    TORCH_CHECK(lse.size(0) == s_q && lse.size(1) == h_q);

    const at::cuda::CUDAGuard guard(q.device());
    auto qc = q.contiguous();
    auto kvc = kv.contiguous();
    auto goc = grad_out.contiguous();
    auto outc = out.contiguous();
    auto idxc = indices.contiguous();
    auto lsec = lse.contiguous();

    auto delta = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));
    constexpr int kPreWarps = 4;
    const int rows = s_q * h_q;
    mla_bwd_preprocess_kernel<<<(rows + kPreWarps - 1) / kPreWarps, kPreWarps * kWarpSize, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(outc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(goc.data_ptr()), delta.data_ptr<float>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto dq = at::empty_like(qc);
    // fp32 because the scatter is an atomicAdd: many (query, top-k) pairs select the
    // same key row, and bf16 atomics would both lose precision and need a CAS loop.
    auto dkv_f32 = at::zeros({s_kv, kDQK}, q.options().dtype(at::kFloat));

    const size_t smem = (kHeadBlock * kQKStride + kBlockT * kQKStride + kHeadBlock * kDOStride +
                         2 * kHeadBlock * kPStride + 2 * kBlockT * kHStride) *
                            sizeof(__nv_bfloat16) +
                        kBlockT * sizeof(int);
    static bool attr_set = false;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_bwd_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        attr_set = true;
    }

    const dim3 grid(h_q / kHeadBlock, s_q);
    sparse_mla_bwd_kernel<<<grid, kNumThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kvc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(goc.data_ptr()), idxc.data_ptr<int>(),
        lsec.data_ptr<float>(), delta.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr()), dkv_f32.data_ptr<float>(), s_q, s_kv, h_q,
        topk, static_cast<float>(sm_scale), static_cast<int>(q_start_index_s));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dq, dkv_f32.to(at::kBFloat16).view({s_kv, 1, kDQK})};
}