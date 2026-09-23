// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

// SM80 (Ampere) BF16 lightning-indexer backward.
//
// Forward recap:  logits[m, n] = sum_h w[m, h] * relu(q[m, h, :] . k[n, :])
// Backward, restricted to the top-k positions selected in the forward:
//
//   s[h, j]  = q[m, h, :] . k[idx[j], :]
//   dw[h]    = sum_j gl[j] * relu(s[h, j])
//   ds[h, j] = gl[j] * w[h] * (s[h, j] > 0)
//   dq[h, :] = sum_j ds[h, j] * k[idx[j], :]
//   dk[idx[j], :] += sum_h ds[h, j] * q[m, h, :]
//
// so three GEMMs per j-block plus a sparse gather.  One CTA owns one query token
// and loops over j-blocks, which keeps the dq accumulator in shared memory and
// writes it once; splitting j across CTAs would need atomics on dq as well.
//
// mma.sync is `.row.col`, i.e. B must be n-major with k contiguous.  Two of the
// three GEMMs want the *transpose* of a buffer that is already resident, so
// those fragments are assembled from pairs of 16-bit shared loads instead of
// materialising a transposed copy: that removes ~29 KB of shared memory and, more
// importantly, the strided transpose writes (8192 elements per j-block).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "common.cuh"

namespace indexer_sm80 {

constexpr int kBlockJ = 64;   // top-k positions per iteration
constexpr int kBwdWarps = 8;
constexpr int kBwdThreads = kBwdWarps * kWarpSize;
constexpr int kDsStride = kBlockJ + 8;  // padded, same rationale as kSmemStride

static_assert(kBlockJ == kBwdWarps * 8, "GEMM1 gives each warp one n-tile of 8");
static_assert(kHeadDim == kBwdWarps * 16, "GEMM2/3 give each warp two n-tiles of 8");
static_assert(kNumHeads == 32, "epilogues assume two 16-row mma tiles over heads");

// Assemble one bf16 pair from a column of a row-major shared buffer, i.e. read
// src[k0][col] and src[k0 + 1][col] where the buffer's fast axis is `col`.
__device__ __forceinline__ uint32_t load_pair_transposed(const __nv_bfloat16* src, int stride,
                                                         int k0, int col) {
    const uint16_t lo = reinterpret_cast<const uint16_t*>(src)[k0 * stride + col];
    const uint16_t hi = reinterpret_cast<const uint16_t*>(src)[(k0 + 1) * stride + col];
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

}  // namespace indexer_sm80

namespace indexer_sm80 {

constexpr int kSmemQBwd = kNumHeads * kSmemStride;
constexpr int kSmemKgBwd = kBlockJ * kSmemStride;
constexpr int kSmemDs = kNumHeads * kDsStride;

__global__ __launch_bounds__(kBwdThreads) void mqa_logits_bwd_kernel(
    const float* __restrict__ grad_logits,  // [s_q, topk]
    const __nv_bfloat16* __restrict__ q,    // [s_q, kNumHeads, kHeadDim]
    const __nv_bfloat16* __restrict__ k,    // [s_kv, kHeadDim]
    const float* __restrict__ weights,      // [s_q, kNumHeads]
    const int* __restrict__ ks,             // [s_q]
    const int* __restrict__ ke,             // [s_q]
    const int* __restrict__ topk_indices,   // [s_q, topk]
    float* __restrict__ grad_q,             // [s_q, kNumHeads, kHeadDim]
    float* __restrict__ grad_kv,            // [s_kv, kHeadDim]
    float* __restrict__ grad_weights,       // [s_q, kNumHeads]
    int s_q, int s_kv, int topk) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKg = sQ + kSmemQBwd;
    __nv_bfloat16* sDs = sKg + kSmemKgBwd;
    float* sdQ = reinterpret_cast<float*>(sDs + kSmemDs);
    float* sWsc = sdQ + kNumHeads * kHeadDim;
    float* sGw = sWsc + kNumHeads;
    float* sGl = sGw + kNumHeads;
    int* sIdx = reinterpret_cast<int*>(sGl + kBlockJ);

    const int m = blockIdx.x;
    if (m >= s_q) return;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int g = lane >> 2;
    const int p = lane & 3;

    const int k_start = ks[m];
    const int k_end = ke[m];

    // Stage q for this token and zero the accumulators.
    for (int i = tid; i < kNumHeads * (kHeadDim / 8); i += kBwdThreads) {
        const int h = i / (kHeadDim / 8);
        const int c = (i % (kHeadDim / 8)) * 8;
        *reinterpret_cast<uint4*>(sQ + h * kSmemStride + c) =
            *reinterpret_cast<const uint4*>(q + (static_cast<size_t>(m) * kNumHeads + h) * kHeadDim + c);
    }
    for (int i = tid; i < kNumHeads; i += kBwdThreads) {
        sWsc[i] = weights[static_cast<size_t>(m) * kNumHeads + i];
        sGw[i] = 0.f;
    }
    for (int i = tid; i < kNumHeads * kHeadDim; i += kBwdThreads) sdQ[i] = 0.f;
    __syncthreads();

    for (int j0 = 0; j0 < topk; j0 += kBlockJ) {
        // ---- sparse gather of the kv rows for this j-block -----------------
        for (int i = tid; i < kBlockJ; i += kBwdThreads) {
            const int j = j0 + i;
            int idx = -1;
            if (j < topk) {
                idx = topk_indices[static_cast<size_t>(m) * topk + j];
                if (idx < k_start || idx >= k_end || idx >= s_kv) idx = -1;
            }
            sIdx[i] = idx;
            sGl[i] = (idx >= 0) ? grad_logits[static_cast<size_t>(m) * topk + j] : 0.f;
        }
        __syncthreads();
        for (int i = tid; i < kBlockJ * (kHeadDim / 8); i += kBwdThreads) {
            const int jj = i / (kHeadDim / 8);
            const int c = (i % (kHeadDim / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            const int idx = sIdx[jj];
            if (idx >= 0) {
                v = *reinterpret_cast<const uint4*>(k + static_cast<size_t>(idx) * kHeadDim + c);
            }
            *reinterpret_cast<uint4*>(sKg + jj * kSmemStride + c) = v;
        }
        __syncthreads();

        // ---- GEMM1: s[h, j] = q[h, :] . kg[j, :] ---------------------------
        // A = sQ (rows h, k = head dim), B = sKg (n = j, k = head dim).
        float s_acc[2][4] = {};
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
            const int kcol = kk * 16 + 2 * p;
            uint32_t b[2];
            const __nv_bfloat16* brow = sKg + (warp * 8 + g) * kSmemStride;
            b[0] = *reinterpret_cast<const uint32_t*>(brow + kcol);
            b[1] = *reinterpret_cast<const uint32_t*>(brow + kcol + 8);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const __nv_bfloat16* arow = sQ + (mt * 16 + g) * kSmemStride;
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t*>(arow + kcol);
                a[1] = *reinterpret_cast<const uint32_t*>(arow + 8 * kSmemStride + kcol);
                a[2] = *reinterpret_cast<const uint32_t*>(arow + kcol + 8);
                a[3] = *reinterpret_cast<const uint32_t*>(arow + 8 * kSmemStride + kcol + 8);
                mma_m16n8k16_bf16(s_acc[mt], a, b);
            }
        }

        // ---- epilogue: ds, dw ---------------------------------------------
        // Thread holds rows {mt*16+g, mt*16+g+8} and cols {warp*8+2p, +1}.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {           // r = 0 -> row g, r = 1 -> row g+8
                const int h = mt * 16 + g + r * 8;
                float dw = 0.f;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int jj = warp * 8 + 2 * p + c;
                    const float s = s_acc[mt][r * 2 + c];
                    const float gl = sGl[jj];
                    dw += gl * fmaxf(s, 0.f);
                    const float ds = (s > 0.f) ? gl * sWsc[h] : 0.f;
                    sDs[h * kDsStride + jj] = __float2bfloat16(ds);
                }
                dw += __shfl_xor_sync(0xffffffffu, dw, 1);
                dw += __shfl_xor_sync(0xffffffffu, dw, 2);
                if (p == 0) atomicAdd(&sGw[h], dw);
            }
        }
        __syncthreads();

        // ---- GEMM2: dq[h, d] += sum_j ds[h, j] * kg[j, d] ------------------
        // A = sDs (rows h, k = j).  B wants [n = d][k = j], which is sKg
        // transposed, hence the paired 16-bit loads.
        {
            float acc[2][2][4] = {};
            for (int kk = 0; kk < kBlockJ / 16; ++kk) {
                const int kcol = kk * 16 + 2 * p;
                uint32_t a[2][4];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const __nv_bfloat16* arow = sDs + (mt * 16 + g) * kDsStride;
                    a[mt][0] = *reinterpret_cast<const uint32_t*>(arow + kcol);
                    a[mt][1] = *reinterpret_cast<const uint32_t*>(arow + 8 * kDsStride + kcol);
                    a[mt][2] = *reinterpret_cast<const uint32_t*>(arow + kcol + 8);
                    a[mt][3] = *reinterpret_cast<const uint32_t*>(arow + 8 * kDsStride + kcol + 8);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int n = warp * 16 + nt * 8 + g;
                    uint32_t b[2];
                    b[0] = load_pair_transposed(sKg, kSmemStride, kk * 16 + 2 * p, n);
                    b[1] = load_pair_transposed(sKg, kSmemStride, kk * 16 + 2 * p + 8, n);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) mma_m16n8k16_bf16(acc[mt][nt], a[mt], b);
                }
            }
            // Each (h, d) pair is owned by exactly one lane, so a plain += is safe.
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int r = 0; r < 2; ++r)
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int h = mt * 16 + g + r * 8;
                            const int d = warp * 16 + nt * 8 + 2 * p + c;
                            sdQ[h * kHeadDim + d] += acc[mt][nt][r * 2 + c];
                        }
        }

        // ---- GEMM3: dk[idx[j], d] += sum_h ds[h, j] * q[h, d] --------------
        // A = ds transposed (rows j, k = h); B = sQ transposed ([n = d][k = h]).
        {
            float acc[kBlockJ / 16][2][4] = {};
            for (int kk = 0; kk < kNumHeads / 16; ++kk) {
                const int k0 = kk * 16 + 2 * p;
                uint32_t a[kBlockJ / 16][4];
#pragma unroll
                for (int mt = 0; mt < kBlockJ / 16; ++mt) {
                    const int row = mt * 16 + g;
                    a[mt][0] = load_pair_transposed(sDs, kDsStride, k0, row);
                    a[mt][1] = load_pair_transposed(sDs, kDsStride, k0, row + 8);
                    a[mt][2] = load_pair_transposed(sDs, kDsStride, k0 + 8, row);
                    a[mt][3] = load_pair_transposed(sDs, kDsStride, k0 + 8, row + 8);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int n = warp * 16 + nt * 8 + g;
                    uint32_t b[2];
                    b[0] = load_pair_transposed(sQ, kSmemStride, k0, n);
                    b[1] = load_pair_transposed(sQ, kSmemStride, k0 + 8, n);
#pragma unroll
                    for (int mt = 0; mt < kBlockJ / 16; ++mt)
                        mma_m16n8k16_bf16(acc[mt][nt], a[mt], b);
                }
            }
#pragma unroll
            for (int mt = 0; mt < kBlockJ / 16; ++mt)
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int jj = mt * 16 + g + r * 8;
                    const int idx = sIdx[jj];
                    if (idx < 0) continue;
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int d = warp * 16 + nt * 8 + 2 * p + c;
                            const float v = acc[mt][nt][r * 2 + c];
                            if (v != 0.f)
                                atomicAdd(grad_kv + static_cast<size_t>(idx) * kHeadDim + d, v);
                        }
                }
        }
        __syncthreads();
    }

    for (int i = tid; i < kNumHeads * kHeadDim; i += kBwdThreads) {
        grad_q[static_cast<size_t>(m) * kNumHeads * kHeadDim + i] = sdQ[i];
    }
    for (int i = tid; i < kNumHeads; i += kBwdThreads) {
        grad_weights[static_cast<size_t>(m) * kNumHeads + i] = sGw[i];
    }
}

}  // namespace indexer_sm80

std::vector<at::Tensor> bf16_mqa_logits_bwd(
    const at::Tensor& grad_logits, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& weights, const at::Tensor& cu_seqlen_ks,
    const at::Tensor& cu_seqlen_ke, const at::Tensor& topk_indices) {
    using namespace indexer_sm80;
    TORCH_CHECK(q.dtype() == at::kBFloat16 && k.dtype() == at::kBFloat16);
    TORCH_CHECK(grad_logits.dtype() == at::kFloat && weights.dtype() == at::kFloat);
    TORCH_CHECK(topk_indices.dtype() == at::kInt);
    TORCH_CHECK(q.dim() == 3 && k.dim() == 2 && grad_logits.dim() == 2);
    TORCH_CHECK(q.size(1) == kNumHeads && q.size(2) == kHeadDim);

    const int s_q = q.size(0);
    const int s_kv = k.size(0);
    const int topk = topk_indices.size(1);
    TORCH_CHECK(grad_logits.size(0) == s_q && grad_logits.size(1) == topk);
    TORCH_CHECK(topk_indices.size(0) == s_q);

    auto gl = grad_logits.contiguous();
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto wc = weights.contiguous();
    auto idx = topk_indices.contiguous();

    const at::cuda::CUDAGuard guard(q.device());
    auto opts = q.options().dtype(at::kFloat);
    auto grad_q = at::empty({s_q, kNumHeads, kHeadDim}, opts);
    auto grad_kv = at::zeros({s_kv, kHeadDim}, opts);
    auto grad_w = at::empty({s_q, kNumHeads}, opts);

    const size_t smem = (kSmemQBwd + kSmemKgBwd + kSmemDs) * sizeof(__nv_bfloat16) +
                        (kNumHeads * kHeadDim + 2 * kNumHeads + kBlockJ) * sizeof(float) +
                        kBlockJ * sizeof(int);
    static bool attr_set = false;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(mqa_logits_bwd_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        attr_set = true;
    }

    mqa_logits_bwd_kernel<<<s_q, kBwdThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        gl.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()), wc.data_ptr<float>(),
        cu_seqlen_ks.data_ptr<int>(), cu_seqlen_ke.data_ptr<int>(), idx.data_ptr<int>(),
        grad_q.data_ptr<float>(), grad_kv.data_ptr<float>(), grad_w.data_ptr<float>(), s_q,
        s_kv, topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_q, grad_kv, grad_w};
}