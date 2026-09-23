// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

// SM80 (Ampere) BF16 lightning-indexer forward.
//
// Computes, for every query token m and key token n:
//     logits[m, n] = sum_h weights[m, h] * relu(dot(q[m, h, :], k[n, :]))
// masked to -inf outside [cu_seqlen_ks[m], cu_seqlen_ke[m]).
//
// This is the SM80 replacement for `deep_gemm.fp8_mqa_logits`, which only has
// SM90/SM100 implementations.  Ampere has no FP8 tensor cores, so operands are
// BF16 and the per-token dequant scales of the FP8 path disappear (the caller
// folds `softmax_scale` into `weights` exactly as before).
//
// The relu and the weighted reduction over heads happen inside the epilogue,
// in registers.  Materialising the pre-reduction [s_q, num_heads, s_kv] tensor
// would be `num_heads`x larger than the output and is what makes a plain
// cuBLAS GEMM + separate epilogue unusable here.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace indexer_sm80 {

// Tile shape.  kBlockQ query tokens x kBlockN key tokens per CTA, one warp per
// query token.  kBlockN is processed in kNChunk-wide slices so that the mma
// accumulators stay in registers (32 f32 per thread).
constexpr int kHeadDim = 128;
constexpr int kNumHeads = 32;
constexpr int kBlockQ = 8;
constexpr int kBlockN = 128;
constexpr int kNChunk = 32;
constexpr int kWarpSize = 32;
constexpr int kNumThreads = kBlockQ * kWarpSize;

// Row stride of the shared-memory staging buffers.  kHeadDim + 8 makes the
// 4-byte mma fragment loads conflict-free: bank = (g * 68 + p) % 32 is distinct
// for all 32 lanes (g = lane >> 2, p = lane & 3), whereas an unpadded stride of
// 128 collapses to bank == p and costs an 8-way conflict.
constexpr int kSmemStride = kHeadDim + 8;

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4],
                                                 const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Cooperative bf16 copy of `rows` rows of kHeadDim elements into a padded shared
// buffer of row stride kSmemStride.  `src_row_offset(r)` returns the element
// offset of shared row r in the source buffer, which lets callers express
// non-uniform strides (the q tensor is [s_q, kNumHeads, kHeadDim], so staging a
// block of kHeadBlock heads jumps by kNumHeads between tokens).  Rows at or past
// `row_limit` are zero filled so out-of-range tokens contribute nothing.

}  // namespace indexer_sm80