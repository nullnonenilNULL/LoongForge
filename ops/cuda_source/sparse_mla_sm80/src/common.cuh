// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

// Shared definitions for the SM80 sparse MLA kernels.
//
// Shapes are those of the fused-DSA path on GLM-5: h_q is padded to 128,
// d_qk = 576 (kv_lora_rank 512 + qk_pos_emb_head_dim 64), d_v = 512, h_kv = 1.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sparse_mla_sm80 {

constexpr int kDQK = 576;
constexpr int kDV = 512;
constexpr int kWarpSize = 32;

// A CTA owns one query token and kHeadBlock of its heads.
//
// Ideally a CTA would own all 128 heads so the gathered kv is read exactly once
// (2.36 MB per token at topk=2048); the arithmetic intensity of the whole
// operator is ~237 FLOP/byte, just above A100's 200 ridge point, so one pass is
// what makes it compute bound.  It does not fit: the output accumulator alone is
// 128 x 512 fp32 = 256 KB, which is the entire register file of an SM and well
// past the 164 KB shared-memory limit.  kHeadBlock is therefore a
// traffic-versus-capacity knob, and 128 / kHeadBlock passes are made over the kv
// gather.  See README for the measured cost.
constexpr int kHeadBlock = 16;
constexpr int kBlockT = 32;  // top-k positions staged per iteration
constexpr int kNumWarps = 4;
constexpr int kNumThreads = kNumWarps * kWarpSize;

// Padded shared strides; an unpadded 576 (= 18 * 32 words) would make every mma
// fragment load hit the same bank for all lanes sharing a column pair.
constexpr int kQKStride = kDQK + 8;
constexpr int kPStride = kBlockT + 8;

static_assert(kBlockT == kNumWarps * 8, "QK gives each warp exactly one n-tile of 8");
static_assert(kDV % (kNumWarps * 8) == 0, "PV splits d_v across warps in 8-wide tiles");
static_assert(kHeadBlock == 16, "softmax epilogue assumes a single 16-row mma tile");
static_assert(kDQK % 16 == 0 && kBlockT % 16 == 0, "mma k-steps are 16 wide");

__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Assemble one bf16 pair from a column of a row-major shared buffer: reads
// src[k0][col] and src[k0 + 1][col].  Used where mma's `.row.col` B operand needs
// the transpose of a buffer that is already resident, which is cheaper than
// materialising the transpose.
__device__ __forceinline__ uint32_t load_pair_transposed(const __nv_bfloat16* src, int stride,
                                                         int k0, int col) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
    return static_cast<uint32_t>(p[k0 * stride + col]) |
           (static_cast<uint32_t>(p[(k0 + 1) * stride + col]) << 16);
}

}  // namespace sparse_mla_sm80