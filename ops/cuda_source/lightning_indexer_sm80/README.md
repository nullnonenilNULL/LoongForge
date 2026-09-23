# lightning_indexer_sm80

BF16 lightning-indexer kernels for **NVIDIA Ampere (SM80)**, used by the fused
DeepSeek Sparse Attention (DSA) path when running on A100 / A800.

## Why this exists

Ampere (SM80) has no FP8 tensor cores, so the lightning-indexer operands are BF16
here: `q`/`k` are bfloat16 and only `softmax_scale` is folded into `weights` (there
is no per-token dequant scale). This package provides both the forward and backward
indexer kernels for the fused-DSA path on A100 / A800.

## Why not cuBLAS

The inner product `q[m,h,:] · k[n,:]` is an ordinary GEMM, but `relu` is applied
**before** the weighted sum over heads, so a library GEMM has to materialise
`Z[s_q, num_heads, s_kv]` — `num_heads`× larger than the output:

| shape (ChunkPipe, seq 32768, chunksize 4096) | bytes |
| --- | --- |
| intermediate `Z` bf16 | 8.6 GB |
| output `logits` fp32 | 537 MB |

cuBLASLt epilogues cover RELU and per-tensor scaling but not "per-row scale +
cross-row-group reduction", so the fusion is not expressible. Splitting into 32
per-head GEMMs does not help either (relu precedes the sum, so `beta=1`
accumulation is unusable). Hence a custom kernel that keeps relu and the head
reduction in registers.

## API

```python
from lightning_indexer_sm80 import (
    bf16_mqa_logits, bf16_mqa_logits_bwd,
    ref_bf16_mqa_logits, ref_bf16_mqa_logits_bwd, causal_kv_range,
)

logits = bf16_mqa_logits(
    q,             # [s_q, 32, 128]  bfloat16
    k,             # [s_kv, 128]     bfloat16
    weights,       # [s_q, 32]       float32, softmax_scale already folded in
    cu_seqlen_ks,  # [s_q] int32, inclusive KV window start
    cu_seqlen_ke,  # [s_q] int32, exclusive KV window end
)                  # -> [s_q, s_kv] float32, -inf outside the window

grad_q, grad_k, grad_weights = bf16_mqa_logits_bwd(
    grad_logits,   # [s_q, topk] float32
    q, k, weights, cu_seqlen_ks, cu_seqlen_ke,
    topk_indices,  # [s_q, topk] int32; negative or out-of-window entries ignored
)                  # all float32; grad_weights is wrt the *scaled* weights
```

`causal_kv_range(s_q, chunk_offset, device)` reproduces the window used by the
unpacked (pretrain) fused-DSA path: `ks = 0`, `ke = arange(s_q) + chunk_offset + 1`.
`chunk_offset` is non-zero under ChunkPipe, where `s_kv` spans every chunk seen so
far while `s_q` is a single chunk — both cases are covered by the tests.

## Build

```bash
pip install --no-build-isolation -e .
```

Defaults to `sm_80`. `TORCH_CUDA_ARCH_LIST` is honoured but **filtered** to
sm_80 and above: the kernel uses `mma.sync.aligned.m16n8k16...bf16`, which ptxas
rejects for sm_75, and many images ship a broad default list such as
`"7.5 8.0 8.6 9.0 10.0 12.0+PTX"`.

## Test

```bash
python -m pytest tests/ -q            # 28 cases
python tests/test_mqa_logits_fwd.py   # smoke test + throughput
python tests/test_mqa_logits_bwd.py
```

Forward accuracy versus the chunked PyTorch reference is ~1.4e-7 relative (the
kernel accumulates in fp32, so only summation order differs).  Backward is
~1.6e-3 on `d_q`/`d_k` and ~1.8e-7 on `d_weights`: `ds` is rounded to BF16 before
the two gradient GEMMs. Thresholds: `d_q`/`d_k` < 1e-2, `d_weights` < 1e-3.

## Implementation notes

### Forward

Tile: `kBlockQ = 8` query tokens × `kBlockN = 128` keys per CTA, one warp per
query token, 256 threads.

- Heads are staged in blocks of `kHeadBlock = 16` so a CTA needs 73 KB of shared
  memory and two fit per SM. Staging all 32 heads at once costs 103 KB and halves
  occupancy.
- The shared row stride is `kHeadDim + 8`. With an unpadded stride of 128 every
  mma fragment load collapses to `bank == lane & 3`, an 8-way conflict; the pad
  makes `bank = (g * 68 + p) % 32` distinct across all 32 lanes.
- `q` is `[s_q, num_heads, head_dim]`, so within a head block consecutive shared
  rows are contiguous but the step between tokens is `num_heads` rows, not
  `kHeadBlock`. `load_q_block` handles this; a uniform row stride silently reads
  the wrong heads.
- Whole tiles beyond the causal frontier are skipped before any mma, but still
  write `-inf` because the output tensor is uninitialised.

### Backward

One CTA per query token, 8 warps, looping over `kBlockJ = 64` top-k positions.
Three GEMMs per j-block: recompute `s`, then `dq = ds @ kg` and
`dk = ds^T @ q`.

- Keeping one token per CTA lets the `dq` accumulator live in shared memory and be
  written once. Splitting j across CTAs would need atomics on `dq` too.
- `mma.sync` is `.row.col`, so B must be n-major with k contiguous. Two of the
  three GEMMs need the transpose of a buffer that is already resident; those
  fragments are assembled from pairs of 16-bit shared loads
  (`load_pair_transposed`) instead of materialising a transposed copy. That saves
  ~29 KB of shared memory and, more importantly, avoids 8192 strided transpose
  writes per j-block.
- Shared memory is 48 KB, so three CTAs fit per SM.
- Out-of-window and negative top-k indices are neutralised once, when the j-block
  is gathered: the kv row is zero filled and `grad_logits` is forced to 0.

## Performance status

| kernel | shape | time | effective |
| --- | --- | --- | --- |
| forward | `s_q = s_kv = 2048`, causal | 0.30 ms | 58 TFLOP/s |
| forward | `s_q = s_kv = 4096`, causal | 1.13 ms | 61 TFLOP/s |
| backward | `s_q = s_kv = 1024`, topk 512 | 1.24 ms | 10 TFLOP/s |
| backward | `s_q = s_kv = 2048`, topk 1024 | 4.04 ms | 13 TFLOP/s |

Both are correct and both are untuned. Optimisation backlog, in expected order of
payoff:

**Forward** (~20 % of the 312 TFLOP/s BF16 peak)

1. `ldmatrix` for the A/B fragments — currently 12 separate 32-bit shared loads
   per k-step feed only 4 mma.
2. Raise mma-per-fragment-load by giving each warp two m-tiles (32 heads) or two
   query tokens, trading shared memory for arithmetic density.
3. Occupancy is 16 warps/SM out of 64 (shared-memory bound); double buffering the
   Q head block would overlap the staging copies.
4. The epilogue reduction writes from 4 of 32 lanes; a transpose through shared
   memory would use the full warp.

**Backward** (~4 % of peak — the weaker of the two)

1. Scalar fp32 `atomicAdd` on `grad_kv` is the prime suspect: 8192 atomics per
   j-block, ~268 M for `s_q = 2048, topk = 1024`. SM80 has no vectorised fp32
   atomic (`red.global.add.v2/v4.f32` requires a newer architecture), so the
   structural fix is to invert the index and gather per kv row instead of
   scattering per query token. Measure before redesigning.
2. GEMM3's A fragments cost 32 16-bit loads per k-step for 16 mma. Storing `ds`
   in both orientations (`[h][j]` and `[j][h]`, +5 KB) makes them single 32-bit
   loads; the values are already in registers in the GEMM1 epilogue.
3. `kBlockJ = 64` gives GEMM3 only two k-steps (K = 32 heads); a larger j-block
   amortises the gather and the A-fragment loads better.

