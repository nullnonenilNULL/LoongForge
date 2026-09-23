# sparse_mla_sm80

Sparse MLA kernels for **NVIDIA Ampere (SM80)**, used by the fused DeepSeek Sparse
Attention (DSA) path on A100 / A800. Package name is `flash_mla_sm80`.

## Status

| kernel | state |
| --- | --- |
| `flash_mla_sparse_fwd_sm80` | implemented, 14 tests passing |
| `flash_mla_sparse_bwd_sm80` | implemented, 14 tests passing |

LoongForge's `sm80` backend routes both directions here. It deliberately does
**not** use the pre-existing TileLang `major != 10` fallback: with `h_q = 128` that
op splits the heads in two and then asks for 216 KB of dynamic shared memory (Q, dO
and the dkv accumulators for 64 heads × 512) against A100's 164 KB limit, failing
with `Failed to set the allowed dynamic shared memory size to 221184`. Splitting
the heads four ways would fit, but that is a change to the TileLang op itself.

## API

```python
from flash_mla_sm80 import flash_mla_sparse_fwd_sm80, ref_sparse_mla_fwd

out, max_logits, lse, p_out = flash_mla_sparse_fwd_sm80(
    q,                    # [s_q, h_q, 576] bfloat16
    kv,                   # [s_kv, 1, 576]  bfloat16
    indices,              # [s_q, 1, topk]  int32, negative / >= s_kv ignored
    sm_scale,             # default 576 ** -0.5
    d_v=512,
    q_start_index_s=0,    # absolute position of query 0 = ChunkPipe chunk offset
    write_p_out=True,
)

grad_q, grad_kv = flash_mla_sparse_bwd_sm80(
    q, kv, out, grad_out, indices, lse, sm_scale, q_start_index_s=0,
)
```

Not implemented, because the GLM-5 fused path hardcodes them away: `topk_length`,
`attn_sink`, `window_size`.

Two contracts worth restating:

- **`lse` is natural log**, not base 2: the kernel computes `logf(li) + mi * ln2`.
  `+inf` marks query tokens with no valid key.
- **`p_out` is the *unscaled* masked `qk`**, not `qk * sm_scale` and not a
  probability. `triton_attn_dist` applies the scale itself. `test_p_out_is_unscaled`
  pins this down.

`ref_sparse_mla_fwd` is an autograd-safe PyTorch reference. The TileLang op in
`tilelang_ops/sparse_mla_fwd.py` is an independent second reference and agrees.

## Build & test

```bash
pip install --no-build-isolation -e .
python -m pytest tests/ -q            # 14 cases
python tests/test_sparse_mla_fwd.py   # smoke test + throughput
```

`TORCH_CUDA_ARCH_LIST` is honoured but filtered to sm_80 and above: the kernel
uses `mma.sync.aligned.m16n8k16...bf16`, which ptxas rejects for sm_75, and many
images ship a broad default list.

Accuracy against the reference: `p_out` ~7e-7, `lse` ~2e-7, `out` ~5e-3, `dq` ~5e-3,
`dkv` ~4e-3. The last three are looser because `P` is rounded to BF16 before the PV
GEMM.

The backward's `fast_mode` argument is accepted and ignored: there is only one
variant here.

## Implementation notes

Both kernels use the same decomposition: a CTA owns one query token and
`kHeadBlock = 16` of its heads, with 4 warps and `kBlockT = 32` top-k positions
staged per iteration. The forward runs a standard online softmax; the backward
recomputes `P` from `q`, `kv` and `lse` rather than reading it back, so the forward
only has to hand over `lse`.

- **Head blocking is a capacity compromise, and it is the main performance
  limit.** One CTA per token covering all 128 heads would read the gathered kv
  exactly once (2.36 MB per token at topk=2048); the operator's arithmetic
  intensity is ~237 FLOP/byte against A100's ~200 ridge point, so a single pass is
  what would make it compute bound. It does not fit: the forward's output
  accumulator alone is 128 × 512 fp32 = 256 KB, which is the entire register file of
  an SM and well past the 164 KB shared-memory limit, and the backward's dQ
  accumulator is worse at 128 × 576. With `kHeadBlock = 16` the gather is read eight
  times.
- Head blocks are the **fast** grid axis (`grid = (h_q / kHeadBlock, s_q)`) so the
  CTAs sharing a token — and therefore the whole gather — are launched adjacently
  and hit in L2. With the token on `x` they are `h_q / kHeadBlock` apart and each
  re-reads from HBM.
- Shared row stride is `576 + 8`; unpadded, every mma fragment load collapses onto
  one bank per column pair.
- `mma.sync` is `.row.col`, so a GEMM whose B operand is a resident row-major buffer
  needs it transposed. Rather than materialising the transpose, fragments are
  assembled from pairs of 16-bit shared loads (`load_pair_transposed`). The forward
  needs this for V in the PV GEMM; the backward needs it for the gathered kv in dQ
  and for Q and dO in the two dKV chains.
- The backward stages `P` and `dQK` in **both** `[h][t]` and `[t][h]` layouts: dQ
  consumes them with heads as the mma M dimension and dKV with top-k positions as M.
  Writing both costs 6 KB of shared memory and saves a transposed read in the inner
  loop.
- Because K and V are the same tensor, a key row's gradient is the sum of two
  GEMMs — `dQK^T @ Q` over all 576 dims and `P^T @ dO` over the first 512. Both land
  in one accumulator, walked in 64-wide slices of `d` so the accumulator stays at 16
  registers per thread alongside dQ's 72.
- dKV is scattered with `atomicAdd` into an fp32 buffer and cast at the end: many
  (query, top-k) pairs select the same key row, so the scatter genuinely collides,
  and bf16 atomics would both lose precision and need a CAS loop.
- Shared memory is 58 KB forward and 77 KB backward, so two CTAs fit per SM either
  way.

## Performance status

| kernel | shape | time | effective |
| --- | --- | --- | --- |
| forward | `s_q = topk = 512`, `h_q = 128` | 5.34 ms | 13.7 TFLOP/s |
| forward | `s_q = topk = 1024`, `h_q = 128` | 20.3 ms | 14.4 TFLOP/s |
| backward | `s_q = topk = 512`, `h_q = 128` | 12.6 ms | 14.6 TFLOP/s |

~4.5 % of the 312 TFLOP/s BF16 peak. Correct but untuned; swapping the forward's
grid axes (above) bought only 4 %, which says the kernels are issue bound rather
than bandwidth bound. Backlog, in expected order of payoff:

1. **`ldmatrix` for the mma fragments.** The QK loop issues 6 separate 32-bit
   shared loads per k-step to feed a single mma; `ldmatrix.x4` replaces the four A
   loads with one instruction. Load-to-mma ratio is `4 / n_tiles_per_warp + 2`, so
   this is the dominant term at `kBlockT = 32`.
2. **More n-tiles per warp.** Raising `kBlockT` to 128 gives each warp four QK
   n-tiles and cuts the ratio from 6 to 3, at the cost of `kBlockT × 584 × 2` bytes
   of shared memory.
3. **Larger `kHeadBlock`.** 32 heads halves the gather passes but doubles the
   forward's PV accumulator to 128 registers per thread; worth measuring for spills.
4. Double buffer the kv gather so the `cp.async` overlaps the QK mma.
5. **Backward only: cut the dKV atomics.** Every CTA atomically adds 576 floats per
   top-k position, and the eight head blocks of a token all collide on the same key
   rows. A per-CTA shared-memory reduction across head blocks, or a two-stage
   scatter, would remove most of that traffic.
