# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SM80 BF16 lightning indexer: python interface + numerical reference.

Replaces `deep_gemm.fp8_mqa_logits` on Ampere, which has SM90/SM100 kernels only.
Operands are BF16 because SM80 has no FP8 tensor cores; the caller therefore
folds only `softmax_scale` into `weights` (no per-token dequant scale).
"""

from typing import Tuple

import torch

try:
    from . import cuda as _C  # type: ignore
except ImportError:  # pragma: no cover - built out of tree / editable dev
    _C = None


def _require_ext():
    if _C is None:
        raise ImportError(
            "lightning_indexer_sm80 CUDA extension is not built. "
            "Run `pip install --no-build-isolation -e .` in cuda_source/lightning_indexer_sm80."
        )
    return _C


def bf16_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Lightning-indexer logits.

    Args:
        q:            [s_q, num_heads, head_dim] bfloat16 (num_heads=32, head_dim=128)
        k:            [s_kv, head_dim]           bfloat16
        weights:      [s_q, num_heads]           float32, softmax_scale already folded in
        cu_seqlen_ks: [s_q] int32, inclusive KV window start per query token
        cu_seqlen_ke: [s_q] int32, exclusive KV window end per query token

    Returns:
        logits: [s_q, s_kv] float32, -inf outside the window.
    """
    return _require_ext().bf16_mqa_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke)


def ref_bf16_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    head_chunk: int = 8,
) -> torch.Tensor:

    """Pure-PyTorch reference, chunked over heads.

    The obvious `einsum('mhd,nd->hmn')` formulation materialises a tensor
    `num_heads` times larger than the output (8.6 GB bf16 at s_q=4096,
    s_kv=32768), so accumulate over head chunks instead.
    """
    s_kv = k.shape[0]
    num_heads = q.shape[1]
    kf = k.float()
    logits = torch.zeros(q.shape[0], s_kv, device=q.device, dtype=torch.float32)
    for h0 in range(0, num_heads, head_chunk):
        h1 = min(h0 + head_chunk, num_heads)
        score = torch.einsum("mhd,nd->mhn", q[:, h0:h1].float(), kf)
        logits += (score.relu() * weights[:, h0:h1].unsqueeze(-1)).sum(dim=1)
    pos = torch.arange(s_kv, device=q.device)
    mask = (pos[None, :] >= cu_seqlen_ks[:, None]) & (pos[None, :] < cu_seqlen_ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))


def causal_kv_range(s_q: int, chunk_offset: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """KV window used by the unpacked (pretrain) fused-DSA path.

    Mirrors dsa_fused_kernels.py:2349-2360: k_start is all zeros and
    k_end = arange(s_q) + chunk_offset + 1.  chunk_offset is non-zero under
    ChunkPipe, where s_kv spans all chunks seen so far while s_q is one chunk.
    """
    ks = torch.zeros(s_q, dtype=torch.int32, device=device)
    ke = torch.arange(s_q, dtype=torch.int32, device=device) + chunk_offset + 1
    return ks, ke


def bf16_mqa_logits_bwd(
    grad_logits: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indexer backward, restricted to the selected top-k positions.

    Args:
        grad_logits:  [s_q, topk] float32, gradient of the loss wrt the selected logits
        q:            [s_q, num_heads, head_dim] bfloat16
        k:            [s_kv, head_dim]           bfloat16
        weights:      [s_q, num_heads]           float32, the same scaled weights as the forward
        cu_seqlen_ks: [s_q] int32
        cu_seqlen_ke: [s_q] int32
        topk_indices: [s_q, topk] int32; negative or out-of-window entries are ignored

    Returns:
        ``(grad_q, grad_k, grad_weights)``, all float32.  ``grad_weights`` is with
        respect to the *scaled* weights, matching the FP8 kernel, so the caller still
        has to apply the chain rule for whatever it folded in.
    """
    return tuple(
        _require_ext().bf16_mqa_logits_bwd(
            grad_logits.contiguous(), q, k, weights, cu_seqlen_ks, cu_seqlen_ke,
            topk_indices.int(),
        )
    )


def ref_bf16_mqa_logits_bwd(
    grad_logits: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    token_chunk: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for :func:`bf16_mqa_logits_bwd`."""
    s_q, num_heads, head_dim = q.shape
    s_kv = k.shape[0]
    qf, kf, wf = q.float(), k.float(), weights.float()
    grad_q = torch.zeros_like(qf)
    grad_k = torch.zeros_like(kf)
    grad_w = torch.zeros_like(wf)

    for m0 in range(0, s_q, token_chunk):
        m1 = min(m0 + token_chunk, s_q)
        idx = topk_indices[m0:m1].long()
        pos = idx.clamp(min=0, max=max(s_kv - 1, 0))
        valid = (idx >= 0) & (pos >= cu_seqlen_ks[m0:m1, None]) & (pos < cu_seqlen_ke[m0:m1, None])

        kg = kf[pos]
        s = torch.einsum("mhd,mtd->mht", qf[m0:m1], kg)
        gl = (grad_logits[m0:m1].float() * valid).unsqueeze(1)

        grad_w[m0:m1] = (gl * s.relu()).sum(dim=-1)
        ds = gl * wf[m0:m1].unsqueeze(-1) * (s > 0)
        grad_q[m0:m1] = torch.einsum("mht,mtd->mhd", ds, kg)
        contrib = torch.einsum("mht,mhd->mtd", ds, qf[m0:m1])
        grad_k.index_add_(0, pos.reshape(-1),
                          (contrib * valid.unsqueeze(-1)).reshape(-1, head_dim))

    return grad_q, grad_k, grad_w