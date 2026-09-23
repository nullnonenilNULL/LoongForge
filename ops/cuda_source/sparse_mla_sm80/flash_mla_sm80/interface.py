# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SM80 sparse MLA: python interface and numerical reference."""

from typing import Optional, Tuple

import torch

try:
    from . import cuda as _C  # type: ignore
except ImportError:  # pragma: no cover
    _C = None


def _require_ext():
    if _C is None:
        raise ImportError(
            "flash_mla_sm80 CUDA extension is not built. Run "
            "`pip install --no-build-isolation -e .` in cuda_source/sparse_mla_sm80."
        )
    return _C


def flash_mla_sparse_fwd_sm80(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    q_start_index_s: int = 0,
    write_p_out: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    window_size: int = 0,
):
    """Sparse MLA forward on Ampere.

    Signature matches ``flash_mla_fwd.flash_mla_sparse_fwd`` so the two are
    interchangeable behind a backend switch.

    Args:
        q:        [s_q, h_q, 576] bfloat16
        kv:       [s_kv, 1, 576]  bfloat16
        indices:  [s_q, 1, topk]  int32; negative or >= s_kv entries are ignored
        sm_scale: defaults to ``d_qk ** -0.5``
        q_start_index_s: absolute position of query 0, i.e. the ChunkPipe chunk offset
        write_p_out: also return the unscaled masked qk scores

    Returns:
        ``(out, max_logits, lse[, p_out])``.  ``lse`` is **natural log** with ``+inf``
        for query tokens that have no valid key.  Note the SM100 docstring claims
        base-2; that kernel also returns natural log (``logf(li) + mi * ln2``).

    Not implemented (unused by the GLM-5 fused-DSA path, which hardcodes
    ``attn_sink=None``, ``window_size=0`` and never passes ``topk_length``):
    ``topk_length``, ``attn_sink``, ``window_size``.
    """
    if topk_length is not None:
        raise NotImplementedError("sm80 sparse MLA does not support topk_length")
    if attn_sink is not None:
        raise NotImplementedError("sm80 sparse MLA does not support attn_sink")
    if window_size:
        raise NotImplementedError("sm80 sparse MLA does not support window_size")
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return tuple(
        _require_ext().sparse_prefill_fwd_sm80(
            q, kv, indices, float(sm_scale), int(d_v), int(q_start_index_s), bool(write_p_out)
        )
    )


def flash_mla_sparse_bwd_sm80(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: Optional[float] = None,
    q_start_index_s: int = 0,
    topk_length: Optional[torch.Tensor] = None,
    fast_mode: bool = False,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA backward on Ampere, returning ``(grad_q, grad_kv)``.

    ``lse`` must be the natural-log LSE returned by the forward, with ``+inf`` for
    query tokens that have no valid key.  P is recomputed from q, kv and lse rather
    than stashed, so the only extra state the forward has to keep is ``lse``.

    ``fast_mode`` is accepted and ignored: on SM100 it selects a lower-precision
    variant, and there is only one variant here.
    """
    if topk_length is not None:
        raise NotImplementedError("sm80 sparse MLA does not support topk_length")
    if attn_sink is not None:
        raise NotImplementedError("sm80 sparse MLA does not support attn_sink")
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    grad_q, grad_kv = _require_ext().sparse_prefill_bwd_sm80(
        q, kv, out, grad_out, indices, lse, float(sm_scale), int(q_start_index_s)
    )
    return grad_q, grad_kv


def ref_sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    q_start_index_s: int = 0,
    token_chunk: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-safe PyTorch reference, 3-D in/out, chunked over query tokens.

    Neither shipped reference is usable as-is: ``flash_mla_fwd``'s writes
    ``lse[lonely] = inf`` in place on the output of ``logsumexp`` (autograd refuses
    to differentiate through it), and ``flash_mla_bwd``'s calls ``.backward()`` on
    the forward's 4-tuple while passing ``is_casual`` into the ``d_v`` slot.  The
    TileLang op in ``tilelang_ops/sparse_mla_fwd.py`` is an independent third
    reference and agrees with this one.
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    assert h_kv == 1, f"h_kv must be 1, got {h_kv}"
    if sm_scale is None:
        sm_scale = d_qk**-0.5
    kvf = kv[:, 0].float()
    idx_all = indices[:, 0]

    outs, maxes, lses, p_outs = [], [], [], []
    for m0 in range(0, s_q, token_chunk):
        m1 = min(m0 + token_chunk, s_q)
        idx = idx_all[m0:m1].long()
        gathered = kvf[idx.clamp(min=0, max=max(s_kv - 1, 0))]
        qk = torch.einsum("shd,std->sht", q[m0:m1].float(), gathered)

        limit = torch.arange(q_start_index_s + m0, q_start_index_s + m1,
                             device=q.device, dtype=idx.dtype).unsqueeze(-1)
        valid = ((idx >= 0) & (idx < s_kv) & (idx <= limit)).unsqueeze(1)

        neg_inf = torch.full_like(qk, float("-inf"))
        p_outs.append(torch.where(valid, qk, neg_inf))
        score = torch.where(valid, qk * sm_scale, neg_inf)

        maxes.append(score.amax(dim=-1))
        lse = torch.logsumexp(score, dim=-1)
        lonely = torch.isneginf(lse)
        pos_inf = torch.full_like(lse, float("inf"))
        prob = torch.exp(score - torch.where(lonely, pos_inf, lse).unsqueeze(-1))
        outs.append(torch.einsum("sht,std->shd", prob, gathered[..., :d_v]))
        lses.append(torch.where(lonely, pos_inf, lse))

    return (torch.cat(outs).to(q.dtype), torch.cat(maxes), torch.cat(lses),
            torch.cat(p_outs))