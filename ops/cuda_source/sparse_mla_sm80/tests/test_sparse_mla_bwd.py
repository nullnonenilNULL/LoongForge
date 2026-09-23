# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and performance tests for the SM80 sparse MLA backward.

Gradients are checked against autograd through ``ref_sparse_mla_fwd`` with fp32
leaves, which is the only reference available: ``flash_mla_bwd``'s shipped
``ref_sparse_mla_bwd_interface`` calls ``.backward()`` on the forward's 4-tuple and
passes ``is_casual`` into the ``d_v`` positional slot.
"""

import pytest
import torch

from flash_mla_sm80 import (
    flash_mla_sparse_bwd_sm80,
    flash_mla_sparse_fwd_sm80,
    ref_sparse_mla_fwd,
)

D_QK = 576
D_V = 512

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(s_q, s_kv, h_q, topk, chunk_offset=0, seed=0, pad_frac=0.0, scale=0.25):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s_q, h_q, D_QK, device="cuda", dtype=torch.bfloat16, generator=g) * scale
    kv = torch.randn(s_kv, 1, D_QK, device="cuda", dtype=torch.bfloat16, generator=g) * scale
    dO = torch.randn(s_q, h_q, D_V, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    limit = (torch.arange(s_q, device="cuda") + chunk_offset).clamp_max(s_kv - 1)
    rnd = torch.rand(s_q, 1, topk, device="cuda", generator=g)
    idx = (rnd * (limit + 1).view(s_q, 1, 1).float()).floor().int()
    if pad_frac > 0:
        idx[:, :, int(topk * (1 - pad_frac)):] = -1
    return q, kv, dO, idx


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _ref_grads(q, kv, dO, idx, sm_scale, chunk_offset):
    qr = q.float().detach().requires_grad_(True)
    kr = kv.float().detach().requires_grad_(True)
    out = ref_sparse_mla_fwd(qr.bfloat16(), kr.bfloat16(), idx, sm_scale, D_V, chunk_offset)[0]
    out.float().mul(dO.float()).sum().backward()
    return qr.grad, kr.grad


def _check(s_q, s_kv, h_q, topk, chunk_offset=0, seed=0, pad_frac=0.0, tol=3e-2):
    q, kv, dO, idx = _inputs(s_q, s_kv, h_q, topk, chunk_offset, seed, pad_frac)
    sm_scale = D_QK**-0.5
    out, _, lse, _ = flash_mla_sparse_fwd_sm80(
        q, kv, idx, sm_scale, D_V, q_start_index_s=chunk_offset, write_p_out=True)
    dq, dkv = flash_mla_sparse_bwd_sm80(
        q, kv, out, dO, idx, lse, sm_scale, q_start_index_s=chunk_offset)
    ref_dq, ref_dkv = _ref_grads(q, kv, dO, idx, sm_scale, chunk_offset)

    assert torch.isfinite(dq).all() and torch.isfinite(dkv).all()
    errs = {"dq": _rel(dq, ref_dq), "dkv": _rel(dkv, ref_dkv)}
    for name, err in errs.items():
        assert err < tol, f"{name} relative error {err:.3e} exceeds {tol:.1e}"
    return errs


@pytest.mark.parametrize("s_q,s_kv,topk", [(32, 64, 32), (64, 64, 64), (128, 128, 128),
                                          (128, 256, 64)])
def test_grads(s_q, s_kv, topk):
    _check(s_q, s_kv, 128, topk)


@pytest.mark.parametrize("chunk_idx", [1, 2, 3])
def test_chunkpipe_offsets(chunk_idx):
    chunksize = 64
    _check(chunksize, chunksize * (chunk_idx + 1), 128, 128,
           chunk_offset=chunksize * chunk_idx, seed=chunk_idx)


@pytest.mark.parametrize("h_q", [16, 32, 64, 128])
def test_head_counts(h_q):
    _check(64, 128, h_q, 128, seed=5)


def test_padded_topk():
    _check(64, 128, 128, 256, pad_frac=0.5, seed=9)


def test_duplicate_indices_accumulate():
    """Every query selecting the same key must sum into that row, not race."""
    s_q, s_kv, h_q, topk = 64, 8, 128, 32
    q, kv, dO, idx = _inputs(s_q, s_kv, h_q, topk, seed=11)
    idx = torch.zeros_like(idx)  # all queries attend to key 0 only
    sm_scale = D_QK**-0.5
    out, _, lse, _ = flash_mla_sparse_fwd_sm80(q, kv, idx, sm_scale, D_V, write_p_out=True)
    dq, dkv = flash_mla_sparse_bwd_sm80(q, kv, out, dO, idx, lse, sm_scale)
    ref_dq, ref_dkv = _ref_grads(q, kv, dO, idx, sm_scale, 0)
    assert _rel(dkv, ref_dkv) < 3e-2
    assert dkv[1:].abs().max() == 0, "only key 0 should receive gradient"


def test_lonely_rows():
    """Queries with no valid key must give zero gradient, not NaN."""
    s_q, s_kv, h_q, topk = 32, 64, 128, 64
    q, kv, dO, idx = _inputs(s_q, s_kv, h_q, topk)
    idx = torch.full_like(idx, -1)
    sm_scale = D_QK**-0.5
    out, _, lse, _ = flash_mla_sparse_fwd_sm80(q, kv, idx, sm_scale, D_V, write_p_out=True)
    dq, dkv = flash_mla_sparse_bwd_sm80(q, kv, out, dO, idx, lse, sm_scale)
    assert torch.isposinf(lse).all()
    assert dq.abs().max() == 0 and dkv.abs().max() == 0


if __name__ == "__main__":
    import sys
    import time

    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    for args, name in [((128, 128, 128, 128), "causal 128"),
                       ((64, 256, 128, 128, 192), "chunkpipe chunk 4 of 4"),
                       ((64, 128, 128, 256, 0, 9, 0.5), "padded topk")]:
        try:
            errs = _check(*args)
            print(f"[OK]   {name}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")

    for s_q, topk in [(512, 512)]:
        q, kv, dO, idx = _inputs(s_q, s_q, 128, topk)
        sm_scale = D_QK**-0.5
        out, _, lse, _ = flash_mla_sparse_fwd_sm80(q, kv, idx, sm_scale, D_V, write_p_out=True)
        for _ in range(2):
            flash_mla_sparse_bwd_sm80(q, kv, out, dO, idx, lse, sm_scale)
        torch.cuda.synchronize()
        t0, iters = time.perf_counter(), 5
        for _ in range(iters):
            flash_mla_sparse_bwd_sm80(q, kv, out, dO, idx, lse, sm_scale)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        # qk + dp + dq + two dkv chains
        flops = s_q * 128 * topk * (D_QK + D_V + D_QK + D_QK + D_V) * 2
        print(f"s_q={s_q} topk={topk} h_q=128: {ms:.2f} ms  "
              f"{flops / (ms * 1e-3) / 1e12:.1f} TFLOP/s (peak ~312)")
    sys.exit(1 if failures else 0)