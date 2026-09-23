# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and performance tests for the SM80 sparse MLA forward.

Shapes follow the GLM-5 fused-DSA path: h_q padded to 128, d_qk = 576, d_v = 512,
h_kv = 1.  ChunkPipe cases (q_start_index_s > 0, s_kv > s_q) are covered
explicitly, as are fully-masked ("lonely") query rows.
"""

import pytest
import torch

from flash_mla_sm80 import flash_mla_sparse_fwd_sm80, ref_sparse_mla_fwd

D_QK = 576
D_V = 512

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_inputs(s_q, s_kv, h_q, topk, chunk_offset=0, seed=0, pad_frac=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s_q, h_q, D_QK, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25
    kv = torch.randn(s_kv, 1, D_QK, device="cuda", dtype=torch.bfloat16, generator=g) * 0.25
    limit = (torch.arange(s_q, device="cuda") + chunk_offset).clamp_max(s_kv - 1)
    rnd = torch.rand(s_q, 1, topk, device="cuda", generator=g)
    idx = (rnd * (limit + 1).view(s_q, 1, 1).float()).floor().int()
    if pad_frac > 0:
        # emulate padded_flashinfer_topk, which fills the tail with invalid entries
        keep = int(topk * (1 - pad_frac))
        idx[:, :, keep:] = -1
    return q, kv, idx


def _rel(a, b, mask=None):
    a, b = a.float(), b.float()
    if mask is not None:
        a, b = a[mask], b[mask]
    if b.numel() == 0:
        return 0.0
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _check(s_q, s_kv, h_q, topk, chunk_offset=0, seed=0, pad_frac=0.0, tol=2e-2):
    q, kv, idx = _make_inputs(s_q, s_kv, h_q, topk, chunk_offset, seed, pad_frac)
    sm_scale = D_QK**-0.5
    out, maxl, lse, p_out = flash_mla_sparse_fwd_sm80(
        q, kv, idx, sm_scale, D_V, q_start_index_s=chunk_offset, write_p_out=True)
    r_out, r_max, r_lse, r_p = ref_sparse_mla_fwd(q, kv, idx, sm_scale, D_V, chunk_offset)

    finite = torch.isfinite(r_p)
    assert torch.equal(finite, torch.isfinite(p_out)), "p_out mask mismatch"
    errs = {
        "out": _rel(out, r_out),
        "p_out": _rel(p_out, r_p, finite),
        "lse": _rel(torch.nan_to_num(lse, posinf=0.0), torch.nan_to_num(r_lse, posinf=0.0)),
    }
    assert torch.equal(torch.isposinf(lse), torch.isposinf(r_lse)), "lonely-row lse mismatch"
    for name, err in errs.items():
        assert err < tol, f"{name} relative error {err:.3e} exceeds {tol:.1e}"
    return errs


@pytest.mark.parametrize("s_q,s_kv,topk", [(32, 64, 64), (64, 64, 64), (128, 128, 128),
                                          (256, 256, 256)])
def test_causal(s_q, s_kv, topk):
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
    """padded_flashinfer_topk fills the tail with invalid indices."""
    _check(64, 128, 128, 256, pad_frac=0.5, seed=9)


def test_lonely_rows():
    """A query with no valid key must give out=0 and lse=+inf, not NaN."""
    s_q, s_kv, h_q, topk = 32, 64, 128, 64
    q, kv, idx = _make_inputs(s_q, s_kv, h_q, topk)
    idx = torch.full_like(idx, -1)
    out, maxl, lse, p_out = flash_mla_sparse_fwd_sm80(
        q, kv, idx, D_QK**-0.5, D_V, write_p_out=True)
    assert out.abs().max() == 0
    assert torch.isposinf(lse).all()
    assert torch.isneginf(p_out).all()


def test_p_out_is_unscaled():
    """p_out must be the raw qk, not qk * sm_scale (triton_attn_dist scales it)."""
    q, kv, idx = _make_inputs(32, 64, 128, 64, seed=3)
    _, _, _, p_a = flash_mla_sparse_fwd_sm80(q, kv, idx, 1.0, D_V, write_p_out=True)
    _, _, _, p_b = flash_mla_sparse_fwd_sm80(q, kv, idx, 0.1, D_V, write_p_out=True)
    finite = torch.isfinite(p_a)
    assert _rel(p_a, p_b, finite) < 1e-6


if __name__ == "__main__":
    import sys
    import time

    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    for args, name in [((256, 256, 128, 256), "causal 256"),
                       ((64, 256, 128, 128, 192), "chunkpipe chunk 4 of 4"),
                       ((64, 128, 128, 256, 0, 9, 0.5), "padded topk")]:
        try:
            errs = _check(*args)
            print(f"[OK]   {name}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")

    for s_q, topk in [(512, 512), (1024, 1024)]:
        q, kv, idx = _make_inputs(s_q, s_q, 128, topk)
        for _ in range(3):
            flash_mla_sparse_fwd_sm80(q, kv, idx, D_QK**-0.5, D_V, write_p_out=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            flash_mla_sparse_fwd_sm80(q, kv, idx, D_QK**-0.5, D_V, write_p_out=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        tflops = s_q * 128 * topk * (D_QK + D_V) * 2 / (ms * 1e-3) / 1e12
        print(f"s_q={s_q} topk={topk} h_q=128: {ms:.3f} ms  {tflops:.1f} TFLOP/s (peak ~312)")
    sys.exit(1 if failures else 0)