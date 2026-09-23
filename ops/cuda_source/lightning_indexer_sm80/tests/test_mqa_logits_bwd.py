# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and performance tests for the SM80 BF16 lightning-indexer backward.

Thresholds follow the SM100 kernel's own test suite: dq/dk < 1e-2, dw < 1e-3.
ChunkPipe cases (chunk_offset > 0, s_kv > s_q) are covered explicitly.
"""

import pytest
import torch

from lightning_indexer_sm80 import (
    bf16_mqa_logits_bwd,
    causal_kv_range,
    ref_bf16_mqa_logits_bwd,
)

NUM_HEADS = 32
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_inputs(s_q, s_kv, topk, chunk_offset=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, NUM_HEADS, device="cuda", dtype=torch.float32, generator=g)
    gl = torch.randn(s_q, topk, device="cuda", dtype=torch.float32, generator=g)
    ks, ke = causal_kv_range(s_q, chunk_offset, "cuda")
    ke = ke.clamp_max(s_kv)
    # sample distinct in-window indices per token, padding the tail with -1
    limit = (ke - ks).clamp_min(0)
    rnd = torch.rand(s_q, topk, device="cuda", generator=g)
    idx = (rnd * limit.unsqueeze(1).float()).floor().int() + ks.unsqueeze(1)
    idx = torch.where(torch.arange(topk, device="cuda")[None, :] < limit[:, None], idx,
                      torch.full_like(idx, -1))
    return q, k, w, gl, ks, ke, idx


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _check(s_q, s_kv, topk, chunk_offset=0, seed=0):
    q, k, w, gl, ks, ke, idx = _make_inputs(s_q, s_kv, topk, chunk_offset, seed)
    got = bf16_mqa_logits_bwd(gl, q, k, w, ks, ke, idx)
    ref = ref_bf16_mqa_logits_bwd(gl, q, k, w, ks, ke, idx)
    errs = {}
    for name, tol, a, b in zip(("d_q", "d_k", "d_w"), (1e-2, 1e-2, 1e-3), got, ref):
        assert torch.isfinite(a).all(), f"{name} has non-finite entries"
        errs[name] = _rel(a, b)
        assert errs[name] < tol, f"{name} relative error {errs[name]:.3e} exceeds {tol:.1e}"
    return errs


@pytest.mark.parametrize("s_q,s_kv,topk", [(64, 64, 64), (128, 128, 64), (256, 256, 128),
                                          (512, 512, 256)])
def test_causal(s_q, s_kv, topk):
    _check(s_q, s_kv, topk)


@pytest.mark.parametrize("chunk_idx", [1, 2, 3])
def test_chunkpipe_offsets(chunk_idx):
    chunksize = 128
    _check(chunksize, chunksize * (chunk_idx + 1), 128,
           chunk_offset=chunksize * chunk_idx, seed=chunk_idx)


@pytest.mark.parametrize("s_q,s_kv,topk", [(1, 1, 64), (7, 130, 64), (100, 300, 192)])
def test_ragged(s_q, s_kv, topk):
    _check(s_q, s_kv, topk, seed=11)


def test_all_indices_invalid():
    """A fully padded top-k row must produce exactly zero gradients."""
    s_q, s_kv, topk = 32, 64, 64
    q, k, w, gl, ks, ke, _ = _make_inputs(s_q, s_kv, topk)
    idx = torch.full((s_q, topk), -1, dtype=torch.int32, device="cuda")
    dq, dk, dw = bf16_mqa_logits_bwd(gl, q, k, w, ks, ke, idx)
    assert dq.abs().max() == 0 and dk.abs().max() == 0 and dw.abs().max() == 0


if __name__ == "__main__":
    import sys
    import time

    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    for args, name in [((512, 512, 256), "causal 512"),
                       ((128, 512, 128, 384), "chunkpipe chunk 4 of 4"),
                       ((100, 300, 192), "ragged")]:
        try:
            errs = _check(*args)
            print(f"[OK]   {name}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")

    for s, topk in [(1024, 512), (2048, 1024)]:
        q, k, w, gl, ks, ke, idx = _make_inputs(s, s, topk)
        for _ in range(3):
            bf16_mqa_logits_bwd(gl, q, k, w, ks, ke, idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            bf16_mqa_logits_bwd(gl, q, k, w, ks, ke, idx)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        # three GEMMs of s_q * heads * topk * head_dim MACs each
        tflops = 3 * s * NUM_HEADS * topk * HEAD_DIM * 2 / (ms * 1e-3) / 1e12
        print(f"s_q=s_kv={s} topk={topk}: {ms:.3f} ms  {tflops:.1f} TFLOP/s (bf16 peak ~312)")
    sys.exit(1 if failures else 0)