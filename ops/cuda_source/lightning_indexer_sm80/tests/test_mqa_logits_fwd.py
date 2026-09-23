# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and performance tests for the SM80 BF16 lightning-indexer forward.

Covers the ChunkPipe-relevant cases explicitly: chunk_offset > 0 and s_kv > s_q.
Under ChunkPipe a query chunk of length `chunksize` attends to the concatenation
of all chunks seen so far, so s_kv grows while s_q stays fixed and the causal
window is shifted by chunk_offset.
"""

import itertools

import pytest
import torch

from lightning_indexer_sm80 import bf16_mqa_logits, causal_kv_range, ref_bf16_mqa_logits

NUM_HEADS = 32
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_inputs(s_q, s_kv, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, NUM_HEADS, device="cuda", dtype=torch.float32, generator=g)
    return q, k, w


def _compare(out, ref, tol=1e-2):
    """Relative error over the finite entries, plus exact agreement of the mask."""
    finite_ref = torch.isfinite(ref)
    finite_out = torch.isfinite(out)
    assert torch.equal(finite_ref, finite_out), "mask mismatch (-inf positions differ)"
    a = out[finite_ref].float()
    b = ref[finite_ref].float()
    denom = b.abs().max().clamp_min(1e-6)
    err = (a - b).abs().max() / denom
    assert err < tol, f"max relative error {err:.3e} exceeds {tol:.1e}"
    return err.item()


# --- causal, chunk_offset = 0 (plain pretrain, first chunk) -----------------
@pytest.mark.parametrize("s_q,s_kv", [(128, 128), (256, 256), (512, 512), (1024, 1024)])
def test_causal_aligned(s_q, s_kv):
    q, k, w = _make_inputs(s_q, s_kv)
    ks, ke = causal_kv_range(s_q, 0, "cuda")
    _compare(bf16_mqa_logits(q, k, w, ks, ke), ref_bf16_mqa_logits(q, k, w, ks, ke))


# --- ChunkPipe: chunk_offset > 0 and s_kv > s_q -----------------------------
@pytest.mark.parametrize(
    "chunksize,chunk_idx",
    list(itertools.product([128, 256], [1, 2, 3])),
)
def test_chunkpipe_offsets(chunksize, chunk_idx):
    s_q = chunksize
    chunk_offset = chunksize * chunk_idx
    s_kv = chunk_offset + chunksize
    q, k, w = _make_inputs(s_q, s_kv, seed=chunk_idx)
    ks, ke = causal_kv_range(s_q, chunk_offset, "cuda")
    _compare(bf16_mqa_logits(q, k, w, ks, ke), ref_bf16_mqa_logits(q, k, w, ks, ke))


# --- ragged shapes: not multiples of the tile ------------------------------
@pytest.mark.parametrize("s_q,s_kv", [(1, 1), (7, 13), (100, 300), (130, 4096), (250, 137)])
def test_ragged(s_q, s_kv):
    q, k, w = _make_inputs(s_q, s_kv, seed=7)
    ks, ke = causal_kv_range(s_q, 0, "cuda")
    ke = ke.clamp_max(s_kv)
    _compare(bf16_mqa_logits(q, k, w, ks, ke), ref_bf16_mqa_logits(q, k, w, ks, ke))


# --- arbitrary windows (non-zero k_start), as used by packed sequences ------
def test_windowed():
    s_q, s_kv = 256, 512
    q, k, w = _make_inputs(s_q, s_kv, seed=3)
    ks = (torch.arange(s_q, device="cuda") // 64 * 64).int()
    ke = (ks + 128).clamp_max(s_kv).int()
    _compare(bf16_mqa_logits(q, k, w, ks, ke), ref_bf16_mqa_logits(q, k, w, ks, ke))


# --- fully-masked tiles must still produce -inf, not garbage ---------------
def test_all_masked_rows():
    s_q, s_kv = 256, 1024
    q, k, w = _make_inputs(s_q, s_kv, seed=5)
    ks = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    ke = torch.zeros(s_q, dtype=torch.int32, device="cuda")
    out = bf16_mqa_logits(q, k, w, ks, ke)
    assert torch.isneginf(out).all()


if __name__ == "__main__":
    import sys

    import time

    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    for s_q, s_kv, off, name in [
        (1024, 1024, 0, "causal 1k"),
        (512, 2048, 1536, "chunkpipe chunk 3 of 4"),
        (130, 4096, 0, "ragged"),
    ]:
        q, k, w = _make_inputs(s_q, s_kv)
        ks, ke = causal_kv_range(s_q, off, "cuda")
        ke = ke.clamp_max(s_kv)
        try:
            err = _compare(bf16_mqa_logits(q, k, w, ks, ke), ref_bf16_mqa_logits(q, k, w, ks, ke))
            print(f"[OK]   {name}: rel_err={err:.3e}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")

    # throughput
    for s in [2048, 4096]:
        q, k, w = _make_inputs(s, s)
        ks, ke = causal_kv_range(s, 0, "cuda")
        for _ in range(5):
            bf16_mqa_logits(q, k, w, ks, ke)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            bf16_mqa_logits(q, k, w, ks, ke)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        # causal halves the work
        tflops = s * NUM_HEADS * s * HEAD_DIM * 2 / 2 / (ms * 1e-3) / 1e12
        print(f"s_q=s_kv={s}: {ms:.3f} ms  {tflops:.1f} TFLOP/s (bf16 peak ~312)")
    sys.exit(1 if failures else 0)