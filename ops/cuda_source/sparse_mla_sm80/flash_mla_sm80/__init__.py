# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SM80 (Ampere) sparse MLA kernels for the fused-DSA path."""

from .interface import (
    flash_mla_sparse_bwd_sm80,
    flash_mla_sparse_fwd_sm80,
    ref_sparse_mla_fwd,
)

__all__ = [
    "flash_mla_sparse_fwd_sm80",
    "flash_mla_sparse_bwd_sm80",
    "ref_sparse_mla_fwd",
]