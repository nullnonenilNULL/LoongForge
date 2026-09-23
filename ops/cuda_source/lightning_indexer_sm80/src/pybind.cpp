// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <vector>

at::Tensor bf16_mqa_logits(const at::Tensor& q, const at::Tensor& k, const at::Tensor& weights,
                           const at::Tensor& cu_seqlen_ks, const at::Tensor& cu_seqlen_ke);

std::vector<at::Tensor> bf16_mqa_logits_bwd(
    const at::Tensor& grad_logits, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& weights, const at::Tensor& cu_seqlen_ks,
    const at::Tensor& cu_seqlen_ke, const at::Tensor& topk_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM80 BF16 lightning indexer (forward + backward)";
    m.def("bf16_mqa_logits", &bf16_mqa_logits, py::arg("q"), py::arg("k"), py::arg("weights"),
          py::arg("cu_seqlen_ks"), py::arg("cu_seqlen_ke"));
    m.def("bf16_mqa_logits_bwd", &bf16_mqa_logits_bwd, py::arg("grad_logits"), py::arg("q"),
          py::arg("k"), py::arg("weights"), py::arg("cu_seqlen_ks"), py::arg("cu_seqlen_ke"),
          py::arg("topk_indices"));
}