// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> sparse_prefill_fwd_sm80(const at::Tensor& q, const at::Tensor& kv,
                                                const at::Tensor& indices, double sm_scale,
                                                int64_t d_v, int64_t q_start_index_s,
                                                bool write_p_out);

std::vector<at::Tensor> sparse_prefill_bwd_sm80(const at::Tensor& q, const at::Tensor& kv,
                                                const at::Tensor& out, const at::Tensor& grad_out,
                                                const at::Tensor& indices, const at::Tensor& lse,
                                                double sm_scale, int64_t q_start_index_s);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM80 (Ampere) sparse MLA for fused DeepSeek Sparse Attention";
    m.def("sparse_prefill_fwd_sm80", &sparse_prefill_fwd_sm80, py::arg("q"), py::arg("kv"),
          py::arg("indices"), py::arg("sm_scale"), py::arg("d_v"), py::arg("q_start_index_s"),
          py::arg("write_p_out"));
    m.def("sparse_prefill_bwd_sm80", &sparse_prefill_bwd_sm80, py::arg("q"), py::arg("kv"),
          py::arg("out"), py::arg("grad_out"), py::arg("indices"), py::arg("lse"),
          py::arg("sm_scale"), py::arg("q_start_index_s"));
}