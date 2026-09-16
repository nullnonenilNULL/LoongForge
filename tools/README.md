# LoongForge Tools

Offline utilities that support the LoongForge training workflow. Top-level
directories are organized **by function**; where a tool is model/domain-specific
it lives under a domain subdirectory (e.g. `data_preprocess/embodied/`).

## Layout

| Path | Purpose |
|---|---|
| [`convert_checkpoint/`](./convert_checkpoint/) | Offline bidirectional **HuggingFace ↔ Megatron-Core** checkpoint conversion (LLM & VLM), MoE expert merging, and FP8 conversion. |
| [`dist_checkpoint/`](./dist_checkpoint/) | Online distributed **HuggingFace load/save** used by the training loop, plus the **DCP → safetensors** consolidator. |
| [`data_preprocess/`](./data_preprocess/) | Dataset preparation and generation for [`llm/`](./data_preprocess/llm/), [`vlm/`](./data_preprocess/vlm/), and [`embodied/`](./data_preprocess/embodied/) training. |
| [`adaptive_fp8/`](./adaptive_fp8/) | Benchmarks Transformer Engine parallel Linear layers (BF16 / FP8) and exports the **adaptive-FP8 policy file** consumed by training (`benchmark_te_parallel_layers.py`). |

## Conventions

- Most tools run directly by path from the repository root, e.g.
  `python tools/<tool>/<script>.py ...`.
- Some tools are also importable as packages (e.g. `tools.dist_checkpoint`,
  `tools.convert_checkpoint`); the repository root must be on `PYTHONPATH` for
  those imports.
- Standalone pipelines with their own dependency set (e.g.
  [`data_preprocess/embodied/ego2robot`](./data_preprocess/embodied/ego2robot/))
  document their own installation and invocation in their local README.
