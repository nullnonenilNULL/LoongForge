# Distributed Checkpoint

Topology-aware distributed checkpoint sharding and **online HuggingFace
load/save**. Unlike `convert_checkpoint/` (an offline CLI that writes converted
checkpoints to disk), this package is imported by the training loop to load and
save HuggingFace-format weights directly under the active parallel layout — no
intermediate on-disk conversion step.

Used by the trainer, e.g. `load_hf_checkpoint_online` in
`loongforge/train/training_utils.py`.

## Layout

| Path | Purpose |
|---|---|
| `checkpoint/` | HuggingFace checkpoint loader / saver / converter and a round-trip test. |
| `core/` | Tensor-parallel gather (`tp_gather.py`), topology sharding (`topo_sharder.py`), and the argument `parser.py`. |
| `config/` | `ParallelConfig` (TP / PP / EP / etc. layout description). |
| `utils/` | Timing, memory tracking, and comparison helpers. |
| `test/` | HF checkpoint converter tests (`test_hf_checkpoint_converter.py` / `.sh`). |
| [`dcp_to_safetensors.py`](./dcp_to_safetensors.py) | Offline, single-process consolidator: merge a **DCP-sharded** checkpoint (e.g. an embodied FSDP `steps_N/dcp/`) into a single-file `model.safetensors` or `.pt`. No distributed init required. |

## Notes

- Import path is `tools.dist_checkpoint.*`; the repository root must be on
  `PYTHONPATH`.
- It reuses parameter mappings from `tools.convert_checkpoint` (e.g. common
  config, key mappings, ETP maps) rather than duplicating them.
