# Checkpoint Conversion

Offline bidirectional conversion between **HuggingFace** and **Megatron-Core
(mcore)** checkpoint formats, for both LLMs and VLMs. Also handles MoE expert
merging and FP8 (bf16 ↔ fp8) conversion.

## Entry Point

`module_convertor/model.py` orchestrates conversion. For LLMs (single step):

```bash
python tools/convert_checkpoint/module_convertor/model.py \
    --load_platform=huggingface --save_platform=mcore \
    --config_file=<yaml> --convert_file=<json> \
    --tensor_model_parallel_size=N --pipeline_model_parallel_size=M \
    --load_ckpt_path=<hf_path> --save_ckpt_path=<mcore_path>
```

For VLMs, convert the language model, vision encoder, and adapter/projector
separately, then merge with `mcore/merge_megatron.py`. Ready-made scripts live
under `examples/<model>/checkpoint_convert/`.

## Layout

| Path | Purpose |
|---|---|
| `module_convertor/` | Conversion orchestration (`model.py`) plus adapter / vision-patch helpers. |
| `huggingface/` | HuggingFace checkpoint/config readers and writers, incl. compressed-tensor (de)quant. |
| `mcore/` | Megatron-Core checkpoint/config handling, MoE support, and `merge_megatron.py` / expert merging. |
| `key_mappings/` | Parameter-name mapping between omni/vanilla layouts and key reversers. |
| `common/` | Shared abstract checkpoint/config base classes and constants. |
| `utils/` | Checkpoint, config, and ETP/EP mapping utilities. |
| `kimi_k3/` | Model-specific weight transforms for Kimi-K3. |
| `arguments.py` | Shared CLI argument definitions. |

Additional tooling: `mcore/merge_megatron_expert.py` for MoE expert merging and
FP8 conversion support.
