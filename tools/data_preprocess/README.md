# Data Preprocessing

Dataset preparation tools, organized by training modality.

| Path | Purpose |
|---|---|
| [`llm/`](./llm/) | Tokenize and pack **text** corpora for LLM pretraining and SFT. |
| [`vlm/`](./vlm/) | Convert multimodal datasets to **WebDataset** for the Energon loader, plus offline sequence packing. See [`vlm/README.md`](./vlm/README.md). |
| [`embodied/`](./embodied/) | Generate and prepare **embodied / VLA** training data — the `ego2robot` ego-video → robot pipeline and DreamZero dataset/feature precompute. |

Each subdirectory documents its own scripts and expected input/output layout in
its local README.
