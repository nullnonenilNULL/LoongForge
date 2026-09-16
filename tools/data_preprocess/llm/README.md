# LLM Data Preprocessing

Tokenize and pack raw **text** corpora into the indexed binary format consumed
by LoongForge / Megatron LLM training.

| Script | Purpose |
|---|---|
| `preprocess_pretrain_data.py` | Process large pretraining corpora into tokenized, indexed data shards. |
| `preprocess_sft_data.py` | Process supervised fine-tuning (SFT) conversations into training-ready samples. |

## Usage

```bash
# Pretraining corpus
python tools/data_preprocess/llm/preprocess_pretrain_data.py --help

# SFT data
python tools/data_preprocess/llm/preprocess_sft_data.py --help
```

Run each script with `--help` for the full argument list (tokenizer, input JSON,
output prefix, worker count, sequence length, etc.). The resulting data prefix is
passed to training via `--data-path`.
