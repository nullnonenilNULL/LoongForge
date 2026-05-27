# Quick Start: VLA Model SFT Training on Kunlunxin P800

## Quick Start: VLA Model SFT Training

This document guides you through the quick start process for fine-tuning Vision-Language-Action (VLA) models using the LoongForge framework on P800.

## 0. Resource Preparation

Before starting, download the required model weights, tokenizer, and datasets.
All downloads use HuggingFace. Install the CLI first:

```bash
pip install "huggingface_hub[cli]"
```

### 0.1 Download Model Weights

```bash
hf download lerobot/pi05_base --local-dir ./pi05_base
```

> **Note:** This model requires approximately **14.5 GB** of disk space.

### 0.2 Download Tokenizer

The tokenizer is a separate download. **Gated access** — you must log in to HuggingFace and accept the license agreement at https://huggingface.co/google/paligemma-3b-pt-224 before downloading.

```bash
hf login  # authenticate with your HF token
hf download google/paligemma-3b-pt-224 --local-dir ./paligemma-3b-pt-224
```

### 0.3 Download Dataset

We use the Libero-10 dataset in LeRobot v3.0 format for robot manipulation training.

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir ./data/libero_10
```

## 1. Checkpoint Conversion

After downloading resources in Section 0, you need to convert the HF checkpoint to torch format before training. This step is the same as the GPU version:

* **Checkpoint conversion**: Convert HuggingFace weights (from Section 0.1) to PyTorch format — see [Quick Start: Pi0.5 Training](https://loongforge.readthedocs.io/en/latest/vla_tutorial/quick_start_pi05_training.html) Section 2.1.

## 2. SFT Training Script

LoongForge currently provides SFT training example scripts for various models. After entering the container, you can find relevant scripts in the `examples_xpu/{model}/finetuning/` directory. Below is an example SFT training script for `PI 0.5`. Please refer to the comments for the purpose of each script section:

```bash
#!/usr/bin/env bash
# Pi05 sanity SFT launcher. This leverages the lightweight pi05 trainer
# (dummy data, single forward/backward) to verify the wiring inside the Omni
# framework. Adjust paths if your repo layout differs.

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}

export XMLIR_ENABLE_FAST_FC=true         # Used in torch.nn.linear.py (LinearWithActFunction, etc.)
export XMLIR_MATMUL_FAST_MODE=1          # Accelerate xblas fc computation accumulation under bf16
export XMLIR_ENABLE_LINEAR_FC_FUSION=1   # Allow linear to bypass xblas fcfusion in certain scenarios, e.g., use addmm; default is 1
export XMLIR_PARALLEL_SAVE_MEMORY=false  # false: higher memory usage but better performance; true: lower memory usage but reduced performance
export XDNN_USE_FAST_GELU=true           # High-precision gelu operator implementation
export BKCL_FORCE_SYNC=1                 # Force CPU synchronization before communication; reduces performance
export BKCL_TREE_THRESHOLD=0             # Set to 0 to disable tree algorithm
export BKCL_ENABLE_XDR=1                 # Enable XDR (XPU direct RDMA); enables direct RDMA from XPU to RDMA NIC, required for multi-node training
export BKCL_RDMA_VERBS=1                 # Used together with BKCL_QPS_PER_CONNECTION; currently only needed for Hygon machines
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4   # Adjust based on actual environment; configure according to NIC connectivity for multi-node setup
export XTE_USE_MULTI_TENSOR_ADAMW=True   # Optimizer adam aligned with GPU multi_tensor_adamw implementation

# Distributed launch (defaults single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
  --num-workers 16
)

# Core training args — pi05 trainer only needs minimal Megatron flags
TRAINING_ARGS=(
    --training-phase sft
    --micro-batch-size 12
    --global-batch-size 96
    --train-iters 50
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format torch
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --seed 1234
    --lr 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --lr-decay-iters 30000
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-eps 1e-8
    --adam-beta2 0.95
    --weight-decay 0.01
    --finetune
    --bf16
    --init-model-with-meta-device
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30
    --optimizer-cpu-offload
    --optimizer-offload-fraction 0.05
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
    --random-fallback-cpu
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
)

#if [ -n "${WANDB_API_KEY}" ]; then
#    LOGGING_ARGS+=(
#        --wandb-project ${WANDB_PROJECT}
#        --wandb-exp-name ${WANDB_NAME}
#    )
#fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

## Monitoring Logs

By default, the script outputs TensorBoard logs to the directory specified by `TENSORBOARD_PATH`. You can view training curves through TensorBoard.

Additionally, if wandb is installed, you can configure the `WANDB_API_KEY` to upload training metrics to wandb for online monitoring.