#!/bin/bash
# ============================================================
# setup_env.sh
# Run ONCE on a login node or lightwork interactive session:
#   interactive -p lightwork --mem=10G
#   bash claude/jobs/setup_env.sh
# ============================================================

set -e

SCRATCH=/scratch/tprasad6
PROJECT=$SCRATCH/PP-MisDet/claude

echo ">>> Creating directories..."

mkdir -p $PROJECT/logs
mkdir -p $PROJECT/checkpoints
mkdir -p $PROJECT/submission

echo ">>> Loading mamba..."
module load mamba/latest

echo ">>> Creating conda environment: misdet"
mamba create -n misdet python=3.10 -y
source activate misdet

echo ">>> Installing PyTorch (CUDA 12.1)..."
pip install torch==2.2.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo ">>> Installing ML dependencies..."
pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    accelerate==0.29.0 \
    bitsandbytes==0.43.0 \
    open-clip-torch==2.24.0 \
    timm==0.9.16 \
    einops \
    pandas \
    scikit-learn \
    tqdm \
    wandb \
    Pillow

echo ">>> Downloading Qwen2-VL-7B weights to scratch..."
hf download Qwen/Qwen2-VL-7B-Instruct \
    --local-dir $PROJECT/claude/models/qwen2vl

echo ">>> Downloading SigLIP weights..."
hf download google/siglip-so400m-patch14-384 \
    --local-dir $PROJECT/claude/models/siglip

echo ">>> Downloading DINOv2 weights..."
hf download facebook/dinov2-large \
    --local-dir $PROJECT/claude/models/dinov2

echo ""
echo "=== Setup complete ==="
echo "Scratch project root: $PROJECT"
echo "Activate env with:    module load mamba/latest && source activate misdet"
