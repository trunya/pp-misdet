#!/bin/bash
# ============================================================
# train.sh
# Submit with: sbatch claude/jobs/train.sh
# ============================================================
#SBATCH -J misdet_train
#SBATCH -p htc
#SBATCH -q public
#SBATCH -o /scratch/%u/PP-MisDet/claude/logs/train_%j.out
#SBATCH -e /scratch/%u/PP-MisDet/claude/logs/train_%j.err
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ── paths ──────────────────────────────────────────────────
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/PP-MisDet/claude

# ── environment ────────────────────────────────────────────
module load mamba/latest
source activate misdet
echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'from transformers import Qwen2VLForConditionalGeneration; print(\"ok\")')"

export HF_HOME=$SCRATCH/hf_cache
export WANDB_DIR=$PROJECT/logs
export TOKENIZERS_PARALLELISM=false

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"

# ── training ───────────────────────────────────────────────
cd $PROJECT

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/train.py \
    --model_path      $PROJECT/models/qwen2vl \
    --siglip_path     $PROJECT/models/siglip \
    --dinov2_path     $PROJECT/models/dinov2 \
    --data_root       $SCRATCH/PP-MisDet/data \
    --output_dir      $PROJECT/checkpoints/run_$SLURM_JOB_ID \
    --epochs          5 \
    --batch_size      4 \
    --grad_accum      4 \
    --lora_r          16 \
    --lora_alpha      32 \
    --lr              2e-4 \
    --loss_w_binary   0.5 \
    --loss_w_ground   0.3 \
    --loss_w_type     0.2 \
    --fp16

echo "End time: $(date)"
