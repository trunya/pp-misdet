#!/bin/bash
#SBATCH -p general
#SBATCH -q public
#SBATCH -J misdet_train
#SBATCH -o /scratch/%u/logs/train_%j.out
#SBATCH -e /scratch/%u/logs/train_%j.err
#SBATCH -t 1-12:00:00          # 36h — adjust if needed
#SBATCH --gres=gpu:a100:4       # 4× A100 80GB on one node
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load mamba/latest
mamba activate misdet

export SCRATCH=/scratch/$USER
export HF_HOME=$SCRATCH/hf_cache
export WANDB_DIR=$SCRATCH/wandb

cd /scratch/tprasad6/PP-MisDet/claude/pp-misdet

torchrun --nproc_per_node=4 train.py \
  --model_path $SCRATCH/models/qwen2vl \
  --data_root $SCRATCH/dgm4plus \
  --output_dir $SCRATCH/checkpoints/run1 \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lr 2e-4 \
  --loss_weights 0.5 0.3 0.2 \
  --fp16