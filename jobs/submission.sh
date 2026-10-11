#!/bin/bash
#SBATCH -p general
#SBATCH -q public
#SBATCH -J misdet_infer
#SBATCH -o /scratch/%u/logs/infer_%j.out
#SBATCH -t 0-04:00:00          # fits in htc window too
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load mamba/latest
mamba activate misdet

cd /scratch/tprasad6/PP-MisDet/claude

python infer.py \
  --model_path /scratch/tprasad6/PP-MisDet/claude/models/qwen2vl \
  --checkpoint /scratch/tprasad6/PP-MisDet/claude/checkpoints/run1/best.pt \
  --test_dir /scratch/tprasad6/PP-MisDet/data/test \
  --output_dir /scratch/tprasad6/PP-MisDet/claude/submission

cd /scratch/tprasad6/PP-MisDet/claude/submission
zip submission.zip classification.csv text_grounding.json