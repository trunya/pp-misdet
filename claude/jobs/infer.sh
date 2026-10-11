#!/bin/bash
# ============================================================
# infer.sh
# Submit with: sbatch claude/jobs/infer.sh --checkpoint <path>
# Or hardcode CKPT below after training.
# ============================================================
#SBATCH -J misdet_infer
#SBATCH -p htc
#SBATCH -q public
#SBATCH -o /scratch/tprasad6/PP-MisDet/claude/logs/infer_%j.out
#SBATCH -e /scratch/tprasad6/PP-MisDet/claude/logs/infer_%j.err
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ── paths ──────────────────────────────────────────────────
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/PP-MisDet

# Set checkpoint path here or pass via --export CKPT=... at sbatch time
CKPT=${CKPT:-/scratch/tprasad6/PP-MisDet/claude/checkpoints/run_65494226/best.pt}
# ── environment ────────────────────────────────────────────
module load mamba/latest
source activate misdet

export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Checkpoint   : $CKPT"
echo "Start time   : $(date)"

# ── inference ──────────────────────────────────────────────
cd $PROJECT

python /scratch/tprasad6/PP-MisDet/claude/src/infer.py \
    --model_path  $PROJECT/claude/models/qwen2vl \
    --siglip_path $PROJECT/claude/models/siglip \
    --dinov2_path $PROJECT/claude/models/dinov2 \
    --checkpoint  $CKPT \
    --data_root   $SCRATCH/PP-MisDet/data \
    --output_dir  $PROJECT/claude/submission \
    --batch_size  8 \
    --ground_thresh 0.3

# ── package submission ─────────────────────────────────────
echo ">>> Packaging submission..."
cd $PROJECT/claude/submission
zip -j submission1.zip classification1.csv text_grounding1.json

echo ">>> Done. Submission at: $PROJECT/claude/submission/submission1.zip"
echo "End time: $(date)"
