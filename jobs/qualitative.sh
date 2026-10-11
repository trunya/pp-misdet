#!/bin/bash
#SBATCH -J qualitative
#SBATCH -p htc
#SBATCH -q public
#SBATCH -o /scratch/tprasad6/PP-MisDet/logs/qualitative2_%j.out
#SBATCH -e /scratch/tprasad6/PP-MisDet/logs/qualitative2_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load mamba/latest
source activate misdet

python /scratch/tprasad6/PP-MisDet/claude/src/qualitative2.py
