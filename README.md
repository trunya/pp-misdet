# PP-MisDet: Semantic Misinformation Detection on DGM⁴+

Official code for our submission to the **PP-MisDet Challenge @ CVPR 2026** on the DGM⁴+ dataset.

> **Val F1: 0.9654** | Trained on 4× A100 80GB

## Overview

We present a multimodal misinformation detection system that jointly addresses three tasks:
1. **Binary classification** — authentic vs. manipulated + confidence score
2. **Manipulation type** — `text_swap` vs. `text_attribute`
3. **Text grounding** — which token indices in the caption are misleading

**Scoring:** `Score = 0.5 × F1_binary + 0.3 × F1_text + 0.2 × F1_type`

## Architecture

- **SigLIP** (ViT-SO400M) — cross-modal image + text embeddings
- **DINOv2** (ViT-L) — visual foreground/background consistency
- **Fusion MLP** — concat + projection to 1024-dim shared space
- **Qwen2-VL-7B** — LoRA fine-tuned backbone (r=16, α=32, 0.12% trainable params)
- **Three task heads** — binary, type, grounding

## Repository Structure

```
PP-MisDet/
│   ├── src/
│   │   ├── dataset.py        # DGM4+ dataloader
│   │   ├── model.py          # Full model architecture
│   │   ├── train.py          # Training script (torchrun DDP)
│   │   ├── infer.py          # Inference + submission generation
│   │   ├── qualitative.py    # Qualitative example extraction
│   │   └── qualitative2.py   # Filtered qualitative examples
│   └── jobs/
│       ├── train.sh          # SLURM training job (4× A100)
│       ├── infer.sh          # SLURM inference job
│       ├── qualitative.sh    # SLURM qualitative job
│       └── setup_env.sh      # Environment setup script
└── requirements.txt
```

## Setup

### 1. Environment

```bash
module load mamba/latest
mamba create -n misdet python=3.10 -y
conda activate misdet

pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install git+https://github.com/huggingface/transformers.git@v4.45.0
pip install peft==0.10.0 accelerate==0.29.0
pip install open-clip-torch timm einops pandas scikit-learn tqdm Pillow
```

### 2. Download Model Weights

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir claude/models/qwen2vl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir claude/models/siglip
huggingface-cli download facebook/dinov2-large --local-dir claude/models/dinov2
```

### 3. Dataset

Download the DGM⁴+ dataset and extract to:
```
data/
├── train/
│   ├── images/
│   └── train.json
├── val/
│   ├── images/
│   └── val.json
└── test/
    ├── images/
    └── test_public.json
```

## Training

```bash
# On ASU Sol Supercomputer
sbatch claude/jobs/train.sh

# Or directly
cd /path/to/PP-MisDet
torchrun --nproc_per_node=4 claude/src/train.py \
    --model_path claude/models/qwen2vl \
    --siglip_path claude/models/siglip \
    --dinov2_path claude/models/dinov2 \
    --data_root data \
    --output_dir checkpoints/run1 \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lr 2e-4 \
    --fp16
```

## Inference

```bash
sbatch claude/jobs/infer.sh

# Or directly
python claude/src/infer.py \
    --model_path claude/models/qwen2vl \
    --siglip_path claude/models/siglip \
    --dinov2_path claude/models/dinov2 \
    --checkpoint checkpoints/run1/best.pt \
    --data_root data \
    --output_dir submission \
    --ground_thresh 0.3

cd submission && zip submission.zip classification.csv text_grounding.json
```

## Results

| Epoch | Val Loss | Val Acc | Val F1 |
|-------|----------|---------|--------|
| 1 | 0.1669 | 92.93% | 0.9413 |
| 2 | 0.1164 | 93.07% | 0.9427 |
| 3 | 0.1094 | 94.80% | 0.9563 |
| **4** | **0.0942** | **95.87%** | **0.9654** |
| 5 | 0.0973 | 95.47% | 0.9626 |

## Hardware

- 4× NVIDIA A100 80GB (ASU Sol Supercomputer)
- Training time: ~6 minutes

## Citation

```bibtex
@misc{singh2025dgm4datasetextensionglobal,
    title={DGM4+: Dataset Extension for Global Scene Inconsistency},
    author={Gagandeep Singh and Samudi Amarsinghe and Priyanka Singh and Xue Li},
    year={2025},
    eprint={2509.26047},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.26047},
}
```

## Acknowledgements

Thanks to **Bharatesh C.** for introducing me to this challenge, and to ASU Research Computing for access to the Sol Supercomputer.
