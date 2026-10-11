#!/bin/bash
#SBATCH -J check_thresh
#SBATCH -p public
#SBATCH -q public
#SBATCH -o /scratch/tprasad6/PP-MisDet/logs/thresh_%j.out
#SBATCH -e /scratch/tprasad6/PP-MisDet/logs/thresh_%j.err
#SBATCH -t 0-00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load mamba/latest
conda run -n misdet python -c "
import torch, sys
sys.path.insert(0, '/scratch/tprasad6/PP-MisDet/claude/src')
from transformers import AutoTokenizer, SiglipProcessor
from dataset import DGM4Dataset
from model import PPMisDetModel

MODELS = '/scratch/tprasad6/PP-MisDet/claude/models'
CKPT   = '/scratch/tprasad6/PP-MisDet/claude/checkpoints/run_50430766/best.pt'
device = 'cuda'

processor = SiglipProcessor.from_pretrained(f'{MODELS}/siglip')
tokenizer = AutoTokenizer.from_pretrained(f'{MODELS}/siglip')
def transform(img):
    return processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)

ds = DGM4Dataset('/scratch/tprasad6/PP-MisDet/data', 'val', transform, tokenizer)
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

model = PPMisDetModel(f'{MODELS}/siglip', f'{MODELS}/dinov2', f'{MODELS}/qwen2vl').to(device)
model.load_state_dict(torch.load(CKPT, map_location=device), strict=False)
model.eval()

max_scores = []
with torch.no_grad():
    for batch in loader:
        out = model(batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device))
        mask = batch['label'] == 1
        if mask.sum() > 0:
            scores = torch.sigmoid(out['ground_logits'][mask]).cpu()
            max_scores.extend(scores.max(dim=1).values.tolist())

max_scores.sort()
print(f'Total manipulated: {len(max_scores)}')
print(f'Min: {min(max_scores):.4f}  Max: {max(max_scores):.4f}')
for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    n = sum(1 for s in max_scores if s > t)
    print(f'threshold {t:.2f}: {n}/{len(max_scores)} have grounding')
"
EOF