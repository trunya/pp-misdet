import torch, json
import sys
sys.path.insert(0, '/scratch/tprasad6/PP-MisDet/claude/src')
from transformers import AutoTokenizer, SiglipProcessor
from dataset import DGM4Dataset
from model import PPMisDetModel
import torch.nn.functional as F

MODELS = '/scratch/tprasad6/PP-MisDet/claude/models'
CKPT   = '/scratch/tprasad6/PP-MisDet/claude/checkpoints/run_49387879/best.pt'
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

manip_scores = []
with torch.no_grad():
    for batch in loader:
        out = model(batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device))
        mask = batch['label'] == 1
        if mask.sum() > 0:
            scores = torch.sigmoid(out['ground_logits'][mask]).cpu()
            manip_scores.append(scores.max(dim=1).values)

all_scores = torch.cat(manip_scores)
for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
    nonempty = (all_scores > t).sum().item()
    print(f'threshold {t}: {nonempty}/{len(all_scores)} manipulated samples have grounding')