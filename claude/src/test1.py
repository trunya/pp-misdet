import torch, json, sys
sys.path.insert(0, '/scratch/tprasad6/PP-MisDet/claude/src')
from transformers import AutoTokenizer, SiglipProcessor
from dataset import DGM4Dataset
from model import PPMisDetModel
import torch.nn.functional as F

MODELS = '/scratch/tprasad6/PP-MisDet/claude/models'
CKPT   = '/scratch/tprasad6/PP-MisDet/claude/checkpoints/run_50430766/best.pt'
device = 'cuda'

processor = SiglipProcessor.from_pretrained(f'{MODELS}/siglip')
tokenizer = AutoTokenizer.from_pretrained(f'{MODELS}/siglip')
def transform(img):
    return processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)

ds = DGM4Dataset('/scratch/tprasad6/PP-MisDet/data', 'val', transform, tokenizer)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

model = PPMisDetModel(f'{MODELS}/siglip', f'{MODELS}/dinov2', f'{MODELS}/qwen2vl').to(device)
model.load_state_dict(torch.load(CKPT, map_location=device), strict=False)
model.eval()

found = 0
with torch.no_grad():
    for batch in loader:
        if batch['label'].item() == 0:
            continue
        out = model(batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device))
        pred_label = out['binary_logits'].argmax().item()
        pred_type  = out['type_logits'].argmax().item()
        scores     = torch.sigmoid(out['ground_logits'][0]).cpu()
        pred_tokens = (scores > 0.3).nonzero(as_tuple=True)[0].tolist()

        if pred_label == 1 and pred_tokens and batch['token_labels'][0].sum() > 0:
            caption = batch['caption'][0]
            tokens  = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
            gt_tokens = batch['token_labels'][0].nonzero(as_tuple=True)[0].tolist()
            type_map = {0: 'origin', 1: 'text_swap', 2: 'text_attribute'}
            print(f'ID: {batch[\"id\"][0]}')
            print(f'Image: {ds.records[ds.records.index(next(r for r in ds.records if str(r[\"id\"]) == batch[\"id\"][0]))][\"image\"]}')
            print(f'Caption: {caption}')
            print(f'GT type: {type_map[batch[\"manip_type\"].item()]}')
            print(f'Pred type: {type_map[pred_type]}')
            print(f'GT grounded tokens: {[tokens[i] for i in gt_tokens]}')
            print(f'Pred grounded tokens: {[tokens[i] for i in pred_tokens]}')
            found += 1
            if found == 3:
                break