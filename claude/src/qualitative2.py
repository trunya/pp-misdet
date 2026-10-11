import torch, sys
sys.path.insert(0, '/scratch/tprasad6/PP-MisDet/claude/src')
from transformers import AutoTokenizer, SiglipProcessor
from dataset import DGM4Dataset
from model import PPMisDetModel

MODELS = '/scratch/tprasad6/PP-MisDet/claude/models'
CKPT   = '/scratch/tprasad6/PP-MisDet/claude/checkpoints/run_65494226/best.pt'
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

type_map = {0: 'origin', 1: 'text_swap', 2: 'text_attribute'}
found_auth = 0
found_manip = 0

with torch.no_grad():
    for batch in loader:
        out = model(batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device))
        pred_label = out['binary_logits'].argmax().item()
        pred_type  = out['type_logits'].argmax().item()
        gt_label   = batch['label'].item()
        gt_type    = batch['manip_type'].item()
        caption    = batch['caption'][0]
        sample_id  = batch['id'][0]
        tokens     = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
        scores     = torch.sigmoid(out['ground_logits'][0]).cpu()
        pred_tokens = (scores > 0.3).nonzero(as_tuple=True)[0].tolist()
        gt_tokens   = batch['token_labels'][0].nonzero(as_tuple=True)[0].tolist()

        # look up record by id
        rec = next((r for r in ds.records if str(r['id']) == sample_id), None)
        if rec is None:
            continue

        # good authentic example
        if found_auth < 2 and gt_label == 0 and pred_label == 0 and len(caption.split()) > 6:
            print(f'=== AUTHENTIC ===')
            print(f'ID: {sample_id}')
            print(f'Image: {rec["image"]}')
            print(f'Caption: {caption}')
            print()
            found_auth += 1

        # good manipulated example
        if found_manip < 2 and gt_label == 1 and pred_label == 1 and pred_type == gt_type and len(pred_tokens) <= 4 and pred_tokens:
            print(f'=== MANIPULATED ===')
            print(f'ID: {sample_id}')
            print(f'Image: {rec["image"]}')
            print(f'Caption: {caption}')
            print(f'GT type: {type_map[gt_type]}')
            print(f'Pred type: {type_map[pred_type]} ✓')
            print(f'GT grounded tokens: {[tokens[i] for i in gt_tokens]}')
            print(f'Pred grounded tokens: {[tokens[i] for i in pred_tokens]}')
            print()
            found_manip += 1

        if found_auth >= 2 and found_manip >= 2:
            break
