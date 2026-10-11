"""
infer.py
Runs inference on the test split and writes:
    submission/classification.csv
    submission/text_grounding.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, SiglipProcessor
from tqdm import tqdm

from dataset import DGM4Dataset
from model import PPMisDetModel

ID2LABEL = {0: "authentic", 1: "manipulated"}
ID2TYPE  = {0: "origin",    1: "text_swap",   2: "text_attribute"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  required=True)
    p.add_argument("--siglip_path", required=True)
    p.add_argument("--dinov2_path", required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--max_tokens",  type=int, default=64)
    p.add_argument("--ground_thresh", type=float, default=0.3,
                   help="Sigmoid threshold for grounding token selection")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, ground_thresh):
    model.eval()

    cls_rows   = []
    grounding  = {}

    for batch in tqdm(loader, desc="Inference"):
        pixel_values = batch["image"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        ids          = batch["id"]

        outputs = model(pixel_values, input_ids, attn_mask)

        # ── binary classification ──────────────────────────
        bin_probs   = F.softmax(outputs["binary_logits"], dim=-1)  # (B, 2)
        confidence  = bin_probs[:, 1].cpu().tolist()               # P(manipulated)
        pred_labels = outputs["binary_logits"].argmax(dim=-1).cpu().tolist()

        # ── manipulation type ──────────────────────────────
        type_preds  = outputs["type_logits"].argmax(dim=-1).cpu().tolist()

        # ── grounding ──────────────────────────────────────
        ground_probs = torch.sigmoid(outputs["ground_logits"]).cpu()  # (B, T)

        for i, sample_id in enumerate(ids):
            label = ID2LABEL[pred_labels[i]]
            mtype = ID2TYPE[type_preds[i]] if label == "manipulated" else "origin"
            conf  = confidence[i] if label == "manipulated" else (1.0 - confidence[i])

            cls_rows.append({
                "id":               sample_id,
                "label":            label,
                "confidence":       round(conf, 6),
                "manipulation_type": mtype,
            })

            # grounding: empty list for authentic, token indices for manipulated
            if label == "manipulated":
                token_indices = (ground_probs[i] > ground_thresh).nonzero(as_tuple=True)[0].tolist()
            else:
                token_indices = []
            grounding[sample_id] = token_indices

    return cls_rows, grounding


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── tokenizer + transform ─────────────────────────────
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.siglip_path)
    processor = SiglipProcessor.from_pretrained(args.siglip_path)

    def image_transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    # ── dataset ───────────────────────────────────────────
    test_ds = DGM4Dataset(
        args.data_root, "test", image_transform, tokenizer, args.max_tokens
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────
    print("Loading model and checkpoint...")
    model = PPMisDetModel(
        siglip_path=args.siglip_path,
        dinov2_path=args.dinov2_path,
        qwen_path=args.model_path,
        max_tokens=args.max_tokens,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── inference ─────────────────────────────────────────
    cls_rows, grounding = run_inference(model, test_loader, device, args.ground_thresh)

    # ── write outputs ─────────────────────────────────────
    cls_path   = Path(args.output_dir) / "classification.csv"
    ground_path = Path(args.output_dir) / "text_grounding.json"

    pd.DataFrame(cls_rows).to_csv(cls_path, index=False)
    with open(ground_path, "w") as f:
        json.dump(grounding, f, indent=2)

    print(f"Wrote {len(cls_rows)} predictions → {cls_path}")
    print(f"Wrote grounding → {ground_path}")
    print("Done.")


if __name__ == "__main__":
    main()