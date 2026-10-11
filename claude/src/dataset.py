"""
dataset.py
Loads DGM4+ image-caption pairs for train / val / test splits.

Actual data layout:
    data/
    ├── train/
    │   ├── images/       # jpg files e.g. cf547d44c3.jpg
    │   └── train.json    # list of dicts
    ├── val/
    │   ├── images/
    │   └── val.json
    └── test/
        ├── images/
        └── test_public.json

JSON record format (train/val):
    {
        "id":            100796,
        "image":         "cf547d44c3.jpg",
        "text":          "A man stands ...",
        "fake_cls":      "origin" | "text_swap" | "text_attribute",
        "fake_text_pos": [11],          # token indices; [] if authentic
        "fake_image_box": [...],        # unused for now
        "mtcnn_boxes":   [...]          # unused for now
    }

Test records are identical but without fake_cls / fake_text_pos.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# origin counts as authentic (label=0), swap/attr are manipulated (label=1)
FAKE_CLS2LABEL = {"origin": 0, "text_swap": 1, "text_attribute": 1}
FAKE_CLS2TYPE  = {"origin": 0, "text_swap": 1, "text_attribute": 2}

# for decoding predictions back to strings
ID2LABEL = {0: "authentic", 1: "manipulated"}
ID2TYPE  = {0: "origin",    1: "text_swap",   2: "text_attribute"}


class DGM4Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,                  # 'train' | 'val' | 'test'
        image_transform=None,
        tokenizer=None,
        max_tokens: int = 64,
    ):
        self.split      = split
        self.image_dir  = Path(data_root) / split / "images"
        self.transform  = image_transform
        self.tokenizer  = tokenizer
        self.max_tokens = max_tokens
        self.is_test    = split == "test"

        # ── load JSON ─────────────────────────────────────
        json_names = {
            "train": "train.json",
            "val":   "val.json",
            "test":  "test_public.json",
        }
        json_path = Path(data_root) / split / json_names[split]
        with open(json_path) as f:
            self.records = json.load(f)

        print(f"[DGM4Dataset] {split}: {len(self.records)} samples from {json_path}")

    def __len__(self):
        return len(self.records)

    def _load_image(self, filename: str) -> Image.Image:
        # filename may be a full relative path like data/manipulation/text_swap/abc.jpg
        # actual images are all flat in images/
        p = self.image_dir / Path(filename).name
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx):
        rec       = self.records[idx]
        sample_id = str(rec["id"])
        caption   = rec["text"]
        filename  = rec["image"]

        # ── image ─────────────────────────────────────────
        image = self._load_image(filename)
        if self.transform:
            image = self.transform(image)

        # ── caption tokenization ──────────────────────────
        if self.tokenizer:
            enc = self.tokenizer(
                caption,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).squeeze(0)
        else:
            input_ids      = torch.zeros(self.max_tokens, dtype=torch.long)
            attention_mask = torch.zeros(self.max_tokens, dtype=torch.long)

        item = {
            "id":             sample_id,
            "image":          image,
            "caption":        caption,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            fake_cls = rec["fake_cls"]

            # ── binary label ──────────────────────────────
            item["label"]      = torch.tensor(FAKE_CLS2LABEL[fake_cls], dtype=torch.long)
            item["manip_type"] = torch.tensor(FAKE_CLS2TYPE[fake_cls],  dtype=torch.long)

            # ── token-level grounding ─────────────────────
            token_labels = torch.zeros(self.max_tokens, dtype=torch.float)
            for tok_idx in rec.get("fake_text_pos", []):
                if tok_idx < self.max_tokens:
                    token_labels[tok_idx] = 1.0
            item["token_labels"] = token_labels

        return item


def build_dataloader(data_root, split, image_transform, tokenizer, batch_size, num_workers=4, shuffle=None):
    dataset = DGM4Dataset(
        data_root=data_root,
        split=split,
        image_transform=image_transform,
        tokenizer=tokenizer,
    )
    if shuffle is None:
        shuffle = (split == "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )