"""
train.py
Training entry point for PP-MisDet.
Launched via torchrun (see claude/jobs/train.sh).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, SiglipProcessor
from tqdm import tqdm

from dataset import DGM4Dataset, FAKE_CLS2LABEL, FAKE_CLS2TYPE
from model import PPMisDetModel, JointLoss


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",    required=True)
    p.add_argument("--siglip_path",   required=True)
    p.add_argument("--dinov2_path",   required=True)
    p.add_argument("--data_root",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lora_r",        type=int,   default=16)
    p.add_argument("--lora_alpha",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--loss_w_binary", type=float, default=0.5)
    p.add_argument("--loss_w_ground", type=float, default=0.3)
    p.add_argument("--loss_w_type",   type=float, default=0.2)
    p.add_argument("--fp16",          action="store_true")
    p.add_argument("--max_tokens",    type=int,   default=64)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def log(msg):
    if is_main():
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, grad_accum, fp16, epoch):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main())):
        pixel_values  = batch["image"].to(device)
        input_ids     = batch["input_ids"].to(device)
        attn_mask     = batch["attention_mask"].to(device)
        labels        = batch["label"].to(device)
        manip_types   = batch["manip_type"].to(device)
        token_labels  = batch["token_labels"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model(pixel_values, input_ids, attn_mask)
            loss, loss_parts = criterion(outputs, labels, manip_types, token_labels)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

        if is_main() and step % 50 == 0:
            log(f"  step {step:4d} | loss {loss.item()*grad_accum:.4f} | "
                f"L_bin {loss_parts['L_binary']:.4f} | "
                f"L_gnd {loss_parts['L_ground']:.4f} | "
                f"L_typ {loss_parts['L_type']:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, fp16):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val", disable=not is_main()):
        pixel_values = batch["image"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        labels       = batch["label"].to(device)
        manip_types  = batch["manip_type"].to(device)
        token_labels = batch["token_labels"].to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model(pixel_values, input_ids, attn_mask)
            loss, _ = criterion(outputs, labels, manip_types, token_labels)

        total_loss += loss.item()
        preds = outputs["binary_logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import f1_score, accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return total_loss / len(loader), acc, f1


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── tokenizer + transform ─────────────────────────────
    log("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.siglip_path)
    processor = SiglipProcessor.from_pretrained(args.siglip_path)

    def image_transform(img):
        return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    # ── datasets ──────────────────────────────────────────
    log("Building dataloaders...")
    train_ds = DGM4Dataset(args.data_root, "train", image_transform, tokenizer, args.max_tokens)
    val_ds   = DGM4Dataset(args.data_root, "val",   image_transform, tokenizer, args.max_tokens)

    train_sampler = DistributedSampler(train_ds)
    train_loader  = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────
    log("Initializing model...")
    model = PPMisDetModel(
        siglip_path=args.siglip_path,
        dinov2_path=args.dinov2_path,
        qwen_path=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_tokens=args.max_tokens,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── optimizer + loss ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    criterion = JointLoss(args.loss_w_binary, args.loss_w_ground, args.loss_w_type)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ── training ──────────────────────────────────────────
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            criterion, device, args.grad_accum, args.fp16, epoch,
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, args.fp16)
        scheduler.step()

        log(f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}")

        if is_main() and val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = Path(args.output_dir) / "best.pt"
            torch.save(model.module.state_dict(), ckpt_path)
            log(f"  ✓ Saved best checkpoint (f1={best_f1:.4f}) → {ckpt_path}")

    log("Training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()