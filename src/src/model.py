"""
model.py
Full PP-MisDet model:
    SigLIP  (image + text embeddings)
  + DINOv2  (FG/BG visual embeddings)
  → MLP fusion
  → Qwen2-VL backbone (LoRA fine-tuned)
  → three task heads: binary, type, grounding
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModel,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    SiglipModel,
    SiglipProcessor,
)


# ─────────────────────────────────────────────────────────────
# 1. Dual encoder: SigLIP + DINOv2
# ─────────────────────────────────────────────────────────────

class DualEncoder(nn.Module):
    def __init__(self, siglip_path: str, dinov2_path: str, freeze: bool = True):
        super().__init__()

        # SigLIP — cross-modal
        self.siglip = SiglipModel.from_pretrained(siglip_path)
        self.siglip_processor = SiglipProcessor.from_pretrained(siglip_path)

        # DINOv2 — vision-only FG/BG
        self.dinov2 = AutoModel.from_pretrained(dinov2_path)

        if freeze:
            for p in self.siglip.parameters():
                p.requires_grad = False
            for p in self.dinov2.parameters():
                p.requires_grad = False

        # embedding dims (siglip-so400m=1152, dinov2-large=1024)
        self.siglip_dim = self.siglip.config.vision_config.hidden_size
        self.dinov2_dim = self.dinov2.config.hidden_size

    @property
    def out_dim(self):
        # image_emb + text_emb (siglip) + vision_emb (dinov2)
        return self.siglip_dim * 2 + self.dinov2_dim

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        pixel_values  : (B, 3, H, W)  — pre-processed for SigLIP resolution
        input_ids     : (B, L)
        attention_mask: (B, L)
        Returns:
            fused_feat: (B, out_dim)
        """
        # SigLIP image + text embeddings
        sig_out = self.siglip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        img_emb  = sig_out.image_embeds   # (B, siglip_dim)
        text_emb = sig_out.text_embeds    # (B, siglip_dim)

        # DINOv2 CLS token
        dino_out  = self.dinov2(pixel_values=pixel_values)
        dino_emb  = dino_out.last_hidden_state[:, 0, :]  # (B, dinov2_dim)

        return torch.cat([img_emb, text_emb, dino_emb], dim=-1)  # (B, out_dim)


# ─────────────────────────────────────────────────────────────
# 2. Feature fusion MLP
# ─────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 3. Task heads
# ─────────────────────────────────────────────────────────────

class BinaryHead(nn.Module):
    """authentic (0) vs manipulated (1) + confidence"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)  # logits (B, 2)


class TypeHead(nn.Module):
    """origin (0) / text_swap (1) / text_attribute (2)"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 3)

    def forward(self, x):
        return self.fc(x)  # logits (B, 3)


class GroundingHead(nn.Module):
    """Per-token binary: is this token misleading?"""
    def __init__(self, in_dim: int, max_tokens: int = 64):
        super().__init__()
        self.max_tokens = max_tokens
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, max_tokens),
        )

    def forward(self, x):
        return self.fc(x)  # logits (B, max_tokens)


# ─────────────────────────────────────────────────────────────
# 4. Full model
# ─────────────────────────────────────────────────────────────

class PPMisDetModel(nn.Module):
    def __init__(
        self,
        siglip_path: str,
        dinov2_path: str,
        qwen_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        max_tokens: int = 64,
        freeze_encoders: bool = True,
    ):
        super().__init__()

        # ── dual encoder ──────────────────────────────────
        self.encoder = DualEncoder(siglip_path, dinov2_path, freeze=freeze_encoders)

        # ── fusion MLP ────────────────────────────────────
        fusion_out = 1024
        self.fusion = FusionMLP(self.encoder.out_dim, out_dim=fusion_out)

        # ── Qwen2-VL backbone with LoRA ───────────────────
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)
        self.qwen.print_trainable_parameters()

        # hidden size of the Qwen backbone
        qwen_hidden = self.qwen.config.hidden_size  # 3584 for 7B

        # project fused features into Qwen hidden space and add as prefix
        self.prefix_proj = nn.Linear(fusion_out, qwen_hidden)

        # ── task heads (operate on fused features) ────────
        self.binary_head    = BinaryHead(fusion_out)
        self.type_head      = TypeHead(fusion_out)
        self.grounding_head = GroundingHead(fusion_out, max_tokens=max_tokens)

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        qwen_input_ids=None,
        qwen_attention_mask=None,
        labels=None,
    ):
        """
        Returns dict with logits for each head.
        """
        # ── encoder + fusion ──────────────────────────────
        enc_feat   = self.encoder(pixel_values, input_ids, attention_mask)
        fused      = self.fusion(enc_feat)           # (B, 1024)

        # ── task head outputs ─────────────────────────────
        binary_logits   = self.binary_head(fused)    # (B, 2)
        type_logits     = self.type_head(fused)      # (B, 3)
        ground_logits   = self.grounding_head(fused) # (B, max_tokens)

        return {
            "binary_logits":  binary_logits,
            "type_logits":    type_logits,
            "ground_logits":  ground_logits,
            "fused":          fused,
        }


# ─────────────────────────────────────────────────────────────
# 5. Joint loss
# ─────────────────────────────────────────────────────────────

class JointLoss(nn.Module):
    def __init__(self, w_binary=0.5, w_ground=0.3, w_type=0.2):
        super().__init__()
        self.w_binary = w_binary
        self.w_ground = w_ground
        self.w_type   = w_type
        self.ce   = nn.CrossEntropyLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels, manip_types, token_labels):
        """
        outputs     : dict from PPMisDetModel.forward()
        labels      : (B,)  long  — 0 authentic, 1 manipulated
        manip_types : (B,)  long  — 0 origin, 1 text_swap, 2 text_attr
        token_labels: (B, T) float — per-token grounding targets
        """
        L_binary = self.ce(outputs["binary_logits"], labels)

        # type loss — only on manipulated samples
        manip_mask = labels.bool()
        if manip_mask.sum() > 0:
            L_type = self.ce(
                outputs["type_logits"][manip_mask],
                manip_types[manip_mask],
            )
        else:
            L_type = torch.tensor(0.0, device=labels.device)

        # grounding loss — only on manipulated samples
        if manip_mask.sum() > 0:
            L_ground = self.bce(
                outputs["ground_logits"][manip_mask],
                token_labels[manip_mask],
            )
        else:
            L_ground = torch.tensor(0.0, device=labels.device)

        total = (
            self.w_binary * L_binary
            + self.w_ground * L_ground
            + self.w_type   * L_type
        )
        return total, {"L_binary": L_binary, "L_type": L_type, "L_ground": L_ground}