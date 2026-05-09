"""
Distill a strong teacher (DINOv2 ProtoSAM) into a VMamba-Tiny student via
pseudo-label supervision.

Pipeline:
    1. Pseudo-labels were already generated for TrainDataset by
       generate_pseudo_labels.py (one PNG per training image, mirroring the
       images/ folder structure).
    2. We supervise a VMamba-Tiny encoder + lightweight seg head with
       cross-entropy + Dice against those pseudo-labels.
    3. The final checkpoint is the *full* FewShotSeg state_dict, so
       validation_protosam.py can load it via reload_model_path with the same
       few-shot wrapper it uses for the baselines.

The seg head exists only during distillation; it is discarded at save time.
"""
import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.PolypPseudoDataset import get_polyp_pseudo_dataset


# -- model --------------------------------------------------------------------

EMBED_DIMS = {
    "vmamba_tiny": 768,
    "vmamba_small": 768,
    "vmamba_base": 1024,
    "dinov2_b14": 768,
    "dinov2_l14": 1024,
    "dlfcn_res101": 256,
}


class DistillationModel(nn.Module):
    """FewShotSeg encoder + a small seg head for supervised distillation."""

    def __init__(self, image_size: int, modelname: str, hidden: int = 128):
        super().__init__()
        cfg = {
            "align": False,
            "use_coco_init": False,
            "which_model": modelname,
            "cls_name": "grid_proto",
            "proto_grid_size": 8,
            "feature_hw": [image_size // 8, image_size // 8],
            "reload_model_path": None,
            "lora": 0,
            "use_slice_adapter": False,
            "adapter_layers": 3,
            "debug": False,
            "use_pos_enc": False,
        }
        self.fewshot = FewShotSeg(image_size=image_size, pretrained_path=None, cfg=cfg)
        embed_dim = EMBED_DIMS[modelname]
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 2, 1),  # binary fg/bg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fewshot.get_features(x)  # B, C, h', w'
        logits_small = self.seg_head(feat)
        return F.interpolate(logits_small, size=x.shape[-2:], mode="bilinear", align_corners=False)


# -- losses -------------------------------------------------------------------


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """logits: (B, 2, H, W) raw, target: (B, H, W) long {0,1}."""
    probs = logits.softmax(dim=1)[:, 1]
    target_f = (target == 1).float()
    inter = (probs * target_f).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()


# -- training -----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modelname", default="vmamba_tiny", choices=list(EMBED_DIMS))
    p.add_argument("--image-root", required=True,
                   help="e.g. data/PolypDataset/TrainDataset/images")
    p.add_argument("--pseudo-root", required=True,
                   help="e.g. data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l")
    p.add_argument("--output-dir", required=True,
                   help="where to write the distilled checkpoint")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true",
                   help="disable training-time augmentation")
    p.add_argument("--freeze-stages", type=int, default=0,
                   help="number of VMamba stages to freeze (0–4, ignored for non-VMamba)")
    return p.parse_args()


def lr_lambda(step: int, n_steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, n_steps - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


def freeze_vmamba_stages(model: DistillationModel, n_stages: int) -> None:
    """Freeze the first n_stages of VMamba; preserves last stage(s) trainable."""
    if n_stages <= 0:
        return
    enc = model.fewshot.encoder
    if not hasattr(enc, "layers"):
        return
    for i, layer in enumerate(enc.layers):
        if i < n_stages:
            for p in layer.parameters():
                p.requires_grad_(False)


def save_checkpoint(model: DistillationModel, path: str) -> None:
    """Save FewShotSeg state_dict only — strips seg_head."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.fewshot.state_dict(), path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required — VMamba uses a custom CUDA kernel with no CPU fallback. "
            "Make sure CUDA_VISIBLE_DEVICES is set and the GPU is not fully occupied."
        )
    device = torch.device("cuda")
    print(f"[init] device={device} model={args.modelname} image_size={args.image_size}")

    # --- data ---
    ds = get_polyp_pseudo_dataset(
        image_root=args.image_root,
        pseudo_root=args.pseudo_root,
        image_size=(args.image_size, args.image_size),
        augment=not args.no_augment,
    )
    print(f"[data] dataset size = {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # --- model ---
    model = DistillationModel(args.image_size, args.modelname).to(device)
    if args.modelname.startswith("vmamba"):
        freeze_vmamba_stages(model, args.freeze_stages)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_params/1e6:.2f}M")

    # --- optim ---
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: lr_lambda(s, args.n_steps, args.warmup_steps),
    )

    # --- loop ---
    model.train()
    step = 0
    t0 = time.time()
    running = {"ce": 0.0, "dice": 0.0, "total": 0.0}
    pbar = tqdm(total=args.n_steps, desc="distill")
    while step < args.n_steps:
        for batch in loader:
            if step >= args.n_steps:
                break
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["label"].to(device, non_blocking=True).long()

            logits = model(img)
            l_ce = F.cross_entropy(logits, mask)
            l_dice = dice_loss(logits, mask)
            loss = args.ce_weight * l_ce + args.dice_weight * l_dice

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()

            running["ce"] += float(l_ce.item())
            running["dice"] += float(l_dice.item())
            running["total"] += float(loss.item())
            step += 1
            pbar.update(1)
            pbar.set_postfix(ce=l_ce.item(), dice=l_dice.item(), lr=sched.get_last_lr()[0])

            if step % args.print_every == 0:
                k = args.print_every
                print(
                    f"step {step}/{args.n_steps} "
                    f"ce={running['ce']/k:.4f} dice={running['dice']/k:.4f} "
                    f"total={running['total']/k:.4f} lr={sched.get_last_lr()[0]:.2e}"
                )
                running = {"ce": 0.0, "dice": 0.0, "total": 0.0}

            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"distilled_{args.modelname}_{step}.pth")
                save_checkpoint(model, ckpt)
                print(f"[save] {ckpt}")
    pbar.close()

    final = os.path.join(args.output_dir, f"distilled_{args.modelname}_final.pth")
    save_checkpoint(model, final)
    print(f"[done] elapsed={time.time()-t0:.1f}s final={final}")


if __name__ == "__main__":
    main()
