"""
Gera comparação qualitativa lado a lado: DINOv2-L vs VMamba-Tiny distilado.

Saída: uma imagem PNG por amostra com 4 colunas:
    [Imagem original | Ground Truth | DINOv2-L | VMamba-Tiny distil]

Uso:
    python visualize_predictions.py --n-samples 10
    python visualize_predictions.py --n-samples 20 --seed 123
    python visualize_predictions.py --indices 5 42 100 200
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.ProtoSAM import ProtoSAM, ALPNetWrapper, InputFactory, TYPE_ALPNET
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.PolypDataset import get_polyp_dataset, PolypDataset

DINOV2_CKPT  = "None"
VMAMBA_CKPT  = "runs/distill_vmamba_tiny/distilled_vmamba_tiny_512_10k.pth"
INPUT_SIZE   = 512
SUPPORT_IDX  = 6
OUT_DIR      = "qualitative_results"


def build_protosam(modelname, reload_path, input_size):
    cfg = {
        "align": False,
        "use_coco_init": False,
        "which_model": modelname,
        "cls_name": "grid_proto",
        "proto_grid_size": 8,
        "feature_hw": [input_size // 8, input_size // 8],
        "reload_model_path": reload_path if reload_path != "None" else None,
        "lora": 0,
        "use_slice_adapter": False,
        "adapter_layers": 3,
        "debug": False,
        "use_pos_enc": False,
    }
    alpnet = FewShotSeg(image_size=input_size, pretrained_path=reload_path if reload_path != "None" else None, cfg=cfg)
    alpnet.cuda()
    wrapper = ALPNetWrapper(alpnet)
    sam_checkpoint = "pretrained_model/sam_vit_h.pth"
    model = ProtoSAM(
        image_size=(1024, 1024),
        coarse_segmentation_model=wrapper,
        use_bbox=True, use_points=True, use_mask=False,
        use_cca=True, point_mode="both",
        use_sam_trans=True, coarse_pred_only=False,
        sam_pretrained_path=sam_checkpoint,
        use_neg_points=False,
    )
    return model.cuda().eval()


def run_inference(model, query_images, support_images, support_fg_mask):
    with torch.no_grad():
        inp = InputFactory.create_input(
            input_type=TYPE_ALPNET,
            query_image=query_images,
            support_images=support_images,
            support_labels=support_fg_mask,
            isval=True, val_wsize=2,
            original_sz=query_images.shape[-2:],
            img_sz=query_images.shape[-2:],
            gts=None,
        )
        inp.to(torch.device("cuda"))
        pred, _ = model(query_images, inp, degrees_rotate=0)
    return pred.cpu().detach().numpy().squeeze()


def save_comparison(img_np, gt_np, pred_dino, pred_vmamba, case, idx, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Imagem", "Ground Truth", "DINOv2-L", "VMamba-Tiny distil"]
    data   = [img_np, gt_np, pred_dino, pred_vmamba]

    for ax, title, d in zip(axes, titles, data):
        if d.ndim == 3:
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ax.imshow(d.transpose(1, 2, 0) if d.shape[0] == 3 else d)
        else:
            ax.imshow(d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"{case} — idx {idx}", fontsize=10, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{case}_{idx:04d}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="índices específicos do test set (sobrepõe --n-samples)")
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--vmamba-ckpt", default=VMAMBA_CKPT)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--support-idx", type=int, default=SUPPORT_IDX)
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    cudnn.enabled = True
    os.makedirs(args.out_dir, exist_ok=True)

    sam_trans = ResizeLongestSide(1024)
    tr_dataset, te_dataset = get_polyp_dataset(sam_trans=sam_trans, image_size=(1024, 1024))

    testloader = DataLoader(te_dataset, batch_size=1, shuffle=False,
                            num_workers=2, drop_last=False)

    # support set fixo
    support_images, support_fg_mask, _ = tr_dataset.get_support(n_support=1)

    # seleciona índices
    if args.indices:
        chosen = set(args.indices)
    else:
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(len(te_dataset), size=args.n_samples, replace=False).tolist())

    print(f"Carregando modelos...")
    model_dino   = build_protosam("dinov2_l14",  DINOV2_CKPT,      args.input_size)
    model_vmamba = build_protosam("vmamba_tiny",  args.vmamba_ckpt, args.input_size)
    print("Modelos prontos.")

    saved = []
    for idx, batch in enumerate(testloader):
        if idx not in chosen:
            continue

        query_images = batch["image"].cuda()
        query_labels = batch["label"][0].numpy()
        case = batch["case"][0]

        pred_dino   = run_inference(model_dino,   query_images, support_images, support_fg_mask)
        pred_vmamba = run_inference(model_vmamba, query_images, support_images, support_fg_mask)

        img_np = batch["image"][0].numpy()

        path = save_comparison(img_np, query_labels, pred_dino, pred_vmamba, case, idx, args.out_dir)
        saved.append(path)
        print(f"[{len(saved)}/{len(chosen)}] {path}")

        if len(saved) >= len(chosen):
            break

    print(f"\nSalvo em: {args.out_dir}/")


if __name__ == "__main__":
    main()
