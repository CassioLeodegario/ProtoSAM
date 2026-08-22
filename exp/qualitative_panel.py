"""
Painel qualitativo: mesma imagem, várias resoluções, os dois encoders lado a lado.

Colunas: Consulta | Ground truth | DINOv2-L | VMamba-T destilado
Linhas:  uma por resolução de entrada. A primeira linha mostra o par de suporte.

Não sobrescreve `visualize_predictions.py`, que tem três defeitos que
invalidariam as figuras:
  D6 — faz .eval() no ProtoSAM, que NÃO alcança o FewShotSeg; as predições do
       VMamba sairiam com drop_path ativo, diferentes das tabelas.
  D7 — define SUPPORT_IDX mas nunca repassa; usa o sorteio aleatório.
  -  não mostra o suporte e não varre resolução.

As amostras são escolhidas por DISCORDÂNCIA entre os encoders, nos dois sentidos
(casos onde cada um ganha), para não virar seleção a dedo de um lado só.

Uso:
    python -m exp.qualitative_panel
    python -m exp.qualitative_panel --k 4 --sizes 256 512 1024
    python -m exp.qualitative_panel --indices 3 17 42
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ProtoSAM import ProtoSAM, ALPNetWrapper, InputFactory, TYPE_ALPNET
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.PolypDataset import get_polyp_dataset
from exp.envinfo import REPO_ROOT

# Mesmas cores das figuras quantitativas: a cor segue a entidade em todo o texto.
COLOR = {"dinov2_l14": "#2a78d6", "vmamba_tiny": "#eda100"}
LABEL = {"dinov2_l14": "DINOv2-L", "vmamba_tiny": "VMamba-T destilado"}
GT_COLOR = "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
DIST_CKPT = "runs/distill_vmamba_tiny/distilled_vmamba_tiny_512_10k.pth"


def build_alpnet(modelname, reload_path, image_size):
    cfg = {"align": False, "use_coco_init": False, "which_model": modelname,
           "cls_name": "grid_proto", "proto_grid_size": 8,
           "feature_hw": [image_size // 8, image_size // 8],
           "reload_model_path": None, "lora": 0, "use_slice_adapter": False,
           "adapter_layers": 3, "debug": False, "use_pos_enc": False}
    net = FewShotSeg(image_size=image_size,
                     pretrained_path=reload_path if reload_path != "None" else None,
                     cfg=cfg)
    net.cuda()
    net.eval()   # D6: sem isto o drop_path do VMamba fica ativo na inferencia
    return ALPNetWrapper(net)


def build_protosam():
    return ProtoSAM(
        image_size=(1024, 1024),
        coarse_segmentation_model=None,
        use_bbox=True, use_points=True, use_mask=False,
        use_cca=True, point_mode="both", use_sam_trans=True,
        coarse_pred_only=False,
        sam_pretrained_path=os.path.join(REPO_ROOT, "pretrained_model/sam_vit_h.pth"),
        use_neg_points=False,
    ).cuda().eval()


@torch.no_grad()
def predict(model, query_images, support_images, support_masks):
    inp = InputFactory.create_input(
        input_type=TYPE_ALPNET, query_image=query_images,
        support_images=support_images, support_labels=support_masks,
        isval=True, val_wsize=2,
        original_sz=query_images.shape[-2:], img_sz=query_images.shape[-2:], gts=None)
    inp.to(torch.device("cuda"))
    pred, _ = model(query_images, inp, degrees_rotate=0)
    return np.asarray(pred.cpu().detach()).squeeze()


def dice_of(pred, gt):
    p = (np.asarray(pred) > 0).astype(np.float64)
    g = (np.asarray(gt) > 0).astype(np.float64)
    return float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def to_display(img):
    """
    Tensor do dataset -> RGB em [0,1], com a cor FIEL.

    Com sam_trans a media e o desvio do dataset viram 0 e 1, entao o tensor ja
    esta em escala 0-255: basta dividir. Normalizar por min-max reequilibraria
    os canais e deixaria o tecido azulado em vez de rosado.
    """
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    a = a.astype(np.float64)
    if a.max() > 1.5:
        return np.clip(a / 255.0, 0.0, 1.0)
    return np.clip(a, 0.0, 1.0)


def show(ax, base, mask=None, color=None, title=None, sub=None):
    ax.imshow(base, interpolation="nearest")
    if mask is not None:
        m = (np.asarray(mask) > 0).astype(float)
        ax.imshow(m, cmap=ListedColormap([(0, 0, 0, 0), color]),
                  alpha=0.55, interpolation="nearest", vmin=0, vmax=1)
        ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e1e0d9")
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=5)
    if sub:
        ax.set_xlabel(sub, fontsize=9, color=INK2, labelpad=4)


def panel(sample, support, results, sizes, out_dir):
    idx, case, img, gt = sample
    supp_img, supp_msk = support
    nrows = len(sizes) + 1
    fig, axes = plt.subplots(nrows, 4, figsize=(11.2, 2.9 * nrows))
    fig.patch.set_facecolor("#fcfcfb")

    show(axes[0, 0], supp_img, title="Suporte — imagem")
    show(axes[0, 1], supp_img, supp_msk, GT_COLOR, title="Suporte — máscara (entrada)")
    for c in (2, 3):
        axes[0, c].axis("off")
    axes[0, 2].text(0.0, 0.5,
                    "A máscara do suporte é ENTRADA do modelo,\n"
                    "não checagem: dela saem os protótipos.",
                    fontsize=9.5, color=INK2, va="center",
                    transform=axes[0, 2].transAxes)

    for r, size in enumerate(sizes, start=1):
        show(axes[r, 0], img, title="Consulta" if r == 1 else None, sub=f"{size} px")
        show(axes[r, 1], img, gt, GT_COLOR, title="Ground truth" if r == 1 else None)
        for c, enc in enumerate(["dinov2_l14", "vmamba_tiny"], start=2):
            pred, d = results[(enc, size)]
            show(axes[r, c], img, pred, COLOR[enc],
                 title=LABEL[enc] if r == 1 else None, sub=f"Dice {d:.3f}")

    fig.suptitle(f"{case or 'Kvasir'} — imagem {idx}   ·   suporte índice 927",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"qualitativo_{idx:04d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--select-size", type=int, default=512,
                   help="resolucao usada para medir a discordancia")
    p.add_argument("--k", type=int, default=3, help="amostras por sentido da discordancia")
    p.add_argument("--indices", nargs="+", type=int, default=None)
    p.add_argument("--support-idx", type=int, default=927)
    p.add_argument("--out-dir", default="exp/figures/qualitativo")
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    torch.cuda.set_device(a.gpu_id)
    out_dir = os.path.join(REPO_ROOT, a.out_dir)

    sam_trans = ResizeLongestSide(1024)
    tr, te = get_polyp_dataset(sam_trans=sam_trans, image_size=(1024, 1024))
    supp_imgs, supp_msks, _ = tr.get_support(n_support=1, support_idx=[a.support_idx])
    print(f"suporte: {getattr(tr, 'last_support_paths', '?')}")

    loader = DataLoader(te, batch_size=1, shuffle=False, num_workers=2)
    samples = []
    for i, b in enumerate(loader):
        samples.append((i, b["case"][0], b["image"], b["label"][0].numpy()))
    print(f"{len(samples)} imagens de teste")

    model = build_protosam()

    def run_all(size, idxs):
        out = {}
        for enc, ckpt in [("dinov2_l14", "None"), ("vmamba_tiny", DIST_CKPT)]:
            model.coarse_segmentation_model = build_alpnet(enc, ckpt, size)
            for i in idxs:
                _, _, img, gt = samples[i]
                pred = predict(model, img.cuda(), supp_imgs, supp_msks)
                out[(i, enc)] = (pred, dice_of(pred, gt))
            torch.cuda.empty_cache()
        return out

    if a.indices:
        chosen = list(a.indices)
    else:
        print(f"medindo discordancia em {a.select_size} px sobre {len(samples)} imagens...")
        sel = run_all(a.select_size, range(len(samples)))
        diffs = sorted(((sel[(i, "vmamba_tiny")][1] - sel[(i, "dinov2_l14")][1], i)
                        for i in range(len(samples))), reverse=True)
        top = [i for _, i in diffs[:a.k]]
        bot = [i for _, i in diffs[-a.k:]]
        chosen = top + bot
        print("escolhidas (VMamba ganha):", top)
        print("escolhidas (DINOv2 ganha):", bot)

    per_sample = {i: {} for i in chosen}
    for size in a.sizes:
        r = run_all(size, chosen)
        for i in chosen:
            for enc in ("dinov2_l14", "vmamba_tiny"):
                per_sample[i][(enc, size)] = r[(i, enc)]

    supp_disp = to_display(supp_imgs[0][0] if supp_imgs[0].ndim == 4 else supp_imgs[0])
    supp_mask = np.asarray(supp_msks[0]).squeeze()
    for i in chosen:
        idx, case, img, gt = samples[i]
        p = panel((idx, case, to_display(img[0]), gt), (supp_disp, supp_mask),
                  per_sample[i], a.sizes, out_dir)
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
