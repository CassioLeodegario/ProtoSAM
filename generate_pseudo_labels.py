"""
Generate pseudo-labels for the polyp TrainDataset using ProtoSAM with a
strong teacher backbone (default: DINOv2-L). The output is one binary PNG
per training image, written next to the originals so PolypPseudoDataset
can pair them by filename.

Usage:
    python generate_pseudo_labels.py with \
        modelname=dinov2_l14 \
        clsname=grid_proto \
        protosam_sam_ver=sam_h \
        do_cca=True \
        coarse_pred_only=False \
        n_support=1 \
        support_idx=[6] \
        output_dir=data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l
"""
import os
import time
import shutil
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.ProtoSAM import InputFactory, TYPE_ALPNET
from models.segment_anything.utils.transforms import ResizeLongestSide
from dataloaders.PolypDataset import PolypDataset, get_polyp_dataset

from validation_protosam import get_model, get_support_set_polyps
from config_ssl_upload import ex


@ex.config
def pseudo_cfg():
    # Where to write the pseudo-labels. Subfolders mirror the structure of
    # TrainDataset (Kvasir/, CVC-ClinicDB/, ...).
    output_dir = "data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l"
    # If True, overwrite existing PNGs. If False, skip images already done.
    overwrite_pseudo = False
    # Resize the saved mask back to the original image resolution.
    save_at_original_size = True


def _binarize(pred):
    """ProtoSAM returns a (H, W) tensor with values {0, 1}. Reduce to HxW uint8."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"unexpected pred shape after squeeze: {pred.shape}")
    return (pred > 0.5).astype(np.uint8) * 255


def _output_path(output_dir, image_path, train_root):
    """Mirror the relative path of image_path under output_dir."""
    rel = os.path.relpath(image_path, train_root)
    rel = os.path.splitext(rel)[0] + ".png"
    out = os.path.join(output_dir, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


@ex.automain
def main(_run, _config, _log):
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config["gpu_id"])
    torch.set_num_threads(1)

    output_dir = _config["output_dir"]
    overwrite = _config["overwrite_pseudo"]
    save_orig = _config["save_at_original_size"]
    os.makedirs(output_dir, exist_ok=True)

    _log.info(f"###### Loading teacher model ({_config['modelname']}) ######")
    model = get_model(_config).to(torch.device("cuda"))
    model.eval()

    _log.info("###### Loading TrainDataset ######")
    sam_trans = ResizeLongestSide(1024)
    tr_dataset, _te_dataset = get_polyp_dataset(
        sam_trans=sam_trans, image_size=(1024, 1024)
    )
    train_root = os.path.commonpath(tr_dataset.images)

    # Use a fixed support set so every pseudo-label comes from the same prompt.
    support_images, support_fg_mask, _case = get_support_set_polyps(_config, tr_dataset)
    _log.info(f"###### Support set: n={len(support_images)} ######")

    loader = DataLoader(
        tr_dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=False,
    )

    n_done = 0
    n_skip = 0
    n_empty = 0
    t0 = time.time()
    with tqdm(loader, desc="generating pseudo-labels") as pbar:
        for idx, sample in enumerate(pbar):
            image_path = tr_dataset.images[idx]
            out_path = _output_path(output_dir, image_path, train_root)
            if not overwrite and os.path.exists(out_path):
                n_skip += 1
                pbar.set_postfix(done=n_done, skip=n_skip, empty=n_empty)
                continue

            query_images = sample["image"].cuda()
            with torch.no_grad():
                coarse_in = InputFactory.create_input(
                    input_type=TYPE_ALPNET,
                    query_image=query_images,
                    support_images=support_images,
                    support_labels=support_fg_mask,
                    isval=True,
                    val_wsize=_config["val_wsize"],
                    original_sz=query_images.shape[-2:],
                    img_sz=query_images.shape[-2:],
                    gts=sample["label"],
                )
                coarse_in.to(torch.device("cuda"))
                pred, _scores = model(query_images, coarse_in, degrees_rotate=0)

            mask = _binarize(pred)
            if mask.sum() == 0:
                n_empty += 1

            if save_orig:
                orig_h, orig_w = (int(v) for v in sample["original_size"][0].tolist())
                if mask.shape != (orig_h, orig_w):
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(out_path, mask)
            n_done += 1
            pbar.set_postfix(done=n_done, skip=n_skip, empty=n_empty)

    dt = time.time() - t0
    _log.info(
        f"###### Done. wrote={n_done} skipped={n_skip} empty={n_empty} "
        f"elapsed={dt:.1f}s output_dir={output_dir} ######"
    )
