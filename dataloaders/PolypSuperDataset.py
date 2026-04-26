"""
2D Superpixel Dataset for polyp images (PNG format).
Replaces GenericSuperDatasetv2 for natural/endoscopic 2D images.
Superpixels are generated on-the-fly using Felzenszwalb segmentation.
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage.segmentation import felzenszwalb
from dataloaders.common import BaseDataset
from util.consts import IMG_SIZE


POLYP_DATASET_INFO = {
    "PSEU_LABEL_NAME": ["BGD", "SUPFG"],
    "REAL_LABEL_NAME": ["BGD", "POLYP"],
    "MODALITY": "RGB",
}

SUPERPIX_CONFIG = {
    "SMALL": {"scale": 100, "sigma": 0.8, "min_size": 50},
    "MIDDLE": {"scale": 200, "sigma": 0.8, "min_size": 100},
    "LARGE": {"scale": 400, "sigma": 0.8, "min_size": 200},
}


class PolypSuperDataset(Dataset):
    """
    Superpixel-based few-shot dataset for 2D polyp PNG images.
    Mimics the interface of SuperpixelDataset (GenericSuperDatasetv2)
    so training.py can use it without changes.
    """

    def __init__(
        self,
        base_dir,
        mode,
        image_size,
        transforms,
        num_rep=2,
        fix_length=None,
        superpix_scale="MIDDLE",
        **kwargs,
    ):
        """
        Args:
            base_dir:       path containing 'image/' and 'mask/' subfolders
            mode:           'train' or 'val' (val not used in EFT but kept for compat)
            image_size:     spatial size to resize images (square)
            transforms:     augmentation function (same as SuperpixelDataset)
            num_rep:        number of augmented views per sample (support + query)
            fix_length:     fix dataset length (used as epoch size)
            superpix_scale: 'SMALL' | 'MIDDLE' | 'LARGE'
        """
        super().__init__()
        self.base_dir = base_dir
        self.image_size = image_size
        self.transforms = transforms
        self.num_rep = num_rep
        self.fix_length = fix_length
        self.superpix_cfg = SUPERPIX_CONFIG[superpix_scale]
        self.nclass = 2  # BGD + SUPFG (binary pseudolabel)
        self.tile_z_dim = 1  # RGB images — no tiling needed

        img_dir = os.path.join(base_dir, "image")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(self.img_paths) > 0, f"No PNG images found in {img_dir}"

        if mode == "val":
            # Use last 10% as val if needed
            split = int(0.9 * len(self.img_paths))
            self.img_paths = self.img_paths[split:]
        else:
            split = int(0.9 * len(self.img_paths))
            self.img_paths = self.img_paths[:split]

        self.size = len(self.img_paths)
        print(
            f"[PolypSuperDataset] mode={mode} | images={self.size} | superpix={superpix_scale}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_image(self, path):
        """Load PNG, resize to image_size, return float32 HxWx3 in [0,1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        return img.astype(np.float32) / 255.0

    def _compute_superpixels(self, img_rgb):
        """
        Run Felzenszwalb on a [0,1] float RGB image.
        Returns integer label map HxW with labels starting at 1.
        """
        # felzenszwalb expects uint8 or float in [0,1]
        sp = felzenszwalb(img_rgb, **self.superpix_cfg)
        sp = sp + 1  # shift so 0 is never a valid label (mirrors NIfTI convention)
        return sp.astype(np.int32)

    def _pick_superpixel(self, sp_map, min_pixels=200):
        """Pick a random superpixel label with minimum pixel count."""
        labels, counts = np.unique(sp_map, return_counts=True)
        valid = labels[counts >= min_pixels]
        if len(valid) == 0:
            valid = labels  # fallback: use all
        return random.choice(valid.tolist())

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self.fix_length is not None:
            return self.fix_length
        return self.size

    def __getitem__(self, index):
        index = index % self.size
        img_path = self.img_paths[index]

        img = self._load_image(img_path)  # HxWx3  float32 [0,1]
        sp_map = self._compute_superpixels(img)  # HxW    int32

        superpix_label = self._pick_superpixel(sp_map)
        label_t = np.float32(sp_map == superpix_label)  # HxW binary

        # Stack image + label for joint augmentation (same API as SuperpixelDataset)
        label_4aug = label_t[..., np.newaxis]  # HxWx1
        comp = np.concatenate([img, label_4aug], axis=-1)  # HxWx4

        pair_buffer = []
        for _ in range(self.num_rep):
            if self.transforms is not None:
                img_aug, lb_aug = self.transforms(
                    comp,
                    c_img=3,
                    c_label=1,
                    nclass=self.nclass,
                    is_train=True,
                    use_onehot=False,
                )
            else:
                img_aug = comp[:, :, :3]
                lb_aug = comp[:, :, 3:4]

            # Se a máscara ficou vazia após augmentação, tenta outro superpixel
            if lb_aug.sum() == 0:
                superpix_label = self._pick_superpixel(sp_map)
                label_t = np.float32(sp_map == superpix_label)
                label_4aug = label_t[..., np.newaxis]
                comp = np.concatenate([img, label_4aug], axis=-1)
                img_aug = comp[:, :, :3]
                lb_aug = comp[:, :, 3:4]

            img_t = torch.from_numpy(np.transpose(img_aug, (2, 0, 1))).float()
            lb_t = torch.from_numpy(lb_aug.squeeze(-1)).float()

            pair_buffer.append({"image": img_t, "label": lb_t})

        # Split pair_buffer into support (even) and query (odd) — same as SuperpixelDataset
        support_images, support_mask, support_class = [], [], []
        query_images, query_labels, query_class = [], [], []

        for idx, itm in enumerate(pair_buffer):
            if idx % 2 == 0:
                support_images.append(itm["image"])
                support_class.append(1)
                support_mask.append(self._get_mask(itm["label"]))
            else:
                query_images.append(itm["image"])
                query_class.append(1)
                query_labels.append(itm["label"])

        scan_id = os.path.splitext(os.path.basename(img_path))[0]

        return {
            "class_ids": [support_class],
            "support_images": [support_images],
            "superpix_label": superpix_label,
            "superpix_label_raw": sp_map,
            "support_mask": [support_mask],
            "query_images": query_images,
            "query_labels": query_labels,
            "scan_id": scan_id,
            "z_id": 0,
            "nframe": 1,
        }

    def _get_mask(self, label):
        """Generate fg/bg mask dict — same format as SuperpixelDataset.getMaskMedImg."""
        fg_mask = torch.where(
            label == 1, torch.ones_like(label), torch.zeros_like(label)
        )
        bg_mask = torch.where(
            label != 1, torch.ones_like(label), torch.zeros_like(label)
        )
        return {"fg_mask": fg_mask, "bg_mask": bg_mask}

    # Stub to keep reload_buffer calls from crashing if called externally
    def reload_buffer(self):
        pass
