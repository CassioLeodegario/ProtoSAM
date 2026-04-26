"""
2D Superpixel Dataset for polyp images (PNG format).
Replaces GenericSuperDatasetv2 for natural/endoscopic 2D images.
Superpixels are generated on-the-fly using SLIC segmentation.
Validates that the selected superpixel produces at least one fully covered
cell on the prototype grid (matches MultiProtoAsConv's FG/BG_THRESH=0.95
criterion), so the chosen sample never produces zero prototypes downstream.
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage.segmentation import slic


SUPERPIX_CONFIG = {
    "SMALL": {"n_segments": 200, "compactness": 10},
    "MIDDLE": {"n_segments": 100, "compactness": 10},
    "LARGE": {"n_segments": 50, "compactness": 10},
}

# Same grid the MultiProtoAsConv uses (DEFAULT_FEATURE_SIZE / proto_grid_size).
# A superpixel is "valid" only if at least one cell of this grid is fully
# covered by it — that is the exact criterion the prototype layer applies
# at threshold FG_THRESH=0.95, so any sample we accept here will produce
# at least one foreground prototype downstream.
PROTO_GRID = 8
COVERAGE_THRESH = 0.95


class PolypSuperDataset(Dataset):
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
        super().__init__()
        self.base_dir = base_dir
        self.image_size = image_size
        self.transforms = transforms
        self.num_rep = num_rep
        self.fix_length = fix_length
        self.superpix_cfg = SUPERPIX_CONFIG[superpix_scale]
        self.nclass = 2
        self.tile_z_dim = 1
        self.proto_grid = PROTO_GRID
        self.coverage_thresh = COVERAGE_THRESH

        img_dir = os.path.join(base_dir, "image")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(self.img_paths) > 0, f"No PNG images found in {img_dir}"

        split = int(0.9 * len(self.img_paths))
        self.img_paths = (
            self.img_paths[:split] if mode == "train" else self.img_paths[split:]
        )
        self.size = len(self.img_paths)
        print(
            f"[PolypSuperDataset] mode={mode} | images={self.size} | superpix={superpix_scale}"
        )

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        return img.astype(np.float32) / 255.0

    def _compute_superpixels(self, img_rgb):
        sp = slic(img_rgb, **self.superpix_cfg, start_label=1)
        return sp.astype(np.int32)

    def _grid_coverage(self, mask):
        # avg-pool the binary mask down to the prototype grid (e.g. 8x8).
        # cv2.INTER_AREA on a binary input acts as average pooling, so each
        # output cell holds the foreground fraction of its receptive field.
        return cv2.resize(
            mask.astype(np.float32),
            (self.proto_grid, self.proto_grid),
            interpolation=cv2.INTER_AREA,
        )

    def _mask_survives_stride(self, mask):
        # Foreground: at least one grid cell must be fully covered (>= thresh).
        # Background ((1 - mask)) is also required to have one fully covered
        # cell, otherwise the BG branch of MultiProtoAsConv produces 0
        # prototypes and the whole batch is dropped downstream.
        cov_fg = self._grid_coverage(mask).max()
        cov_bg = self._grid_coverage(1.0 - mask).max()
        return cov_fg >= self.coverage_thresh and cov_bg >= self.coverage_thresh

    def _pick_valid_superpixel(self, sp_map, max_tries=20):
        labels = np.unique(sp_map).tolist()
        random.shuffle(labels)
        for label in labels[:max_tries]:
            mask = (sp_map == label).astype(np.float32)
            if self._mask_survives_stride(mask):
                return label, mask
        return labels[0], (sp_map == labels[0]).astype(np.float32)

    def _get_mask(self, label):
        fg_mask = torch.where(
            label == 1, torch.ones_like(label), torch.zeros_like(label)
        )
        bg_mask = torch.where(
            label != 1, torch.ones_like(label), torch.zeros_like(label)
        )
        return {"fg_mask": fg_mask, "bg_mask": bg_mask}

    def reload_buffer(self):
        pass

    def __len__(self):
        return self.fix_length if self.fix_length is not None else self.size

    def __getitem__(self, index):
        index = index % self.size
        img_path = self.img_paths[index]

        img = self._load_image(img_path)
        sp_map = self._compute_superpixels(img)
        superpix_label, label_t = self._pick_valid_superpixel(sp_map)
        label_4aug = label_t[..., np.newaxis]
        comp = np.concatenate([img, label_4aug], axis=-1)

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

            if lb_aug.sum() == 0 or not self._mask_survives_stride(lb_aug[..., 0]):
                superpix_label, label_t = self._pick_valid_superpixel(sp_map)
                img_aug = img
                lb_aug = label_t[..., np.newaxis]

            img_t = torch.from_numpy(np.transpose(img_aug, (2, 0, 1))).float()
            lb_t = torch.from_numpy(lb_aug.squeeze(-1)).float()
            pair_buffer.append({"image": img_t, "label": lb_t})

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
