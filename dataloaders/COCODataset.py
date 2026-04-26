"""
COCO-format dataset for few-shot multi-class / multi-instance segmentation.
Follows the same interface as PolypDataset so it can be dropped into the
existing ProtoSAM validation pipeline.

Requirements: pycocotools  (pip install pycocotools)
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2

try:
    from pycocotools.coco import COCO
except ImportError:
    raise ImportError("pycocotools is required: pip install pycocotools")


class COCODataset(data.Dataset):
    """
    Dataset for COCO-format annotations.

    __getitem__ returns a dict compatible with the ProtoSAM validation loop:
        image         : (C, H, W) float tensor, normalised
        label         : (H, W) float tensor
                        — binary mask when category_id is set
                        — integer label map (0 = bg, values = category_ids) otherwise
        original_size : (2,) float tensor  [H, W]
        image_size    : (2,) float tensor  [H, W]
        case          : str  (category name in single-class mode, file name otherwise)

    get_support(category_id, n_support) returns
        (support_images, support_labels, case)
    with the same layout as PolypDataset.get_support(), so it plugs directly
    into get_support_set_polyps / the ALPNetInput factory.
    """

    def __init__(
        self,
        image_dir: str,
        ann_file: str,
        image_size: tuple = (1024, 1024),
        category_ids: list = None,
        category_id: int = None,
        augmentations=None,
        sam_trans=None,
        train: bool = True,
        ds_mean=None,
        ds_std=None,
    ):
        """
        Parameters
        ----------
        image_dir     : directory that contains the JPEG/PNG images
        ann_file      : path to the COCO annotation JSON file
        image_size    : (H, W) to resize images/masks to
        category_ids  : list of category IDs to include; None = all categories
        category_id   : if set, __getitem__ returns a binary mask for this class only
        augmentations : callable (image_tensor, mask_tensor) -> (image, mask)
        sam_trans     : ResizeLongestSide SAM transform (handles preprocessing)
        train         : unused flag kept for API compatibility
        ds_mean       : per-channel mean for normalisation (scalar or (3,) tensor)
        ds_std        : per-channel std  for normalisation (scalar or (3,) tensor)
        """
        self.image_dir = image_dir
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.augmentations = augmentations
        self.sam_trans = sam_trans
        self.train = train
        self.category_id = category_id

        self.coco = COCO(ann_file)

        # Resolve which categories are in scope
        if category_ids is not None:
            self.category_ids = list(category_ids)
        else:
            self.category_ids = sorted(self.coco.getCatIds())

        self.cat_info = {cat['id']: cat for cat in self.coco.loadCats(self.category_ids)}

        # Build the list of (image_id, cat_id_or_None) items
        if category_id is not None:
            # Single-class mode: one item per image that contains category_id
            ann_img_ids = sorted({
                ann['image_id']
                for ann in self.coco.loadAnns(self.coco.getAnnIds(catIds=[category_id]))
            })
            self.items = [(img_id, category_id) for img_id in ann_img_ids]
        else:
            # Multi-class mode: one item per image that contains any target category
            img_ids = set()
            for cat_id in self.category_ids:
                for ann in self.coco.loadAnns(self.coco.getAnnIds(catIds=[cat_id])):
                    img_ids.add(ann['image_id'])
            self.items = [(img_id, None) for img_id in sorted(img_ids)]

        # Normalisation constants
        if sam_trans is not None:
            # SAM transform handles preprocessing; skip explicit normalisation
            self.mean = 0.0
            self.std = 1.0
        elif ds_mean is not None and ds_std is not None:
            self.mean = ds_mean
            self.std = ds_std
        else:
            # ImageNet defaults work well for COCO images
            self.mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            self.std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        self.size = len(self.items)

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img_id, cat_id = self.items[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = self._load_image(img_path)

        if cat_id is not None:
            mask = self._get_binary_mask(img_id, cat_id, img_info['height'], img_info['width'])
            case = self.cat_info.get(cat_id, {}).get('name', str(cat_id))
        else:
            mask = self._get_label_map(img_id, img_info['height'], img_info['width'])
            case = img_info['file_name']

        return self._process(image, mask, case)

    # ------------------------------------------------------------------
    # Support-set sampling  (mirrors PolypDataset.get_support)
    # ------------------------------------------------------------------

    def get_support(self, category_id: int, n_support: int = 1):
        """
        Sample n_support images that contain category_id and return their
        images + binary masks, ready to pass into ALPNetInput.

        Returns
        -------
        support_images : list of n_support tensors, each (1, C, H, W)
        support_labels : list of n_support tensors, each (1, H, W)
        case           : str  (category name)
        """
        img_ids = sorted({
            ann['image_id']
            for ann in self.coco.loadAnns(self.coco.getAnnIds(catIds=[category_id]))
        })

        if n_support > len(img_ids):
            raise ValueError(
                f"n_support={n_support} exceeds available images "
                f"({len(img_ids)}) for category {category_id}"
            )

        selected = random.sample(img_ids, n_support)
        support_images, support_labels = [], []
        case = self.cat_info.get(category_id, {}).get('name', str(category_id))

        for img_id in selected:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            image = self._load_image(img_path)
            mask = self._get_binary_mask(img_id, category_id, img_info['height'], img_info['width'])
            out = self._process(image, mask, case)
            support_images.append(out['image'].unsqueeze(0))
            support_labels.append(out['label'].unsqueeze(0))

        return support_images, support_labels, case

    def get_category_names(self) -> dict:
        """Returns {category_id: category_name} for all categories in scope."""
        return {cid: info['name'] for cid, info in self.cat_info.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)

    def _get_binary_mask(self, img_id: int, cat_id: int, h: int, w: int) -> torch.Tensor:
        """Union of all instance masks for cat_id in this image → binary (H, W) tensor."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        mask[mask > 0] = 1
        return torch.from_numpy(mask).float()

    def _get_label_map(self, img_id: int, h: int, w: int) -> torch.Tensor:
        """Integer label map: 0 = background, non-zero = category_id."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        label_map = np.zeros((h, w), dtype=np.int32)
        for ann in anns:
            if ann['category_id'] not in self.category_ids:
                continue
            rle = self.coco.annToMask(ann)
            label_map[rle > 0] = ann['category_id']
        return torch.from_numpy(label_map).float()

    def _process(self, image: torch.Tensor, mask: torch.Tensor, case: str) -> dict:
        original_size = tuple(image.shape[-2:])

        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        # Multi-class label maps (category_id=None mode) contain integer IDs and
        # must be resized with nearest-neighbor to avoid corrupting those values.
        is_multiclass = self.category_id is None

        if self.sam_trans is not None:
            image = self.sam_trans.apply_image_torch(image.unsqueeze(0))
            if is_multiclass:
                target_size = self.sam_trans.get_preprocess_shape(
                    mask.shape[-2], mask.shape[-1], self.sam_trans.target_length
                )
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest'
                ).squeeze(0).squeeze(0)
            else:
                mask = self.sam_trans.apply_image_torch(
                    mask.unsqueeze(0).unsqueeze(0).float()
                ).squeeze(0).squeeze(0)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
        else:
            image = (image - self.mean) / self.std
            image = F.interpolate(
                image.unsqueeze(0), size=self.image_size,
                mode='bilinear', align_corners=False
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=self.image_size, mode='nearest'
            ).squeeze(0).squeeze(0)

        if self.sam_trans is not None:
            image = self.sam_trans.preprocess(image).squeeze(0)
            mask = self.sam_trans.preprocess(mask)

        return {
            'image': image,
            'label': mask,
            'original_size': torch.tensor(original_size, dtype=torch.float),
            'image_size': torch.tensor(self.image_size, dtype=torch.float),
            'case': case,
        }


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def get_coco_dataset(
    image_dir: str,
    ann_file_train: str,
    ann_file_val: str,
    image_size: tuple = (1024, 1024),
    category_ids: list = None,
    category_id: int = None,
    sam_trans=None,
):
    """
    Build train and validation COCODataset instances.

    Parameters
    ----------
    image_dir      : shared image directory (or pass separate dirs per split)
    ann_file_train : annotation JSON for training split
    ann_file_val   : annotation JSON for validation/test split

    Returns
    -------
    ds_train, ds_val : COCODataset instances
    """
    ds_train = COCODataset(
        image_dir=image_dir,
        ann_file=ann_file_train,
        image_size=image_size,
        category_ids=category_ids,
        category_id=category_id,
        sam_trans=sam_trans,
        train=True,
    )
    ds_val = COCODataset(
        image_dir=image_dir,
        ann_file=ann_file_val,
        image_size=image_size,
        category_ids=category_ids,
        category_id=category_id,
        sam_trans=sam_trans,
        train=False,
    )
    return ds_train, ds_val
