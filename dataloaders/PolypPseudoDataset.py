"""
Helper that wires PolypDataset to a directory of pseudo-labels produced by
generate_pseudo_labels.py. The structure under pseudo_root mirrors the
original masks/ folder, so PolypDataset's pairing logic works unchanged.
"""
import os

from dataloaders.PolypDataset import PolypDataset, DATASETS
from dataloaders.PolypTransforms import get_polyp_transform


def get_polyp_pseudo_dataset(
    image_root,
    pseudo_root,
    image_size=(256, 256),
    augment=True,
    datasets=DATASETS,
    sam_trans=None,
):
    """Returns a PolypDataset whose `gts` are pseudo-masks instead of GT."""
    transform_train, transform_test = get_polyp_transform()
    aug = transform_train if augment else transform_test
    if not os.path.isdir(pseudo_root):
        raise FileNotFoundError(
            f"pseudo_root not found: {pseudo_root}. "
            "Run generate_pseudo_labels.py first."
        )
    return PolypDataset(
        root=image_root,
        image_root=image_root,
        gt_root=pseudo_root,
        augmentations=aug,
        train=True,
        sam_trans=sam_trans,
        image_size=image_size,
        datasets=datasets,
    )
