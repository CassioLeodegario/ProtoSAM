#!/bin/bash
# Step 1 of distillation: generate pseudo-labels for the polyp TrainDataset
# using DINOv2-L + ProtoSAM (the strongest baseline in the comparison table).
#
# Output: PNG masks under data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l/
# mirroring the images/ folder structure (Kvasir/, CVC-ClinicDB/, ...).
#
# Cost: ~1h on a single GPU (1305 images, ~3s each through the SAM-h pipeline).

set -e
unset CUDA_VISIBLE_DEVICES
GPUID=0

OUTPUT_DIR="data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l"

python generate_pseudo_labels.py with \
    modelname=dinov2_l14 \
    'model.which_model'=dinov2_l14 \
    base_model=alpnet \
    clsname=grid_proto \
    'model.cls_name'=grid_proto \
    protosam_sam_ver=sam_h \
    coarse_pred_only=False \
    do_cca=True \
    use_bbox=True \
    use_points=True \
    use_mask=False \
    use_neg_points=False \
    point_mode=both \
    n_support=1 \
    'support_idx=[6]' \
    val_wsize=2 \
    proto_grid_size=8 \
    seed=42 \
    dataset=polyps \
    output_dir="$OUTPUT_DIR" \
    overwrite_pseudo=False \
    save_at_original_size=True \
    2>&1 | tee logs_gen_pseudo_dinov2_l.txt
