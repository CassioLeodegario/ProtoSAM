#!/bin/bash
# EFT (Encoder Fine-Tuning) — VMamba-Tiny — Polyp dataset
# Usage: bash run_eft_vmamba_tiny.sh

set -e

DATASET_DIR="/workspace/polyp_raw/TrainDataset/TrainDataset"
PRETRAINED="/workspace/ProtoSAM/pretrained_model/vmamba_tiny_v2.pth"
BACKBONE="vmamba_tiny"
RUN_NAME="eft_vmamba_tiny_polyp"

echo "========================================"
echo " EFT — VMamba-Tiny — Polyp"
echo " Dataset : $DATASET_DIR"
echo " Backbone: $BACKBONE"
echo "========================================"

python training.py \
    with \
    dataset='Polyp_Superpix' \
    'path.Polyp_Superpix.data_dir'="$DATASET_DIR" \
    modelname="$BACKBONE" \
    'model.which_model'="$BACKBONE" \
    use_coco_init=False \
    'model.use_coco_init'=False \
    n_steps=100100 \
    lr=1e-3 \
    batch_size=1 \
    num_workers=4 \
    superpix_scale='MIDDLE' \
    proto_grid_size=8 \
    max_iters_per_load=1000 \
    save_snapshot_every=25000 \
    print_interval=100 \
    which_aug='sabs_aug' \
    exp_prefix="$RUN_NAME" \
    base_model='alpnet' \
    exclude_cls_list=[] \
    clsname='grid_proto' \
    'model.cls_name'='grid_proto' \
    2>&1 | tee "logs_${RUN_NAME}.txt"
