#!/bin/bash
# Step 2 of distillation: train a VMamba-Tiny student against the pseudo-labels
# produced by run_generate_pseudo_labels.sh.
#
# Output: a FewShotSeg checkpoint (vmamba_tiny encoder + grid_proto cls_unit)
# that validation_protosam.py can load via reload_model_path.
#
# Cost target: ~3h. With image_size=256 and batch=4, a VMamba-Tiny step is
# ~0.4-0.6s, so 3000 steps ~= 25-35 min — leaves plenty of room to grow.

set -e
# Let CUDA use whichever GPU the environment exposes — RunPod may allocate
# any device index (e.g. /dev/nvidia6). Forcing CUDA_VISIBLE_DEVICES=0
# hides the actual device when the index doesn't match.
unset CUDA_VISIBLE_DEVICES
GPUID=0

IMAGE_ROOT="data/PolypDataset/TrainDataset/images"
PSEUDO_ROOT="data/PolypDataset/TrainDataset/pseudo_masks_dinov2_l"
OUTPUT_DIR="runs/distill_vmamba_tiny"

if [ ! -d "$PSEUDO_ROOT" ]; then
    echo "ERROR: $PSEUDO_ROOT not found. Run run_generate_pseudo_labels.sh first."
    exit 1
fi

python train_distillation.py \
    --gpu-id $GPUID \
    --modelname vmamba_tiny \
    --image-root "$IMAGE_ROOT" \
    --pseudo-root "$PSEUDO_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --image-size 512 \
    --batch-size 2 \
    --num-workers 4 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --n-steps 10000 \
    --warmup-steps 500 \
    --save-every 2000 \
    --print-every 200 \
    --ce-weight 1.0 \
    --dice-weight 1.0 \
    --seed 42 \
    --freeze-stages 2 \
    2>&1 | tee logs_distill_vmamba_tiny.txt
