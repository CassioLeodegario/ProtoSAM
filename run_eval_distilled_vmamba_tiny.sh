#!/bin/bash
# Step 3 of distillation: evaluate the distilled VMamba-Tiny student inside
# the same ProtoSAM few-shot pipeline used for the baselines, so the numbers
# are directly comparable to:
#   - VMamba-T (no pretraining): Dice 0.4131
#   - VMamba-T + EFT (superpixel SSL): Dice 0.1609
#   - DINOv2-L (teacher): Dice 0.6512

set -e
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

MODEL_NAME='vmamba_tiny'
COARSE_PRED_ONLY="False"
PROTOSAM_SAM_VER="sam_h"
INPUT_SIZE=672
ORGAN="polyps"
MODALITY="polyp"
DATASET='polyps'

PROTO_GRID=8
ALL_EV=( 0 )
SEED=42
NWORKER=4
LORA=0

# === EDIT THIS to point at the distilled checkpoint ===
RELOAD_PATH=( "runs/distill_vmamba_tiny/distilled_vmamba_tiny_final.pth" )
# ======================================================

DO_CCA="True"
ALL_SCALE=( "MIDDLE" )

CPT="${MODEL_NAME}_distill_cca_grid_${PROTO_GRID}_res_${INPUT_SIZE}_${ORGAN}_fold"
SUPP_ID='[6]'

echo ===================================
echo "Loading distilled checkpoint: ${RELOAD_PATH[0]}"
[ -f "${RELOAD_PATH[0]}" ] || { echo "ERROR: checkpoint not found"; exit 1; }
echo ===================================

for ((i=0; i<${#ALL_EV[@]}; i++))
do
    EVAL_FOLD=${ALL_EV[i]}
    CPT_W_FOLD="${CPT}_${EVAL_FOLD}"
    PREFIX="test_distill_vfold${EVAL_FOLD}"
    LOGDIR="./test_${MODALITY}_vmamba_distill/${CPT_W_FOLD}"
    mkdir -p "$LOGDIR"

    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
        python3 validation_protosam.py with \
            "modelname=$MODEL_NAME" \
            "base_model=alpnet" \
            "coarse_pred_only=$COARSE_PRED_ONLY" \
            "protosam_sam_ver=$PROTOSAM_SAM_VER" \
            "curr_cls=$ORGAN" \
            'usealign=False' \
            'optim_type=sgd' \
            reload_model_path="${RELOAD_PATH[i]}" \
            num_workers=$NWORKER \
            scan_per_load=-1 \
            'use_wce=True' \
            exp_prefix=$PREFIX \
            'clsname=grid_proto' \
            eval_fold=$EVAL_FOLD \
            dataset=$DATASET \
            proto_grid_size=$PROTO_GRID \
            min_fg_data=1 seed=$SEED \
            superpix_scale=$SUPERPIX_SCALE \
            path.log_dir=$LOGDIR \
            support_idx=$SUPP_ID \
            lora=$LORA \
            "do_cca=$DO_CCA" \
            "input_size=($INPUT_SIZE, $INPUT_SIZE)" \
            2>&1 | tee "logs_eval_distill_${MODEL_NAME}_fold${EVAL_FOLD}.txt"
    done
done
