#!/bin/bash
set -e
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

MODEL_NAME="vmamba_base"
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
RELOAD_PATH=( "None" )
SKIP_SLICES="True"
DO_CCA="True"
ALL_SCALE=( "MIDDLE")

FREE_DESC=""
CPT="${MODEL_NAME}_${MODALITY}_cca_grid_${PROTO_GRID}_res_${INPUT_SIZE}_${ORGAN}_fold"

SUPP_ID='[927]'

echo ===================================

for ((i=0; i<${#ALL_EV[@]}; i++))
do
    EVAL_FOLD=${ALL_EV[i]}
    CPT_W_FOLD="${CPT}_${EVAL_FOLD}"
    echo $CPT_W_FOLD on GPU $GPUID1
    PREFIX="test_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./test_${MODALITY}_vmamba/${CPT_W_FOLD}"

    if [ ! -d $LOGDIR ]; then
        mkdir -p $LOGDIR
    fi

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
            reload_model_path=${RELOAD_PATH[i]} \
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
            "input_size=($INPUT_SIZE, $INPUT_SIZE)"
    done
done
