#!/bin/bash
# Validation — VMamba-Tiny on Polyps with the EFT checkpoint loaded.
#
# Before running, set RELOAD_PATH to the .pth produced by run_eft_vmamba_tiny.sh.
# Snapshots are saved at:
#   runs/mySSL_eft_vmamba_tiny_polyp_Polyp_Superpix_sets_0_1shot/<run_id>/snapshots/<step>.pth
# With save_snapshot_every=1000 and n_steps=5500 you should have:
<<<<<<< HEAD
#   1000.pth, 2000.pth, 3000.pth, 4000.pth, 5000.pth
# Use the latest (5000.pth) unless you want to compare earlier ones.
=======
#   1001.pth, 2001.pth, 3001.pth, 4001.pth, 5001.pth
# Use the latest (5001.pth) unless you want to compare earlier ones.
>>>>>>> 900fff8f2897625990707a42f4d4b94a09c77c4e
#
# Quick way to find the run_id of the most recent EFT run:
#   ls -1tr runs/mySSL_eft_vmamba_tiny_polyp_Polyp_Superpix_sets_0_1shot/ | tail -n1

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

# === EDIT THIS to your EFT snapshot ===
<<<<<<< HEAD
RUN_ID="13"    # check: ls runs/mySSL_eft_vmamba_tiny_polyp_Polyp_Superpix_sets_0_1shot/
STEP="5000"    # 1000 / 2000 / 3000 / 4000 / 5000
=======
RUN_ID="1"     # check: ls runs/mySSL_eft_vmamba_tiny_polyp_Polyp_Superpix_sets_0_1shot/
STEP="5001"    # 1001 / 2001 / 3001 / 4001 / 5001
>>>>>>> 900fff8f2897625990707a42f4d4b94a09c77c4e
RELOAD_PATH=( "runs/mySSL_eft_vmamba_tiny_polyp_Polyp_Superpix_sets_0_1shot/${RUN_ID}/snapshots/${STEP}.pth" )
# ======================================

SKIP_SLICES="True"
DO_CCA="True"
ALL_SCALE=( "MIDDLE" )

CPT="${MODEL_NAME}_${MODALITY}_eft_${STEP}_cca_grid_${PROTO_GRID}_res_${INPUT_SIZE}_${ORGAN}_fold"

SUPP_ID='[6]'

echo ===================================
echo "Loading EFT checkpoint: ${RELOAD_PATH[0]}"
[ -f "${RELOAD_PATH[0]}" ] || { echo "ERROR: checkpoint not found"; exit 1; }
echo ===================================

for ((i=0; i<${#ALL_EV[@]}; i++))
do
    EVAL_FOLD=${ALL_EV[i]}
    CPT_W_FOLD="${CPT}_${EVAL_FOLD}"
    echo $CPT_W_FOLD on GPU $GPUID1
    PREFIX="test_eft_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./test_${MODALITY}_vmamba_eft/${CPT_W_FOLD}"

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
            2>&1 | tee "logs_eval_eft_${MODEL_NAME}_${STEP}_fold${EVAL_FOLD}_${SUPERPIX_SCALE}.txt"
    done
done
