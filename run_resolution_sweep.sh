#!/bin/bash
# Avalia DINOv2-L e VMamba-Tiny distilado em múltiplas resoluções (256→1024).
# Cada combinação (modelo × resolução) gera uma run separada no W&B.
#
# Hipótese: VMamba (SSM) deve escalar melhor que ViT em resoluções altas.
#
# Uso:
#   bash run_resolution_sweep.sh
#   bash run_resolution_sweep.sh --only-dinov2
#   bash run_resolution_sweep.sh --only-vmamba

set -eo pipefail
unset CUDA_VISIBLE_DEVICES

FAILED_RUNS=()

run_eval() {
    local label=$1; shift
    if python3 "$@"; then
        echo "[OK] $label"
    else
        echo "[SKIP] $label falhou (OOM ou erro) — continuando"
        FAILED_RUNS+=("$label")
    fi
}
GPUID=0

SIZES=(256 384 512 672 768 1024)

ONLY_DINOV2=0
ONLY_VMAMBA=0
for arg in "$@"; do
    case $arg in
        --only-dinov2) ONLY_DINOV2=1 ;;
        --only-vmamba) ONLY_VMAMBA=1 ;;
    esac
done

# === EDITE AQUI: caminho do checkpoint distilado ===
VMAMBA_CKPT="runs/distill_vmamba_tiny/distilled_vmamba_tiny_512_10k.pth"
# ===================================================

if [ ! -f "$VMAMBA_CKPT" ] && [ $ONLY_DINOV2 -eq 0 ]; then
    echo "AVISO: checkpoint VMamba não encontrado em '$VMAMBA_CKPT'"
    echo "  Atualize a variável VMAMBA_CKPT ou rode com --only-dinov2"
    exit 1
fi

# ---------------------------------------------------------------------------
# Sweep intercalado: DINOv2 e VMamba por resolução crescente
# ---------------------------------------------------------------------------
echo "=========================================="
echo "  Sweep intercalado  |  ${#SIZES[@]} resoluções × 2 modelos"
echo "=========================================="
for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "========== size=${SIZE} =========="

    if [ $ONLY_VMAMBA -eq 0 ]; then
        echo "--- DINOv2-L  size=${SIZE} ---"
        LOGDIR="./test_polyp/sweep/dinov2_l14_size_${SIZE}"
        mkdir -p "$LOGDIR"
        run_eval "dinov2_l14_size${SIZE}" \
            validation_protosam.py with \
            modelname=dinov2_l14 \
            base_model=alpnet \
            coarse_pred_only=False \
            protosam_sam_ver=sam_h \
            curr_cls=polyps \
            usealign=True \
            optim_type=sgd \
            reload_model_path=None \
            num_workers=4 \
            scan_per_load=-1 \
            use_wce=True \
            exp_prefix="sweep_dinov2_size${SIZE}" \
            clsname=grid_proto \
            eval_fold=0 \
            dataset=polyps \
            proto_grid_size=8 \
            min_fg_data=1 \
            seed=42 \
            superpix_scale=MIDDLE \
            path.log_dir="$LOGDIR" \
            'support_idx=[6]' \
            lora=0 \
            do_cca=True \
            "input_size=($SIZE, $SIZE)" \
            wandb_project=protosam-polyp-sizes \
            2>&1 | tee "logs_sweep_dinov2_l14_size${SIZE}.txt"
    fi

    if [ $ONLY_DINOV2 -eq 0 ]; then
        echo "--- VMamba-Tiny distil  size=${SIZE} ---"
        LOGDIR="./test_polyp/sweep/vmamba_tiny_distill_size_${SIZE}"
        mkdir -p "$LOGDIR"
        run_eval "vmamba_tiny_size${SIZE}" \
            validation_protosam.py with \
            modelname=vmamba_tiny \
            base_model=alpnet \
            coarse_pred_only=False \
            protosam_sam_ver=sam_h \
            curr_cls=polyps \
            usealign=False \
            optim_type=sgd \
            reload_model_path="$VMAMBA_CKPT" \
            num_workers=4 \
            scan_per_load=-1 \
            use_wce=True \
            exp_prefix="sweep_vmamba_size${SIZE}" \
            clsname=grid_proto \
            eval_fold=0 \
            dataset=polyps \
            proto_grid_size=8 \
            min_fg_data=1 \
            seed=42 \
            superpix_scale=MIDDLE \
            path.log_dir="$LOGDIR" \
            'support_idx=[6]' \
            lora=0 \
            do_cca=True \
            "input_size=($SIZE, $SIZE)" \
            wandb_project=protosam-polyp-sizes \
            2>&1 | tee "logs_sweep_vmamba_tiny_size${SIZE}.txt"
    fi
done

echo ""
echo "Sweep concluído. Resultados no W&B: https://wandb.ai/leodegario/protosam-polyp-sizes"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Runs que falharam (OOM ou erro):"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
fi
