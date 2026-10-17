#!/bin/bash
# Cria os links de dados e pesos do ProtoSAM-vmamba. Rode uma vez após clonar.
#   bash scripts/ligar_dados.sh <pasta que contém ProtoSAM/ e polyp_raw/>
# Ex.: bash scripts/ligar_dados.sh ~/cassio_pereira
set -e
ORIGEM=$(realpath "${1:?informe a pasta com ProtoSAM/ e polyp_raw/}")
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
P="$ORIGEM/ProtoSAM/data/PolypDataset"
R="$ORIGEM/polyp_raw"

# No pod esses três eram links; o download pelo S3 não preserva links.
[ -L "$P/TestDataset/test" ] || { rmdir "$P/TestDataset/test" 2>/dev/null || true; ln -s "$R/TestDataset/TestDataset/Kvasir" "$P/TestDataset/test"; }
[ -e "$P/TrainDataset/images" ] || ln -s "$R/TrainDataset/TrainDataset/image" "$P/TrainDataset/images"
[ -e "$P/TrainDataset/masks" ]  || ln -s "$R/TrainDataset/TrainDataset/mask"  "$P/TrainDataset/masks"

mkdir -p "$RAIZ/data"
[ -e "$RAIZ/data/PolypDataset" ] || ln -s "$P" "$RAIZ/data/PolypDataset"
[ -e "$RAIZ/pretrained_model" ]  || ln -s "$ORIGEM/ProtoSAM/pretrained_model" "$RAIZ/pretrained_model"

echo "treino: $(ls "$RAIZ/data/PolypDataset/TrainDataset/images" | wc -l) (esperado 1450)"
echo "teste:  $(ls "$RAIZ/data/PolypDataset/TestDataset/test/images" | wc -l) (esperado 100)"
ls "$RAIZ/pretrained_model"
