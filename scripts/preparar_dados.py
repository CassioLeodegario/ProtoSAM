"""Gera o cache de episódios EFT-SP (CPU, sem GPU).

    python scripts/preparar_dados.py --resolucao 256
    python scripts/preparar_dados.py --resolucao 512 --superpixels-em 256

Destino padrão: dados_eft/r<resolucao>[_sp<superpixels_em>]. Reexecutar retoma.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import RAIZ, eft_dados  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--resolucao', type=int, default=256)
p.add_argument('--superpixels-em', type=int, default=None)
p.add_argument('--destino', default=None)
a = p.parse_args()

sp = a.superpixels_em or a.resolucao
destino = a.destino or RAIZ / 'dados_eft' / (f'r{a.resolucao}' + (f'_sp{sp}' if sp != a.resolucao else ''))
eft_dados.preparar(destino, resolucao=a.resolucao, superpixels_em=sp)
