"""Ajuste auto-supervisionado com regiões (SAM ou superpixels) como pseudo-máscaras, a 672 px.

    python scripts/treinar_regioes.py configs/eft_regioes/dinov2_s14_sam.yaml
    python scripts/treinar_regioes.py configs/eft_regioes/dinov2_s14_sam.yaml --run runs/<pasta>   # retoma
    python scripts/treinar_regioes.py configs/eft_regioes/vmamba_tiny_sam.yaml --max-passos 50      # ensaio

Rode num screen ou com nohup. Avaliação depois:
    python scripts/avaliar.py --treino runs/<pasta> --passos 4000 --suportes todos
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import config, eft_regioes  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('config')
p.add_argument('--run', default=None, help='pasta de um treino interrompido, para retomar')
p.add_argument('--max-passos', type=int, default=None, help='para antes do fim (ensaio)')
a = p.parse_args()
eft_regioes.treinar(config.carregar(a.config), pasta=a.run, max_passos=a.max_passos)
