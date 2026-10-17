"""Treino EFT-SP a partir de uma config.

    python scripts/treinar_eft.py configs/eft/tiny_512.yaml
    python scripts/treinar_eft.py configs/eft/tiny_512.yaml --retomar runs/<pasta>

Rode dentro de um `screen`; a saída também fica em runs/<pasta>/passos.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import config, eft_treino  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('config')
p.add_argument('--retomar', default=None, help='pasta em runs/ de uma execução interrompida')
a = p.parse_args()

cfg = config.carregar(a.config)
print(json.dumps(cfg, indent=2, ensure_ascii=False))
eft_treino.treinar(cfg, pasta=a.retomar)
