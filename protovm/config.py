"""Leitura de configs YAML com herança simples: `base: <arquivo>` mescla o pai primeiro."""
from pathlib import Path

import yaml

from protovm import RAIZ


def carregar(caminho):
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = (Path.cwd() / caminho) if (Path.cwd() / caminho).exists() else RAIZ / caminho
    cfg = yaml.safe_load(caminho.read_text(encoding='utf-8')) or {}
    pai = cfg.pop('base', None)
    nome = cfg.get('nome', caminho.stem)  # o nome é do próprio arquivo, nunca herdado do pai
    if pai:
        cfg = {**carregar(caminho.parent / pai), **cfg}
    cfg['nome'] = nome
    return cfg
