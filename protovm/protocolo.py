"""Acesso ao protocolo congelado (protocolo/polipos_v1.json).

Todos os caminhos são resolvidos a partir de data/PolypDataset, sem /workspace.
"""
from functools import lru_cache
from pathlib import Path
import os

from protovm import RAIZ
from protovm.util import ler_json

ARQUIVO = RAIZ / 'protocolo' / 'polipos_v1.json'


@lru_cache(maxsize=1)
def carregar():
    return ler_json(ARQUIVO)


def raiz_dados():
    """data/PolypDataset do repositório; PROTOVM_DADOS sobrescreve (ex.: verificação local)."""
    return Path(os.environ.get('PROTOVM_DADOS', RAIZ / carregar()['raiz_dados']))


def imagem_treino(nome):
    return raiz_dados() / carregar()['pasta_treino']['imagens'] / nome


def mascara_treino(nome):
    return raiz_dados() / carregar()['pasta_treino']['mascaras'] / nome


def pool_eft():
    return list(carregar()['pool_eft']['nomes'])


def nomes(conjunto):
    """'V' (145 de validação) ou 'T' (100 do teste Kvasir)."""
    return list(carregar()['conjuntos'][conjunto]['nomes'])


def suporte(id_suporte='S00'):
    """Nome da imagem de suporte. 'K1' e 'S00' são a mesma imagem."""
    if id_suporte == 'K1':
        id_suporte = 'S00'
    return carregar()['suportes']['ids'][id_suporte]


def ids_suportes():
    return sorted(carregar()['suportes']['ids'])
