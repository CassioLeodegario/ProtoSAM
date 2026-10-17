"""Confere, em CPU, que protovm.eft_dados reproduz os dados de out/2026.

Compara com o ProtoSAM-repro: candidatos de superpixel de cada imagem,
plano de validação, os 200 episódios do piloto e os 5000 do treino longo.

    python migracao/verificar_dados_256.py <ProtoSAM-repro> [--destino dados_eft/r256]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import RAIZ, eft_dados  # noqa: E402
from protovm.util import ler_json  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('repro')
p.add_argument('--destino', default=str(RAIZ / 'dados_eft' / 'r256'))
a = p.parse_args()
repro = Path(a.repro) / 'reproducao'
antigo = repro / 'eft_sp_piloto_dados_20261011T152315570066Z'

destino = eft_dados.preparar(a.destino, resolucao=256)

# 1. Candidatos por imagem (dependem de Felzenszwalb e dos filtros).
divergentes = []
for d in ler_json(destino / 'divisao.json'):
    novo = ler_json(destino / 'cache' / f"{d['indice']:04d}.json")
    velho = ler_json(antigo / 'cache' / f"{d['indice']:04d}.json")
    assert novo['nome'] == velho['nome'] and novo['split'] == velho['split']
    if novo['candidatos'] != velho['candidatos']:
        divergentes.append(novo['nome'])
print(f'Candidatos: {1449 - len(divergentes)}/1449 imagens idênticas')


def chave(plano):
    return [(r['indice'], r['regiao']) for r in plano]


planos = eft_dados.planos(destino, 5000)
velho_piloto = ler_json(antigo / 'planos.json')
velho_5000 = ler_json(repro / 'eft_tiny_5000_20261011T224215171455Z' / 'planos.json')
print('Validação (145):', chave(planos['validacao']) == chave(velho_piloto['validacao']))
print('Piloto (200):   ', chave(planos['treino'][:200]) == chave(velho_piloto['treino']))
print('Treino (5000):  ', chave(planos['treino']) == chave(velho_5000['treino']))
assert not divergentes, f'Imagens com candidatos diferentes: {divergentes[:5]}'
