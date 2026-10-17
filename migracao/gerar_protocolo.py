"""Congela o protocolo de pólipos a partir dos registros do ProtoSAM-repro.

Executado uma vez na migração. Guarda apenas nomes de arquivo, sem caminhos
absolutos; protovm.protocolo resolve os caminhos a partir de data/PolypDataset.

Uso: python migracao/gerar_protocolo.py <pasta ProtoSAM-repro>
"""
import csv
import json
import sys
from pathlib import Path

ANTIGO = Path(sys.argv[1] if len(sys.argv) > 1 else '../ProtoSAM-repro')
REPRO = ANTIGO / 'reproducao'
DESTINO = Path(__file__).resolve().parents[1] / 'protocolo' / 'polipos_v1.json'


def ler(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


manifesto = ler(REPRO / 'protocolo_polipos_20261011T143313814944Z/manifesto.json')
divisao = ler(REPRO / 'eft_sp_piloto_dados_20261011T152315570066Z/divisao.json')
plano_multi = ler(REPRO / 'multissuporte_tiny_eft2000_kvasir_v1/plano.json')
with (REPRO / '01_avaliacao_K1_20261004T140725337013Z/manifesto_teste.csv').open(encoding='utf-8') as f:
    teste_kvasir = [linha['query'] for linha in csv.DictReader(f)]

pool = [r['nome'] for r in divisao]
assert pool == sorted(pool) and len(pool) == 1449
assert [r['nome'] for r in sorted(manifesto['pool_adaptacao'], key=lambda r: (r['nome'], r['imagem']))] == pool
validacao = [r['nome'] for r in divisao if r['split'] == 'validacao']
assert len(validacao) == 145
suporte_k1 = manifesto['suporte_k1']['nome']
assert suporte_k1 not in pool
suportes = {s['id']: s['meta']['nome'] for s in plano_multi['suportes']}
assert suportes['S00'] == suporte_k1 and len(suportes) == 20
assert not set(suportes.values()) & set(validacao)
assert len(teste_kvasir) == 100

protocolo = {
    'versao': 1,
    'descricao': 'Protocolo one-shot de pólipos usado desde out/2026 no ProtoSAM-repro.',
    'raiz_dados': 'data/PolypDataset',
    'pasta_treino': {'imagens': 'TrainDataset/images', 'mascaras': 'TrainDataset/masks'},
    'pasta_teste_kvasir': {'imagens': 'TestDataset/test/images', 'mascaras': 'TestDataset/test/masks'},
    'suporte_k1': suporte_k1,
    'suporte_k1_indice_dataset_treino': 927,
    'suporte_k1_pixels_fg_canvas1024': 51940,
    'pool_eft': {
        'descricao': '1450 imagens de treino menos o suporte K1, ordenadas por nome',
        'semente_divisao': 1234,
        'nomes': pool,
    },
    'conjuntos': {
        'V': {'descricao': '145 imagens reservadas do pool; validação anotada para desenvolvimento',
              'nomes': validacao},
        'T': {'descricao': 'Teste oficial Kvasir (100 imagens); usar só na avaliação final',
              'nomes': teste_kvasir},
    },
    'suportes': {
        'descricao': 'S00 = K1; S01-S19 sorteados do treino com semente 20261014 (exp. 15)',
        'semente': plano_multi['semente'],
        'ids': suportes,
    },
}
DESTINO.write_text(json.dumps(protocolo, ensure_ascii=False, indent=1), encoding='utf-8')
print('Gravado:', DESTINO)
