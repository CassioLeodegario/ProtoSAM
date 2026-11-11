"""resultados/registro.csv: uma linha por avaliação ALP + SAM.

Toda avaliação feita por protovm.avaliar entra aqui automaticamente. As tabelas
de ablação são consultas sobre este arquivo (ver notebooks/01_resultados.ipynb).
"""
import csv
from datetime import date

from protovm import RAIZ

ARQUIVO = RAIZ / 'resultados' / 'registro.csv'
COLUNAS = [
    'id', 'data', 'estado',
    'modelo', 'pesos', 'passo', 'treino_estagio', 'treino_norm', 'treino_resolucao',
    'aval_estagio', 'aval_norm', 'aval_resolucao', 'grade',
    'conjunto', 'suporte', 'n', 'dice_alp', 'dice_sam',
    'gpu', 'commit', 'origem', 'obs',
]
# estado: 'novo' (gerado por protovm com o código commitado), 'rascunho' (gerado com alteração
#         sem commit; não entra em tabela), 'importado' (ProtoSAM-repro, out/2026)
# pesos:  'pre_treinado' ou 'eft:<nome do run de treino>'; o hash do arquivo fica no config.json do run
# conjunto: 'V' (145 validação), 'T' (100 teste Kvasir)


def ler():
    if not ARQUIVO.exists():
        return []
    with ARQUIVO.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def adicionar(linha):
    extras = set(linha) - set(COLUNAS)
    assert not extras, f'Colunas desconhecidas no registro: {extras}'
    linhas = ler()
    linha = {c: linha.get(c, '') for c in COLUNAS}
    linha['id'] = linha['id'] or f'R{len(linhas) + 1:04d}'
    linha['data'] = linha['data'] or date.today().isoformat()
    for c in ('dice_alp', 'dice_sam'):
        if linha[c] != '':
            linha[c] = f'{float(linha[c]):.6f}'
    novo = not ARQUIVO.exists()
    ARQUIVO.parent.mkdir(exist_ok=True)
    with ARQUIVO.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            w.writeheader()
        w.writerow(linha)
    return linha['id']


def tabela():
    """registro como DataFrame com tipos numéricos."""
    import pandas as pd
    df = pd.read_csv(ARQUIVO, dtype=str).fillna('')
    for c in ('dice_alp', 'dice_sam', 'n', 'passo', 'aval_resolucao', 'treino_resolucao', 'grade'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
