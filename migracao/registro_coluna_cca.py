"""Acrescenta a coluna `cca` ao registro (nov/2026). Usado uma vez.

Todas as linhas anteriores rodaram com a CCA desligada: é o protocolo de out/2026 (PROTOCOLO.md,
seção Inferência) e o `use_cca: false` gravado nos config.json do ProtoSAM-repro. A regravação
preserva as demais colunas campo a campo; o script confere isso antes de terminar.

    python migracao/registro_coluna_cca.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import registro  # noqa: E402

antes = registro.ler()
assert antes, 'Registro vazio.'
assert 'cca' not in antes[0], 'O registro já tem a coluna cca.'
assert set(antes[0]) == set(registro.COLUNAS) - {'cca'}, set(antes[0]) ^ set(registro.COLUNAS)

with registro.ARQUIVO.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=registro.COLUNAS)
    w.writeheader()
    for linha in antes:
        w.writerow(dict(linha, cca='0'))

depois = registro.ler()
assert registro.cabecalho() == registro.COLUNAS and len(depois) == len(antes)
for a, d in zip(antes, depois):
    assert d['cca'] == '0' and {c: v for c, v in d.items() if c != 'cca'} == a, a['id']
print(f'{len(depois)} linhas regravadas com cca=0; as demais colunas não mudaram.')
