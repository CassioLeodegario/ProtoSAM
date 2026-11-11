"""Gera as tabelas de um experimento a partir do registro.

    python scripts/gerar_tabelas.py experimentos/E00_exemplo            # todos os tabela*.yaml da pasta
    python scripts/gerar_tabelas.py experimentos/E00_exemplo/tabela.yaml

Cada tabela*.yaml vira um .md de mesmo nome ao lado. Formato do YAML: ver protovm/tabelas.py.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import tabelas  # noqa: E402

alvo = Path(sys.argv[1])
arquivos = sorted(alvo.glob('tabela*.yaml')) if alvo.is_dir() else [alvo]
assert arquivos, f'Nenhum tabela*.yaml em {alvo}'
for arquivo in arquivos:
    try:
        texto = tabelas.gerar(yaml.safe_load(arquivo.read_text(encoding='utf-8')))
    except tabelas.TabelaInvalida as erro:
        sys.exit(f'{arquivo}: {erro}')
    destino = arquivo.with_suffix('.md')
    destino.write_text(texto, encoding='utf-8', newline='\n')
    print(f'{destino} gerado')
