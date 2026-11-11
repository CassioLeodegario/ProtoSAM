"""Gerador de tabelas: seleção no registro, diferença pareada e recusas. Sem GPU.

    python tests/test_tabelas.py
"""
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from protovm import registro, tabelas  # noqa: E402


def linha(i, suporte, modelo, sam, estado='novo', cca='0'):
    l = {c: '' for c in registro.COLUNAS}
    l.update(id=f'R{i:04d}', estado=estado, modelo=modelo, conjunto='V', suporte=suporte, cca=cca,
             dice_alp=str(sam - 0.1), dice_sam=str(sam))
    return l


LINHAS = [linha(1, 'S00', 'a', 0.50), linha(2, 'S01', 'a', 0.40), linha(3, 'S00', 'b', 0.60),
          linha(4, 'S01', 'b', 0.70), linha(5, 'S02', 'b', 0.90),
          linha(6, 'S00', 'c', 0.99, estado='rascunho'), linha(7, 'S00', 'a', 0.10, cca='1')]


def recusa(funcao, *args, **kwargs):
    try:
        funcao(*args, **kwargs)
    except tabelas.TabelaInvalida as erro:
        return str(erro)
    raise AssertionError('deveria ter recusado')


def test_selecao_respeita_estado_conjunto_e_cca():
    assert sorted(tabelas.selecionar(LINHAS, {'modelo': 'a'}, 'V', 0)) == ['S00', 'S01']
    assert tabelas.selecionar(LINHAS, {'modelo': 'a'}, 'V', 1)['S00']['id'] == 'R0007'
    assert 'Nenhuma linha' in recusa(tabelas.selecionar, LINHAS, {'modelo': 'c'}, 'V', 0)  # rascunho não entra
    assert 'Nenhuma linha' in recusa(tabelas.selecionar, LINHAS, {'modelo': 'a'}, 'T', 0)
    assert 'inexistentes' in recusa(tabelas.selecionar, LINHAS, {'modeloo': 'a'}, 'V', 0)


def test_filtro_ambiguo_e_recusado():
    duplicada = LINHAS + [linha(8, 'S00', 'a', 0.55)]
    assert 'R0001 e R0008' in recusa(tabelas.selecionar, duplicada, {'modelo': 'a'}, 'V', 0)


def test_diferenca_pareada_usa_so_suportes_comuns():
    d = tabelas.diferenca_pareada({'S00': 0.6, 'S01': 0.7, 'S02': 0.9}, {'S00': 0.5, 'S01': 0.4})
    assert d['n'] == 2 and abs(d['media'] - 0.2) < 1e-9 and d['positivos'] == 2
    assert d['ic'][0] <= d['media'] <= d['ic'][1] and 0.1 - 1e-9 <= d['ic'][0] and d['ic'][1] <= 0.3 + 1e-9
    assert d == tabelas.diferenca_pareada({'S00': 0.6, 'S01': 0.7}, {'S00': 0.5, 'S01': 0.4})  # semente fixa


def test_tabela_em_markdown():
    spec = {'titulo': 't', 'referencia': 'A', 'metricas': ['dice_sam'],
            'linhas': [{'nome': 'A', 'filtro': {'modelo': 'a'}}, {'nome': 'B', 'filtro': {'modelo': 'b'}}]}
    texto = tabelas.gerar(spec, LINHAS)
    assert '| A | 2 | 0,450 ± 0,071 | — | — |' in texto, texto
    assert '| B | 3 | 0,733 ± 0,153 | +0,200 [' in texto and '| 2/2 |' in texto, texto
    assert 'referência' in recusa(tabelas.gerar, dict(spec, referencia='Z'), LINHAS)
    # Uma linha pode fixar a própria CCA: a mesma configuração com a opção desligada e ligada.
    par = {'titulo': 't', 'referencia': 'sem', 'metricas': ['dice_sam'],
           'linhas': [{'nome': 'sem', 'cca': 0, 'filtro': {'modelo': 'a'}}, {'nome': 'com', 'cca': 1, 'filtro': {'modelo': 'a'}}]}
    texto = tabelas.gerar(par, LINHAS)
    assert 'CCA indicada em cada linha' in texto and '| com | 1 | 0,100 ± 0,000 | -0,400 [' in texto, texto


def test_exemplo_do_repositorio_esta_atualizado():
    pasta = RAIZ / 'experimentos' / 'E00_exemplo'
    spec = yaml.safe_load((pasta / 'tabela.yaml').read_text(encoding='utf-8'))
    assert tabelas.gerar(spec) == (pasta / 'tabela.md').read_text(encoding='utf-8'), \
        'tabela.md difere do que o registro gera; rode scripts/gerar_tabelas.py experimentos/E00_exemplo'


if __name__ == '__main__':
    for nome, funcao in list(globals().items()):
        if nome.startswith('test_'):
            funcao()
            print('ok  ', nome)
    print('OK: gerador de tabelas.')
