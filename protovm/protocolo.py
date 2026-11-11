"""Acesso ao protocolo congelado (protocolo/polipos_v1.json) e às travas que o protegem.

Todos os caminhos são resolvidos a partir de data/PolypDataset, sem /workspace.
Este módulo não importa torch: as travas podem ser testadas em qualquer máquina.
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


# ---------------------------------------------------------------------------------------------
# Travas: regras do protocolo conferidas pelo código, não pela memória de quem roda.

FINAIS = RAIZ / 'protocolo' / 'finais.yaml'
REFERENCIAS = RAIZ / 'protocolo' / 'referencias_v1.json'
TRAVADOS = ('T',)  # conjuntos que só rodam para uma configuração final já commitada
CAMPOS_FINAL = ('modelo', 'estagio', 'norm', 'resolucao', 'grade', 'cca', 'pesos_sha256')


class Trava(RuntimeError):
    """Uma regra do protocolo impede a execução."""


def referencias():
    """Valores de referência: estatísticas da entrada e avaliações curtas (tests/test_referencias.py)."""
    return ler_json(REFERENCIAS)


def finais():
    """Configurações autorizadas a rodar nos conjuntos travados, por nome (protocolo/finais.yaml)."""
    import yaml
    itens = yaml.safe_load(FINAIS.read_text(encoding='utf-8')) or []
    return {item['nome']: item for item in itens}


def item_final(final, git, itens=None):
    """Item de finais.yaml que autoriza uma avaliação em conjunto travado.

    git vem de util.git_commit(): o teste só roda com código, protocolo e configs commitados,
    de modo que a data do commit prove que a escolha veio antes do resultado.
    """
    if git['codigo_alterado'] or not git['commit']:
        raise Trava('Há alterações sem commit em código, protocolo ou configs. '
                    'O conjunto de teste só roda com tudo commitado.')
    if not final:
        raise Trava('O conjunto de teste exige --final NOME, um item de protocolo/finais.yaml.')
    itens = finais() if itens is None else itens
    if final not in itens:
        raise Trava(f"'{final}' não está em protocolo/finais.yaml. "
                    f"Itens: {', '.join(sorted(itens)) or 'nenhum'}.")
    return itens[final]


def conferir_final(item, descricao):
    """A configuração avaliada tem de ser exatamente a escrita em finais.yaml.

    descricao é alp.descricao mais 'cca' (a opção do SAM usada). Campo ausente no item vale None:
    passa onde a avaliação também tem None (estágio e norma do DINOv2, hash de modelo pré-treinado)
    e é recusado nos demais.
    """
    diferencas = [f'{c}: finais.yaml tem {item.get(c)!r}, a avaliação tem {descricao.get(c)!r}'
                  for c in CAMPOS_FINAL if item.get(c) != descricao.get(c)]
    if diferencas:
        raise Trava(f"A configuração difere do item '{item['nome']}' de finais.yaml:\n  "
                    + '\n  '.join(diferencas))


def diferencas_da_entrada(obtido, esperado=None):
    """Compara as estatísticas da imagem do suporte K1, como ela chega ao encoder, com as registradas.

    Devolve a lista de divergências (vazia se estiver tudo certo). Um pré-processamento alterado,
    como o estouro de uint8 que inverteu as imagens em mar/2026, desloca a média em décimos.
    """
    esperado = referencias()['entrada_k1'] if esperado is None else esperado
    saida = []
    for campo, tol in (('media', 'tolerancia'), ('desvio', 'tolerancia'), ('fracao_zeros', 'tolerancia'),
                       ('minimo', 'tolerancia_extremos'), ('maximo', 'tolerancia_extremos')):
        if abs(obtido[campo] - esperado[campo]) > esperado[tol]:
            saida.append(f'{campo}: esperado {esperado[campo]:.6f}, obtido {obtido[campo]:.6f}')
    for canal, (o, e) in enumerate(zip(obtido['media_por_canal'], esperado['media_por_canal'])):
        if abs(o - e) > esperado['tolerancia']:
            saida.append(f'média do canal {canal}: esperado {e:.6f}, obtido {o:.6f}')
    return saida
