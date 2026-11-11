"""Travas do protocolo: o que o código recusa ou marca sozinho. Roda em qualquer máquina, sem GPU.

    python tests/test_travas.py   (ou: python -m pytest tests/test_travas.py)
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from protovm import protocolo  # noqa: E402
from protovm.util import estado_do_codigo, git_commit, nome_do_treino  # noqa: E402

LIMPO = {'commit': 'abc123', 'codigo_alterado': False}
SUJO = {'commit': 'abc123', 'codigo_alterado': True}
ITEM = {'nome': 'final_x', 'modelo': 'vmamba_tiny', 'estagio': 2, 'norm': 'sem_norm', 'resolucao': 672,
        'grade': 32, 'pesos_sha256': 'f' * 64}
DESCRICAO = {'modelo': 'vmamba_tiny', 'estagio': 2, 'norm': 'sem_norm', 'resolucao': 672, 'grade': 32,
             'lado_features': 42, 'canais': 384, 'passo_eft': 4000, 'pesos_sha256': 'f' * 64}


def recusa(funcao, *args):
    """Mensagem da trava; falha se a função aceitar o que devia recusar."""
    try:
        funcao(*args)
    except protocolo.Trava as erro:
        return str(erro)
    raise AssertionError(f'{funcao.__name__} aceitou o que devia recusar: {args}')


def test_teste_exige_item_final_commitado():
    itens = {'final_x': ITEM}
    assert protocolo.item_final('final_x', LIMPO, itens) is ITEM
    assert '--final' in recusa(protocolo.item_final, None, LIMPO, itens)
    assert 'não está em' in recusa(protocolo.item_final, 'outro', LIMPO, itens)
    assert 'sem commit' in recusa(protocolo.item_final, 'final_x', SUJO, itens)
    assert 'sem commit' in recusa(protocolo.item_final, 'final_x', {'commit': None, 'codigo_alterado': False}, itens)


def test_configuracao_avaliada_tem_de_ser_a_escrita():
    protocolo.conferir_final(ITEM, DESCRICAO)
    for campo, outro in (('modelo', 'vmamba_small'), ('estagio', 3), ('norm', 'checkpoint'),
                         ('resolucao', 512), ('grade', 42), ('pesos_sha256', 'e' * 64)):
        assert campo in recusa(protocolo.conferir_final, ITEM, dict(DESCRICAO, **{campo: outro}))
    # Campo omitido no item vale None: sem o hash, nenhum checkpoint fica autorizado.
    sem_hash = {k: v for k, v in ITEM.items() if k != 'pesos_sha256'}
    assert 'pesos_sha256' in recusa(protocolo.conferir_final, sem_hash, DESCRICAO)
    # DINOv2: estágio, norma e pesos nulos dos dois lados.
    dino = {'nome': 'dino', 'modelo': 'dinov2_l14', 'resolucao': 672, 'grade': 48}
    protocolo.conferir_final(dino, {'modelo': 'dinov2_l14', 'resolucao': 672, 'grade': 48,
                                    'estagio': None, 'norm': None, 'pesos_sha256': None})


def test_finais_yaml_do_repositorio_e_valido():
    for nome, item in protocolo.finais().items():
        assert item['nome'] == nome and item.get('modelo'), item
        assert set(item) <= {'nome', *protocolo.CAMPOS_FINAL}, f'campo desconhecido em {nome}: {sorted(item)}'


def test_estado_da_linha_do_registro():
    assert estado_do_codigo(LIMPO, LIMPO) == 'novo'
    assert estado_do_codigo(SUJO, LIMPO) == 'rascunho'
    assert estado_do_codigo(LIMPO, SUJO) == 'rascunho'
    assert estado_do_codigo(LIMPO, dict(LIMPO, commit='def456')) == 'rascunho'  # commit trocado no meio
    sem_git = {'commit': None, 'codigo_alterado': True}
    assert estado_do_codigo(sem_git, sem_git) == 'rascunho'


def test_git_responde_neste_repositorio():
    git = git_commit()
    assert set(git) == {'commit', 'codigo_alterado'} and git['commit'] and len(git['commit']) == 40, git


def test_nome_do_treino():
    r = '/home/u/cassio_pereira/ProtoSAM-repro/reproducao'
    assert nome_do_treino(f'{r}/eft_tiny_2000_v2_x/checkpoint_2000.pt') == 'eft_tiny_2000_v2_x'
    assert nome_do_treino(f'{r}/eft_SAM_x/treino_SAM/checkpoint_4000.pt') == 'eft_SAM_x/treino_SAM'
    assert nome_do_treino('/home/u/ProtoSAM-vmamba/runs/20261111_eft_a/checkpoint_0500.pt') == '20261111_eft_a'
    assert nome_do_treino('/dados/avulso/checkpoint.pt') == 'avulso'


def test_entrada_de_referencia_passa_e_imagem_invertida_nao():
    ref = protocolo.referencias()['entrada_k1']
    certo = {c: ref[c] for c in ('media', 'desvio', 'minimo', 'maximo', 'fracao_zeros', 'media_por_canal')}
    assert protocolo.diferencas_da_entrada(certo) == []
    # Uma versão diferente do redimensionamento desloca a média em milésimos: passa.
    assert protocolo.diferencas_da_entrada(dict(certo, media=ref['media'] + 0.001)) == []
    # O bug de mar/2026: cada pixel v virou 256 - v antes da normalização. Fora do padding de zeros,
    # um canal normalizado x vira (1 - 2 * media_imagenet) / desvio_imagenet - x.
    imagenet = ((0.485, 0.229), (0.456, 0.224), (0.406, 0.225))
    util = 1 - ref['fracao_zeros']
    invertida = [util * ((1 - 2 * m) / d - canal / util)
                 for canal, (m, d) in zip(ref['media_por_canal'], imagenet)]
    errado = dict(certo, media_por_canal=invertida, media=sum(invertida) / 3)
    assert len(protocolo.diferencas_da_entrada(errado)) == 4, protocolo.diferencas_da_entrada(errado)


def test_referencias_sao_coerentes_com_o_protocolo():
    ref = protocolo.referencias()
    assert ref['entrada_k1']['imagem'] == protocolo.carregar()['suporte_k1']
    avaliacoes = ref['avaliacoes']
    assert avaliacoes['imagens'] == protocolo.nomes(avaliacoes['conjunto'])[:len(avaliacoes['imagens'])]
    for caso in avaliacoes['casos']:
        assert len(caso['alp']) == len(caso['sam']) == len(avaliacoes['imagens']), caso['nome']
        assert bool(caso['checkpoint']) == bool(caso['checkpoint_sha256']), caso['nome']


if __name__ == '__main__':
    for nome, funcao in list(globals().items()):
        if nome.startswith('test_'):
            funcao()
            print('ok  ', nome)
    print('OK: travas do protocolo.')
