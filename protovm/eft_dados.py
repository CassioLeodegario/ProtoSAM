"""Episódios EFT-SP: superpixels de Felzenszwalb, support = imagem, query = flip horizontal.

Substitui as seis cópias idênticas de eft_dados.py. Com resolucao=256 e
superpixels_em=256 reproduz o cache e os planos de out/2026.

Parâmetros novos:
    resolucao       lado das imagens de treino guardadas no cache
    superpixels_em  lado em que o Felzenszwalb roda; se menor que resolucao,
                    segmentos e área útil são ampliados por vizinho mais próximo.
                    Mantém as mesmas regiões e o mesmo plano em qualquer resolução.
"""
from pathlib import Path
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import felzenszwalb

from protovm import protocolo
from protovm.util import assinatura, gravar_json, ler_json, sha256

FILTROS = {
    'superpixels': {'scale': 100, 'sigma': 1.0, 'min_size': 400},
    'area_util': 'max RGB > 10; maior componente; preencher buracos; erosao 3x3, 3 iteracoes',
    'min_area_interna': 100, 'min_fracao_interna': 0.95,
    'grade': [32, 32], 'kernel': 4, 'limiar_bg': 0.95,
    'elegibilidade': 'FG sobrevive e pelo menos uma celula BG > 0.95 em support e query',
    'query': 'flip horizontal da mesma imagem e regiao',
}
MEDIA = [123.675, 116.28, 103.53]
DESVIO = [58.395, 57.12, 57.375]


def obter_area_util(rgb):
    nao_preto = rgb.max(axis=-1) > 10
    componentes, quantidade = ndi.label(nao_preto)
    if quantidade == 0:
        return np.zeros(nao_preto.shape, dtype=bool)
    tamanhos = np.bincount(componentes.ravel())
    tamanhos[0] = 0
    maior = ndi.binary_fill_holes(componentes == tamanhos.argmax())
    return ndi.binary_erosion(maior, structure=np.ones((3, 3), dtype=bool), iterations=3, border_value=0)


def verificar_grade(mascara):
    tensor = torch.from_numpy(np.ascontiguousarray(mascara)).float()[None, None]
    grade = F.interpolate(tensor, size=(32, 32), mode='nearest')
    fg_local = F.avg_pool2d(grade, kernel_size=4, stride=4)
    bg_local = F.avg_pool2d(1 - grade, kernel_size=4, stride=4)
    return {'pixels_fg': int(grade.sum().item()),
            'celulas_fg': int((fg_local > 0.95).sum().item()),
            'celulas_bg': int((bg_local > 0.95).sum().item())}


def candidatos(rgb):
    util = obter_area_util(rgb).astype(bool)
    segmentos = felzenszwalb(rgb.astype(np.float32) / 255.0, channel_axis=-1, **FILTROS['superpixels'])
    escolhidos = []
    for regiao in np.unique(segmentos):
        original = segmentos == regiao
        mascara = original & util
        if mascara.sum() < 100 or mascara.sum() / original.sum() < 0.95:
            continue
        query = np.fliplr(mascara).copy()
        gs, gq = verificar_grade(mascara), verificar_grade(query)
        if all(g['pixels_fg'] > 0 and g['celulas_bg'] > 0 for g in (gs, gq)):
            escolhidos.append(int(regiao))
    return segmentos.astype(np.int32), util, np.asarray(escolhidos, dtype=np.int32)


def _ler_rgb(caminho, lado):
    with Image.open(caminho) as im:
        return np.asarray(im.convert('RGB').resize((lado, lado), Image.Resampling.BILINEAR))


def _ampliar(mapa, lado):
    fator = lado // mapa.shape[0]
    assert fator * mapa.shape[0] == lado, 'superpixels_em deve dividir resolucao'
    return np.repeat(np.repeat(mapa, fator, axis=0), fator, axis=1)


def _divisao():
    validacao = set(protocolo.nomes('V'))
    pool = protocolo.pool_eft()
    return [{'indice': i, 'nome': nome, 'split': 'validacao' if nome in validacao else 'treino'}
            for i, nome in enumerate(pool)]


def preparar(destino, resolucao=256, superpixels_em=None):
    """Cria (ou completa) o cache. Reexecutar no mesmo destino retoma de onde parou."""
    superpixels_em = superpixels_em or resolucao
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    config = dict(FILTROS, versao=2, resolucao=[resolucao, resolucao],
                  superpixels_em=[superpixels_em, superpixels_em],
                  protocolo_sha256=sha256(protocolo.ARQUIVO))
    divisao = _divisao()
    if (destino / 'config.json').exists():
        assert ler_json(destino / 'config.json') == config, 'Configuração de dados mudou neste destino.'
    else:
        gravar_json(destino / 'config.json', config)
        gravar_json(destino / 'divisao.json', divisao)
    (destino / 'cache').mkdir(exist_ok=True)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        registros = []
        for item in divisao:
            cache = destino / 'cache' / f"{item['indice']:04d}.npz"
            meta = cache.with_suffix('.json')
            if meta.exists() and cache.exists():
                registros.append(ler_json(meta))
                continue
            caminho = protocolo.imagem_treino(item['nome'])
            rgb_sp = _ler_rgb(caminho, superpixels_em)
            segmentos, util, cand = candidatos(rgb_sp)
            rgb = rgb_sp if superpixels_em == resolucao else _ler_rgb(caminho, resolucao)
            if superpixels_em != resolucao:
                segmentos, util = _ampliar(segmentos, resolucao), _ampliar(util, resolucao)
            tmp = cache.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, rgb=rgb, segmentos=segmentos, util=util, candidatos=cand)
            os.replace(tmp, cache)
            registro = dict(item, cache=str(cache.relative_to(destino).as_posix()),
                            candidatos=cand.tolist(), imagem_sha256=sha256(caminho),
                            cache_sha256=sha256(cache))
            gravar_json(meta, registro)
            registros.append(registro)
            if len(registros) % 100 == 0:
                print(f'Preparação: {len(registros)}/{len(divisao)}', flush=True)
    finally:
        torch.set_num_threads(threads)

    excluidos = [r for r in registros if not r['candidatos']]
    gravar_json(destino / 'excluidos_sem_regiao.json', excluidos)
    resumo = {'treino_elegivel': sum(1 for r in registros if r['split'] == 'treino' and r['candidatos']),
              'validacao_elegivel': sum(1 for r in registros if r['split'] == 'validacao' and r['candidatos']),
              'excluidos': len(excluidos)}
    print('Preparação concluída:', json.dumps(resumo), flush=True)
    return destino


def _registros(dados):
    divisao = ler_json(Path(dados) / 'divisao.json')
    return [ler_json(Path(dados) / 'cache' / f"{d['indice']:04d}.json") for d in divisao]


def planos(dados, passos):
    """Plano determinístico (semente 1234). Os primeiros N episódios não dependem
    de `passos`: o plano de 2000 é prefixo do de 5000, e o de 200 é o piloto."""
    registros = _registros(dados)
    treino = sorted((r for r in registros if r['split'] == 'treino' and r['candidatos']),
                    key=lambda r: r['indice'])
    validacao = [r for r in registros if r['split'] == 'validacao' and r['candidatos']]
    rng = np.random.default_rng(1234)
    plano = []
    while len(plano) < passos:
        for idx in rng.permutation(len(treino)):
            r = treino[int(idx)]
            plano.append(dict(r, regiao=int(rng.choice(r['candidatos']))))
            if len(plano) == passos:
                break
    rng_val = np.random.default_rng(1234)
    plano_val = [dict(r, regiao=int(rng_val.choice(r['candidatos']))) for r in validacao]
    assert not {r['indice'] for r in plano} & {r['indice'] for r in plano_val}
    return {'treino': plano, 'validacao': plano_val, 'assinatura': assinatura([plano, plano_val])}


def conferir_cache(dados, registros):
    for r in registros:
        assert sha256(Path(dados) / r['cache']) == r['cache_sha256'], f"Cache alterado: {r['cache']}"


def carregar_episodio(dados, registro, device='cuda'):
    with np.load(Path(dados) / registro['cache'], allow_pickle=False) as d:
        rgb = d['rgb'].copy()
        mascara = (d['segmentos'] == registro['regiao']) & d['util']
    assert mascara.any(), 'Máscara vazia no cache.'

    def imagem(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
        return (t - t.new_tensor(MEDIA)[None, :, None, None]) / t.new_tensor(DESVIO)[None, :, None, None]

    support, query = imagem(rgb), imagem(np.fliplr(rgb))
    fg = torch.from_numpy(mascara.copy()).to(device=device, dtype=torch.float32)[None]
    alvo = torch.from_numpy(np.fliplr(mascara).copy()).to(device=device, dtype=torch.long)[None]
    return support, fg, query, alvo
