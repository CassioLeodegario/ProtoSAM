"""Pool e episódios EFT-SP. Filtros finais do notebook 04, sem máscaras manuais."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import importlib.metadata
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import felzenszwalb

MANIFESTO = Path('/workspace/ProtoSAM-repro/reproducao/protocolo_polipos_20261011T143313814944Z/manifesto.json')
CONFIG = {
    'versao': 1, 'seed_split': 1234, 'seed_episodios': 1234,
    'n_validacao': 145, 'n_pool': 1449, 'resolucao': [256, 256],
    'superpixels': {'scale': 100, 'sigma': 1.0, 'min_size': 400},
    'area_util': 'max RGB > 10; maior componente; preencher buracos; erosao 3x3, 3 iteracoes',
    'min_area_interna': 100, 'min_fracao_interna': 0.95,
    'grade': [32, 32], 'kernel': 4, 'limiar_bg': 0.95,
    'elegibilidade': 'FG sobrevive e pelo menos uma celula BG > 0.95 em support e query',
    'query': 'flip horizontal da mesma imagem e regiao',
    'mascaras_manuais_utilizadas': False,
    'sem_candidato': 'registrar exclusao, sem substituir ou mover imagens entre splits',
    'plano_treino': '200 imagens distintas por permutacao do treino elegivel; uma regiao uniforme por imagem',
    'plano_validacao': 'uma regiao fixa por imagem elegivel de validacao',
}

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()

def assinatura(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def gravar_json(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)

def agora():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def obter_area_util(rgb):
    nao_preto = rgb.max(axis=-1) > 10
    componentes, quantidade = ndi.label(nao_preto)
    if quantidade == 0:
        return np.zeros(nao_preto.shape, dtype=bool)
    tamanhos = np.bincount(componentes.ravel())
    tamanhos[0] = 0
    maior = componentes == tamanhos.argmax()
    maior = ndi.binary_fill_holes(maior)
    return ndi.binary_erosion(
        maior, structure=np.ones((3, 3), dtype=bool),
        iterations=3, border_value=0,
    )

def verificar_grade_eft(mascara):
    tensor = torch.from_numpy(np.ascontiguousarray(mascara)).float()[None, None]
    grade = F.interpolate(tensor, size=(32, 32), mode='nearest')
    fg_local = F.avg_pool2d(grade, kernel_size=4, stride=4)
    bg_local = F.avg_pool2d(1 - grade, kernel_size=4, stride=4)
    return {
        'pixels_fg': int(grade.sum().item()),
        'celulas_fg': int((fg_local > 0.95).sum().item()),
        'celulas_bg': int((bg_local > 0.95).sum().item()),
    }

def candidatos_eft(rgb):
    util = obter_area_util(rgb).astype(bool)
    segmentos = felzenszwalb(
        rgb.astype(np.float32) / 255.0,
        channel_axis=-1, **CONFIG['superpixels'],
    )
    candidatos = []
    for regiao in np.unique(segmentos):
        original = segmentos == regiao
        mascara = original & util
        if mascara.sum() < 100:
            continue
        if mascara.sum() / original.sum() < 0.95:
            continue
        query = np.fliplr(mascara).copy()
        gs, gq = verificar_grade_eft(mascara), verificar_grade_eft(query)
        if all(g['pixels_fg'] > 0 and g['celulas_bg'] > 0 for g in (gs, gq)):
            candidatos.append(int(regiao))
    return segmentos.astype(np.int32), util, np.asarray(candidatos, dtype=np.int32)

def dividir_pool(pool):
    pool = sorted(pool, key=lambda r: (r['nome'], r['imagem']))
    assert len(pool) == 1449
    val = set(np.random.default_rng(1234).permutation(len(pool))[:145].tolist())
    return [{'indice': i, 'nome': r['nome'], 'imagem': r['imagem'],
             'split': 'validacao' if i in val else 'treino'} for i, r in enumerate(pool)]

def preparar_dados(destino=None, manifesto=MANIFESTO):
    # Reduz custo de paralelismo em milhares de operações CPU sobre grades pequenas.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _preparar_dados(destino, manifesto)
    finally:
        torch.set_num_threads(threads)

def _preparar_dados(destino=None, manifesto=MANIFESTO):
    """Reutilize o mesmo destino para continuar preparação interrompida."""
    manifesto = Path(manifesto)
    dados = json.loads(manifesto.read_text())
    pool = sorted(dados['pool_adaptacao'], key=lambda r: (r['nome'], r['imagem']))
    assert len(pool) == 1449, f'Pool inesperado: {len(pool)}'
    caminhos = [str(Path(r['imagem']).resolve()) for r in pool]
    assert len(set(caminhos)) == len(pool), 'Caminhos repetidos no pool.'
    assert str(Path(dados['suporte_k1']['imagem']).resolve()) not in caminhos
    teste = {str(Path(r['imagem']).resolve()) for r in dados['teste']}
    assert not (set(caminhos) & teste), 'Caminho de teste presente no pool.'
    # Colisões de nomes entre datasets não são usadas para detectar vazamento.
    # Auditoria de hashes RGB foi feita anteriormente; paciente continua desconhecido.
    config = dict(CONFIG, manifesto=str(manifesto), manifesto_sha256=sha256(manifesto),
                  codigo_dados_sha256=sha256(__file__),
                  versoes={p: importlib.metadata.version(p) for p in ['numpy', 'Pillow', 'scipy', 'scikit-image', 'torch']})
    divisao = dividir_pool(pool)
    config['divisao_sha256'] = assinatura(divisao)
    if destino is None:
        destino = manifesto.parent.parent / ('eft_sp_piloto_dados_' + agora())
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    print('DADOS:', destino, flush=True)
    if (destino / 'config.json').exists():
        assert json.loads((destino / 'config.json').read_text()) == config, 'Configuração de dados mudou.'
        assert json.loads((destino / 'divisao.json').read_text()) == divisao, 'Divisão mudou.'
    else:
        gravar_json(destino / 'config.json', config)
        gravar_json(destino / 'divisao.json', divisao)
    (destino / 'cache').mkdir(exist_ok=True)
    registros = []
    for i, item in enumerate(divisao):
        cache = destino / 'cache' / f'{i:04d}.npz'
        meta = cache.with_suffix('.json')
        digest = sha256(item['imagem'])
        if meta.exists() and cache.exists():
            registro = json.loads(meta.read_text())
            assert registro['imagem_sha256'] == digest, f'Imagem alterada: {item["imagem"]}'
            assert registro['cache_sha256'] == sha256(cache), f'Cache alterado: {cache}'
        else:
            with Image.open(item['imagem']) as im:
                rgb = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
            segmentos, util, candidatos = candidatos_eft(rgb)
            tmp = cache.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, rgb=rgb, segmentos=segmentos, util=util, candidatos=candidatos)
            os.replace(tmp, cache)
            registro = dict(item, cache=str(cache.relative_to(destino)),
                            candidatos=candidatos.tolist(), imagem_sha256=digest,
                            cache_sha256=sha256(cache))
            gravar_json(meta, registro)
        registros.append(registro)
        if (i + 1) % 25 == 0 or i + 1 == len(divisao):
            print(f'Preparação: {i+1}/{len(divisao)} imagens', flush=True)
    treino = [r for r in registros if r['split'] == 'treino' and r['candidatos']]
    validacao = [r for r in registros if r['split'] == 'validacao' and r['candidatos']]
    excluidos = [r for r in registros if not r['candidatos']]
    gravar_json(destino / 'excluidos_sem_regiao.json', excluidos)
    assert len(treino) >= 200, 'Menos de 200 imagens elegíveis para o piloto.'
    assert validacao, 'Validação sem episódios elegíveis.'
    rng = np.random.default_rng(1234)
    plano_treino = []
    for idx in rng.permutation(len(treino))[:200]:
        r = treino[int(idx)]
        plano_treino.append(dict(r, regiao=int(rng.choice(r['candidatos']))))
    rng_val = np.random.default_rng(1234)
    plano_val = [dict(r, regiao=int(rng_val.choice(r['candidatos']))) for r in validacao]
    pacote = {'config': config, 'treino': plano_treino, 'validacao': plano_val,
              'resumo': {'treino_reservado': 1304, 'validacao_reservada': 145,
                         'treino_elegivel': len(treino), 'validacao_elegivel': len(validacao),
                         'excluidos': len(excluidos), 'episodios_piloto': 200}}
    gravar_json(destino / 'planos.json', pacote)
    print(json.dumps(pacote['resumo'], ensure_ascii=False, indent=2))
    print('Preparação concluída. Nenhum treinamento iniciado.')
    return destino

def carregar_episodio(pasta, registro, device):
    with np.load(Path(pasta) / registro['cache'], allow_pickle=False) as d:
        rgb = d['rgb'].copy()
        mascara = (d['segmentos'] == registro['regiao']) & d['util']
    assert mascara.any(), 'Máscara vazia no cache.'
    def imagem(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
        return (t - t.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]) / t.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    support, query = imagem(rgb), imagem(np.fliplr(rgb))
    fg = torch.from_numpy(mascara.copy()).to(device=device, dtype=torch.float32)[None]
    alvo = torch.from_numpy(np.fliplr(mascara).copy()).to(device=device, dtype=torch.long)[None]
    return support, fg, query, alvo
