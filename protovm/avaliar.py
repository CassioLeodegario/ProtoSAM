"""Avaliação ALP + SAM-H no protocolo de pólipos.

Substitui validar_polipos.py, validar_camadas.py, validar_marcos*.py,
validar_penultima*.py, validar_multissuporte_v1.py e os avaliadores dos
notebooks 01-03 e 16. Uma chamada = um modelo, um conjunto, um suporte.

Pré-processamento, SAM e métricas são os do protocolo de out/2026:
dataset oficial PolypDataset, canvas 1024, SAM-H com pontos + caixa,
isval=True e val_wsize=2 no ALP, Dice médio por imagem no canvas.
"""
from pathlib import Path
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from protovm import RAIZ, protocolo, registro
from protovm.util import (agora, ambiente, estado_do_codigo, fixar_numerica, git_commit, gravar_json,
                          ler_json, semear)

CFG_SAM = {'image_size': [1024, 1024], 'num_points_for_sam': 1, 'use_points': True, 'use_bbox': True,
           'use_mask': False, 'debug': False, 'use_cca': False, 'point_mode': 'both',
           'use_sam_trans': True, 'coarse_pred_only': False, 'use_neg_points': False}


def medir(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {'Dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'IoU': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'precisao': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0, 'TP': tp, 'FP': fp, 'FN': fn}


def exigir_kernel_cuda():
    """Garante que o VMamba usa o kernel selective_scan compilado, nunca o fallback lento em PyTorch."""
    from models import vmamba
    assert vmamba.WITH_SELECTIVESCAN_MAMBA, 'selective_scan_cuda não compilado (ver LEIA-ME, seção Ambiente).'

    def recusar(*args, **kwargs):
        raise RuntimeError('Fallback selective_scan_torch não permitido.')
    vmamba.selective_scan_torch = recusar


def estatisticas_da_entrada(imagem):
    """Resumo de uma imagem (3, H, W) como ela chega ao encoder; ver protocolo.diferencas_da_entrada."""
    x = imagem.float()
    return {'media': float(x.mean()), 'desvio': float(x.std()), 'minimo': float(x.min()),
            'maximo': float(x.max()), 'fracao_zeros': float((x == 0).float().mean()),
            'media_por_canal': [float(v) for v in x.mean(dim=(1, 2))]}


class Datasets:
    """Datasets oficiais carregados uma vez; indexação por nome de arquivo.

    Ao carregar, confere a imagem do suporte K1 contra as estatísticas registradas: se o
    pré-processamento mudou, nenhuma avaliação começa.
    """

    def __init__(self):
        from dataloaders.PolypDataset import get_polyp_dataset
        from models.segment_anything.utils.transforms import ResizeLongestSide
        os.chdir(RAIZ)  # o dataset oficial usa caminhos relativos a ./data
        self.treino, self.teste = get_polyp_dataset(image_size=(1024, 1024), sam_trans=ResizeLongestSide(1024))
        self.indice_treino = {Path(p).name: i for i, p in enumerate(self.treino.images)}
        self.indice_teste = {Path(p).name: i for i, p in enumerate(self.teste.images)}
        k1 = self.amostra(protocolo.carregar()['suporte_k1'], 'V')
        divergencias = protocolo.diferencas_da_entrada(estatisticas_da_entrada(k1['image']))
        if divergencias:
            raise protocolo.Trava('A imagem do suporte K1 não chega ao encoder como no protocolo '
                                  '(o pré-processamento mudou?):\n  ' + '\n  '.join(divergencias))

    def amostra(self, nome, conjunto):
        if conjunto == 'T':
            a = self.teste[self.indice_teste[nome]]
        else:
            a = self.treino[self.indice_treino[nome]]
        assert tuple(a['image'].shape) == (3, 1024, 1024) and tuple(a['label'].shape) == (1024, 1024)
        return a


def construir_protosam(alp, cca=False):
    """SAM-H com os prompts do protocolo.

    cca=True liga a análise de componentes conexos do código oficial: só o componente mais confiante
    da máscara do ALP gera prompts e o SAM devolve uma máscara única. Não muda a máscara do ALP
    medida em avaliar(), que é sempre a bruta.
    """
    from models.ProtoSAM import ProtoSAM, ALPNetWrapper
    cfg = dict(CFG_SAM, use_cca=bool(cca))
    wrapper = ALPNetWrapper(alp)
    protosam = ProtoSAM(coarse_segmentation_model=wrapper, alpnet_image_size=alp.descricao['resolucao'],
                        sam_pretrained_path=str(RAIZ / 'pretrained_model' / 'sam_vit_h.pth'), **cfg)
    protosam = protosam.to('cuda', dtype=torch.float32).eval().requires_grad_(False)
    protosam.cfg_sam = cfg  # o que avaliar() grava no config.json e no registro
    return wrapper, protosam


def avaliar(alp, conjunto='V', suporte='S00', rotulo=None, metadados=None, datasets=None,
            wrapper_sam=None, salvar_mascaras=True, pasta=None, final=None, limite=None):
    """Avalia um modelo já construído. Retorna a pasta do run e grava uma linha no registro.

    rotulo: nome curto do run (vai para o nome da pasta).
    metadados: campos extras para o registro (ex.: pesos, passo, treino_resolucao).
    final: item de protocolo/finais.yaml; obrigatório nos conjuntos travados (T).
    limite: avalia só as N primeiras imagens (ensaio ou teste). Não entra no registro.

    A linha entra como 'novo' se o código estava commitado; com alteração sem commit entra
    como 'rascunho' e não vai para tabela.
    """
    from models.ProtoSAM import ALPNetInput
    git_inicio = git_commit()
    item = protocolo.item_final(final, git_inicio) if conjunto in protocolo.TRAVADOS else None
    fixar_numerica()
    semear(42)
    if alp.descricao['modelo'].startswith('vmamba'):
        exigir_kernel_cuda()
    datasets = datasets or Datasets()
    wrapper, protosam = wrapper_sam or construir_protosam(alp)
    cfg_sam = getattr(protosam, 'cfg_sam', CFG_SAM)
    if item is not None:
        protocolo.conferir_final(item, dict(alp.descricao, cca=cfg_sam['use_cca']))
    metadados = dict(metadados or {})

    nome_suporte = protocolo.suporte(suporte)
    queries = protocolo.nomes(conjunto)
    assert nome_suporte not in queries
    if limite:
        queries = queries[:limite]
    sup = datasets.amostra(nome_suporte, 'V')
    if nome_suporte == protocolo.carregar()['suporte_k1']:
        assert int((sup['label'] > .5).sum()) == protocolo.carregar()['suporte_k1_pixels_fg_canvas1024']
    sup_img = sup['image'].unsqueeze(0).cuda().float()
    sup_mask = sup['label'].unsqueeze(0).cuda().float()

    config = {'conjunto': conjunto, 'suporte': suporte, 'suporte_nome': nome_suporte, 'n': len(queries),
              'alp': alp.descricao, 'sam': dict(cfg_sam, modelo='sam_vit_h'),
              'alp_inferencia': {'isval': True, 'val_wsize': 2}, 'metadados': metadados,
              'final': final, 'limite': limite}
    if pasta is None:
        sufixo = f'_n{limite}' if limite else ''
        pasta = RAIZ / 'runs' / f"{agora()}_aval_{rotulo or alp.descricao['modelo']}_{conjunto}_{suporte}{sufixo}"
        pasta.mkdir(parents=True, exist_ok=False)
        gravar_json(pasta / 'config.json', dict(config, git=git_inicio, ambiente=ambiente()))
    pasta = Path(pasta)
    assert {k: v for k, v in ler_json(pasta / 'config.json').items() if k in config} == config, 'Config mudou.'
    (pasta / 'por_imagem').mkdir(exist_ok=True)
    if salvar_mascaras:
        for sub in ('alp', 'sam'):
            (pasta / sub).mkdir(exist_ok=True)
    print('AVALIAÇÃO:', pasta, flush=True)

    linhas = []
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for pos, nome in enumerate(queries):
            arquivo = pasta / 'por_imagem' / f'{Path(nome).stem}.json'
            if arquivo.exists():
                linhas.append(ler_json(arquivo))
                continue
            a = datasets.amostra(nome, conjunto)
            gt = a['label'].cpu().numpy() > .5
            qry = a['image'].unsqueeze(0).cuda().float()

            def entrada():
                return ALPNetInput(support_images=[sup_img], support_labels=[sup_mask], query_images=qry,
                                   isval=True, val_wsize=2, show_viz=False, supp_fts=None)
            logits = wrapper(entrada())
            assert tuple(logits.shape) == (1, 2, 1024, 1024) and torch.isfinite(logits).all()
            alp_mask = (logits.argmax(1)[0] == 1).cpu().numpy()
            saida, scores = protosam(query_image=qry, coarse_model_input=entrada(), degrees_rotate=0)
            sam_mask = saida.detach().cpu().numpy().squeeze().astype(bool)
            linha = {'query': nome, 'ALP': medir(alp_mask, gt), 'SAM': medir(sam_mask, gt), 'scores_sam': len(scores)}
            if salvar_mascaras:
                for sub, m in (('alp', alp_mask), ('sam', sam_mask)):
                    Image.fromarray(m.astype(np.uint8) * 255).save(pasta / sub / f'{Path(nome).stem}.png')
            gravar_json(arquivo, linha)
            linhas.append(linha)
            print(f"{pos + 1}/{len(queries)} {nome} | ALP={linha['ALP']['Dice']:.4f} SAM={linha['SAM']['Dice']:.4f}", flush=True)

    with (pasta / 'por_imagem.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['query'] + [f'{e}_{m}' for e in ('ALP', 'SAM') for m in ('Dice', 'IoU', 'precisao', 'recall')])
        for l in linhas:
            w.writerow([l['query']] + [l[e][m] for e in ('ALP', 'SAM') for m in ('Dice', 'IoU', 'precisao', 'recall')])
    resumo = {e: {m: float(np.mean([l[e][m] for l in linhas])) for m in ('Dice', 'IoU', 'precisao', 'recall')}
              for e in ('ALP', 'SAM')}
    resumo['n'] = len(linhas)
    gravar_json(pasta / 'resumo.json', resumo)
    print(json.dumps(resumo, indent=2), flush=True)

    if limite:
        print(f'Ensaio com {len(linhas)} imagens: fora do registro.', flush=True)
        return pasta
    estado = estado_do_codigo(git_inicio, git_commit())
    d = alp.descricao
    registro.adicionar({
        'origem': str(pasta.relative_to(RAIZ).as_posix()), 'estado': estado,
        'modelo': d['modelo'], 'aval_estagio': d.get('estagio'), 'aval_norm': d.get('norm'),
        'aval_resolucao': d['resolucao'], 'grade': d['grade'], 'cca': int(cfg_sam['use_cca']),
        'conjunto': conjunto, 'suporte': suporte,
        'n': len(linhas), 'dice_alp': resumo['ALP']['Dice'], 'dice_sam': resumo['SAM']['Dice'],
        'gpu': ambiente()['gpu'], 'commit': (git_inicio['commit'] or '')[:10], **metadados,
    })
    if estado == 'rascunho':
        print('Código com alteração sem commit: a linha entrou no registro como rascunho.', flush=True)
    return pasta
