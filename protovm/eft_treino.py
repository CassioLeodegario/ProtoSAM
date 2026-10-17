"""Treino EFT-SP: um único laço para qualquer variante, estágio, normalização e resolução.

Substitui eft_treino.py (piloto 200), eft_continuar.py (até 2000), eft_5000.py
e a versão da penúltima camada. Os primeiros N passos de um treino longo são
idênticos a um treino de N passos (mesmo plano e mesma sequência de RNG).

Saída em runs/<data>_<nome>/:
    config.json            config do YAML + commit + ambiente
    passos.jsonl           uma linha por atualização
    validacao_NNNN.json    CE e Dice em pseudo-máscaras (não é métrica de pólipo)
    checkpoint_NNNN.pt     marcos pedidos em `checkpoints`
    checkpoint_latest.pt   para retomar; reexecutar o mesmo comando continua daqui
"""
from pathlib import Path
from types import MethodType
import json
import os
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from protovm import RAIZ, eft_dados
from protovm.encoder import construir_alp
from protovm.util import agora, ambiente, fixar_numerica, git_commit, gravar_json, ler_json


def estado_rng():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restaurar_rng(r):
    random.setstate(r['python'])
    n = r['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(r['torch'])
    torch.cuda.set_rng_state_all(r['cuda'])


class _SemPrototipos(Exception):
    pass


def instalar_alinhamento_seguro(alp):
    """Episódio sem protótipos no alinhamento: alinhamento = 0, CE continua.
    Antes era o arquivo eft_alinhamento_seguro.py, aplicado a partir do passo 201."""
    align_original = alp.alignLoss
    pred_original = alp.cls_unit.get_prediction_from_prototypes
    estado = {'ativo': False, 'ultimo': None}

    def prediction(self, prototypes, query, mode, *args, **kwargs):
        if estado['ativo'] and prototypes.shape[0] == 0:
            raise _SemPrototipos(str(mode))
        return pred_original(prototypes, query, mode, *args, **kwargs)

    def align(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        estado['ativo'] = True
        try:
            return align_original(qry_fts, pred, supp_fts, fore_mask, back_mask)
        except _SemPrototipos as erro:
            estado['ultimo'] = {'motivo': 'zero prototipos no alinhamento', 'modo': str(erro)}
            return qry_fts.sum() * 0.0
        finally:
            estado['ativo'] = False

    alp.alignLoss = MethodType(align, alp)
    alp.cls_unit.get_prediction_from_prototypes = MethodType(prediction, alp.cls_unit)
    return estado


def validar_pseudo(alp, dados, plano, arquivo, resolucao):
    """CE e Dice do primeiro plano nas pseudo-máscaras de validação (eval, isval=False)."""
    modo = alp.training
    alp.eval()
    linhas = []
    try:
        with torch.no_grad():
            for registro in plano:
                s, fg, q, alvo = eft_dados.carregar_episodio(dados, registro)
                logits = alp([[s]], [[fg]], [[1 - fg]], [q], isval=False, val_wsize=2, show_viz=False)[0]
                assert logits.shape == (1, 2, resolucao, resolucao) and torch.isfinite(logits).all()
                pred, real = logits.argmax(1) == 1, alvo == 1
                denom = pred.sum().item() + real.sum().item()
                linhas.append({'indice': registro['indice'], 'regiao': registro['regiao'],
                               'ce': F.cross_entropy(logits, alvo).item(),
                               'dice_pseudo_fg': 2.0 * (pred & real).sum().item() / denom})
        resumo = {'n': len(linhas), 'ce_media': float(np.mean([r['ce'] for r in linhas])),
                  'dice_pseudo_fg_medio': float(np.mean([r['dice_pseudo_fg'] for r in linhas])),
                  'por_imagem': linhas}
        gravar_json(arquivo, resumo)
        return resumo
    finally:
        alp.train(modo)


def treinar(cfg, pasta=None):
    """cfg: dicionário já mesclado (ver protovm.config). pasta: retoma uma execução."""
    fixar_numerica()
    dados = (RAIZ / cfg['dados']).resolve()
    dados_cfg = ler_json(dados / 'config.json')
    assert dados_cfg['resolucao'] == [cfg['resolucao']] * 2, 'Cache em outra resolução.'
    plano = eft_dados.planos(dados, cfg['passos'])
    eft_dados.conferir_cache(dados, plano['treino'] + plano['validacao'])

    registro_cfg = dict(cfg, plano_assinatura=plano['assinatura'], dados_config=dados_cfg)
    if pasta is None:
        pasta = RAIZ / 'runs' / f"{agora()}_{cfg['nome']}"
        pasta.mkdir(parents=True, exist_ok=False)
        gravar_json(pasta / 'config.json', dict(registro_cfg, git=git_commit(), ambiente=ambiente()))
    pasta = Path(pasta)
    salvo_cfg = ler_json(pasta / 'config.json')
    assert {k: salvo_cfg[k] for k in registro_cfg} == registro_cfg, 'Config mudou; não retomar.'
    print('EXECUÇÃO:', pasta, flush=True)
    if (pasta / 'resumo.json').exists():
        print('Já concluído.')
        return pasta

    alp = construir_alp(cfg['modelo'], cfg['estagio'], cfg['norm'], cfg['resolucao'], cfg['grade'],
                        treino=True, semente=cfg['semente'],
                        pesos=(RAIZ / cfg['inicializar_de']) if cfg.get('inicializar_de') else None)
    estado_align = instalar_alinhamento_seguro(alp)
    parametros = [p for p in alp.parameters() if p.requires_grad]
    otimizador = torch.optim.AdamW(parametros, lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=cfg['weight_decay'])

    latest = pasta / 'checkpoint_latest.pt'
    passo = 0
    if latest.exists():
        salvo = torch.load(latest, map_location='cpu', weights_only=True)
        alp.load_state_dict(salvo['modelo'], strict=True)
        otimizador.load_state_dict(salvo['otimizador'])
        passo = int(salvo['passo'])
        restaurar_rng(salvo['rng'])
        del salvo
        print(f'Retomando após o passo {passo}.', flush=True)

    def salvar():
        tmp = latest.with_suffix('.tmp')
        torch.save({'formato': 2, 'modelo': alp.state_dict(), 'otimizador': otimizador.state_dict(),
                    'passo': passo, 'rng': estado_rng(), 'config': registro_cfg}, tmp)
        os.replace(tmp, latest)

    def marco(n):
        if n in cfg['checkpoints'] and not (pasta / f'checkpoint_{n:04d}.pt').exists():
            shutil.copy2(latest, pasta / f'checkpoint_{n:04d}.pt')
        arquivo = pasta / f'validacao_{n:04d}.json'
        if n % cfg['validar_a_cada'] == 0 and not arquivo.exists():
            rng = estado_rng()
            try:
                v = validar_pseudo(alp, dados, plano['validacao'], arquivo, cfg['resolucao'])
            finally:
                restaurar_rng(rng)
            print(f"Passo {n}: CE val={v['ce_media']:.6f} | Dice pseudo-FG={v['dice_pseudo_fg_medio']:.6f}", flush=True)

    try:
        if passo == 0:
            salvar()
            marco(0)
        alp.train()
        log = (pasta / 'passos.jsonl').open('a', encoding='utf-8')
        for numero in range(passo + 1, cfg['passos'] + 1):
            r = plano['treino'][numero - 1]
            inicio = time.perf_counter()
            s, fg, q, alvo = eft_dados.carregar_episodio(dados, r)
            otimizador.zero_grad(set_to_none=True)
            estado_align['ultimo'] = None
            logits, alinhamento = alp([[s]], [[fg]], [[1 - fg]], [q], isval=False, val_wsize=2, show_viz=False)[:2]
            assert logits.shape == (1, 2, cfg['resolucao'], cfg['resolucao'])
            ce = F.cross_entropy(logits, alvo)
            total = ce + alinhamento
            assert torch.isfinite(total).item(), f'Perda não finita no passo {numero}.'
            total.backward()
            otimizador.step()
            linha = {'passo': numero, 'indice': r['indice'], 'regiao': r['regiao'],
                     'ce': ce.item(), 'alinhamento': float(alinhamento), 'total': total.item(),
                     'excecao_alinhamento': estado_align['ultimo'],
                     'segundos': round(time.perf_counter() - inicio, 4)}
            log.write(json.dumps(linha, ensure_ascii=False) + '\n')
            log.flush()
            passo = numero
            if numero % 10 == 0 or numero == 1:
                print(f"{numero:05d}/{cfg['passos']} | CE={linha['ce']:.5f} | align={linha['alinhamento']:.5f}", flush=True)
            if numero % cfg['salvar_a_cada'] == 0 or numero in cfg['checkpoints'] or numero == cfg['passos']:
                salvar()
            marco(numero)
        log.close()
        curva = []
        for f in sorted(pasta.glob('validacao_*.json')):
            v = ler_json(f)
            curva.append({'passo': int(f.stem.split('_')[1]), 'ce': v['ce_media'],
                          'dice_pseudo_fg': v['dice_pseudo_fg_medio']})
        gravar_json(pasta / 'resumo.json', {'passos': passo, 'curva': curva,
                                            'checkpoints': sorted(p.name for p in pasta.glob('checkpoint_0*.pt'))})
        print('Concluído:', pasta, flush=True)
        return pasta
    except BaseException:
        (pasta / f'erro_{agora()}.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Interrompido. Rode o mesmo comando com --retomar', pasta, flush=True)
        raise
