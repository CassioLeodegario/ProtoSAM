"""Abdome (rins em RM CHAOS-T2 e em CT SABS): pseudo-máscaras, ajuste auto-supervisionado e avaliação.

Protocolo de DESENVOLVIMENTO (Exp. 12). Os pacientes do fold 0 de teste oficial e o suporte usado em
out/2026 ficam reservados e não são lidos aqui:

    RM  reservados 0-4   validação 5-9    ajuste 10-19
    CT  reservados 0-6   validação 7-14   ajuste 15-29

Ajuste: um episódio = uma fatia sem rótulo e uma região dela (superpixel de Felzenszwalb dos arquivos
superpix-MIDDLE, ou região do gerador automático do SAM). Suporte = fatia + região; query = a mesma
fatia espelhada ('flip') ou deformada ('afim': rotação, escala, translação, espelho e gama).
Por padrão as fatias com rim ficam fora do ajuste (como em out/2026).

Avaliação: cada paciente-suporte da validação fornece a fatia central de cada rim; as queries são
fatias com o rim dos outros pacientes da validação. Dice por fatia -> média por paciente -> média
entre pacientes. Uma linha por (modelo, suporte) em resultados/abdome.csv.

As fatias (256 x 256) são reamostradas para 672 antes do encoder, como em pólipos.
"""
from datetime import date
from functools import lru_cache
from pathlib import Path
import csv
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from protovm import RAIZ
from protovm.eft_treino import estado_rng, instalar_alinhamento_seguro, restaurar_rng
from protovm.encoder import construir_alp
from protovm.util import (agora, ambiente, estado_do_codigo, fixar_numerica, git_commit, gravar_json,
                          ler_json, semear)

LADO = 672
MEDIA = [123.675, 116.28, 103.53]
DESVIO = [58.395, 57.12, 57.375]
ENTRADAS = ('zscore', 'imagenet')
# entrada 'zscore'   -> (x - média) / desvio do volume, repetido em 3 canais (o que o código oficial faz)
#         'imagenet' -> volume recortado nos percentis 0,5 e 99,5, levado a 0-255 e normalizado como ImageNet
MODALIDADES = {
    'MR': {'pasta': 'CHAOST2/chaos_MR_T2_normalized', 'orgaos': {'rim_d': ('RK', 2), 'rim_e': ('LK', 3)},
           'entrada': 'zscore', 'ajuste': list(range(10, 20)), 'validacao': list(range(5, 10)),
           'suportes': list(range(5, 10)), 'passo_fatias': 1},
    'CT': {'pasta': 'SABS/sabs_CT_normalized', 'orgaos': {'rim_d': ('KID_R', 2), 'rim_e': ('KID_l', 3)},
           'entrada': 'imagenet', 'ajuste': list(range(15, 30)), 'validacao': list(range(7, 15)),
           'suportes': list(range(7, 12)), 'passo_fatias': 3},
}
AMG = {'points_per_side': 32, 'points_per_batch': 16, 'pred_iou_thresh': 0.88, 'stability_score_thresh': 0.95,
       'stability_score_offset': 1.0, 'box_nms_thresh': 0.7, 'crop_n_layers': 0, 'min_mask_region_area': 0,
       'output_mode': 'binary_mask'}  # os mesmos de pólipos (dados_SAM_EFT512, out/2026)
REGISTRO = RAIZ / 'resultados' / 'abdome.csv'
COLUNAS = ['id', 'data', 'estado', 'modalidade', 'modelo', 'pesos', 'passo', 'pseudo', 'aug', 'lr', 'excluir',
           'entrada', 'aval_estagio', 'aval_norm', 'cca', 'suporte', 'n',
           'dice_alp_rim_d', 'dice_sam_rim_d', 'dice_alp_rim_e', 'dice_sam_rim_e', 'dice_alp', 'dice_sam',
           'commit', 'origem', 'obs', 'mascara']
MASCARAS = ('indice0', 'maior_score')  # qual das 3 máscaras do SAM é usada: a primeira (oficial) ou a de maior score


# ---------------------------------------------------------------- dados

def pasta_dados(mod):
    sub = MODALIDADES[mod]['pasta']
    for base in (RAIZ / 'data', RAIZ.parent / 'ProtoSAM-repro' / 'data'):
        if (base / sub).is_dir():
            return base / sub
    raise FileNotFoundError(f'Dados de {mod} não encontrados ({sub}).')


def _ler(mod, prefixo, pid):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(pasta_dados(mod) / f'{prefixo}_{pid}.nii.gz')))


@lru_cache(maxsize=None)
def classmap(mod):
    return ler_json(pasta_dados(mod) / 'classmap_1.json')


def fatias_com(mod, orgao, pid):
    return [int(z) for z in classmap(mod)[MODALIDADES[mod]['orgaos'][orgao][0]][str(pid)]]


@lru_cache(maxsize=40)
def volume(mod, pid):
    """imagem crua (Z, 256, 256) float32, rótulos, superpixels e máscara do corpo."""
    img = _ler(mod, 'image', pid).astype(np.float32)
    assert img.shape[1:] == (256, 256) and np.isfinite(img).all(), img.shape
    v = {'img': img, 'rotulo': _ler(mod, 'label', pid), 'sp': _ler(mod, 'superpix-MIDDLE', pid),
         'corpo': _ler(mod, 'fgmask', pid) > 0}
    assert all(v[k].shape == img.shape for k in ('rotulo', 'sp', 'corpo'))
    lo, hi = np.percentile(img, [0.5, 99.5])
    assert hi > lo
    v.update(media=float(img.mean()), desvio=float(img.std()), lo=float(lo), hi=float(hi))
    return v


def _em_255(v, z):
    return np.clip((v['img'][z] - v['lo']) / (v['hi'] - v['lo']), 0, 1) * 255.0


def fatia_uint8(mod, pid, z):
    """Fatia como imagem RGB de 8 bits (entrada do gerador de máscaras do SAM)."""
    return np.repeat(_em_255(volume(mod, pid), z).round().astype(np.uint8)[..., None], 3, axis=2)


def tensor_fatia(mod, pid, z, entrada, lado=LADO):
    """(1, 3, lado, lado) na GPU, normalizado conforme `entrada`."""
    v = volume(mod, pid)
    if entrada == 'zscore':
        x = torch.from_numpy((v['img'][z] - v['media']) / v['desvio'])[None, None].repeat(1, 3, 1, 1).cuda()
    elif entrada == 'imagenet':
        x = torch.from_numpy(_em_255(v, z).astype(np.float32))[None, None].repeat(1, 3, 1, 1).cuda()
        x = (x - x.new_tensor(MEDIA)[None, :, None, None]) / x.new_tensor(DESVIO)[None, :, None, None]
    else:
        raise ValueError(entrada)
    return F.interpolate(x.float(), size=(lado, lado), mode='bilinear', align_corners=False)


def _mascara_672(m256, lado=LADO):
    t = torch.from_numpy(np.ascontiguousarray(m256).astype(np.float32))[None, None].cuda()
    return F.interpolate(t, size=(lado, lado), mode='nearest')[0]  # (1, lado, lado)


# ---------------------------------------------------------------- regiões (pseudo-máscaras)

def _regiao_valida(m, corpo):
    area, util = int(m.sum()), int(corpo.sum())
    return util > 0 and max(50, 0.005 * util) <= area <= 0.8 * util and int((m & corpo).sum()) >= 0.95 * area


def pasta_sam(mod):
    return RAIZ / 'dados_eft' / 'abdome_sam' / mod


def gerar_regioes_sam(mod):
    """Regiões do gerador automático do SAM-H para as fatias de ajuste. Retomável (um .npz por paciente)."""
    from models.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    cfg = MODALIDADES[mod]
    destino = pasta_sam(mod)
    destino.mkdir(parents=True, exist_ok=True)
    gravar_json(destino / 'config.json', {'modalidade': mod, 'AMG': AMG, 'passo_fatias': cfg['passo_fatias'],
                                          'pacientes': cfg['ajuste'], 'git': git_commit()})
    sam = sam_model_registry['vit_h'](checkpoint=str(RAIZ / 'pretrained_model' / 'sam_vit_h.pth')).cuda().eval()
    gerador = SamAutomaticMaskGenerator(sam, **AMG)
    for pid in cfg['ajuste']:
        arquivo = destino / f'p{pid}.npz'
        if arquivo.exists():
            continue
        v, saida, total = volume(mod, pid), {}, 0
        for z in range(0, v['img'].shape[0], cfg['passo_fatias']):
            if not v['corpo'][z].any():
                continue
            with torch.no_grad():
                brutas = gerador.generate(fatia_uint8(mod, pid, z))
            vistas, boas = set(), []
            for b in brutas:
                m = np.asarray(b['segmentation'], dtype=bool)
                chave = m.tobytes()
                if chave not in vistas and _regiao_valida(m, v['corpo'][z]):
                    vistas.add(chave)
                    boas.append(m)
            if boas:
                saida[f'z{z}'] = np.packbits(np.stack(boas), axis=-1)
                total += len(boas)
        tmp = destino / f'p{pid}.tmp.npz'
        np.savez_compressed(tmp, **saida)
        os.replace(tmp, arquivo)
        print(f'{mod} paciente {pid}: {len(saida)} fatias com região, {total} regiões', flush=True)
    print('REGIOES CONCLUIDAS', flush=True)


@lru_cache(maxsize=4)
def _cache_sam(mod, pid):
    with np.load(pasta_sam(mod) / f'p{pid}.npz') as d:
        return {int(k[1:]): np.unpackbits(d[k], axis=-1).astype(bool) for k in d.files}


def regioes_da_fatia(mod, tipo, pid, z):
    """Lista de identificadores de região válidos na fatia (id do superpixel ou índice no cache do SAM)."""
    v = volume(mod, pid)
    if tipo == 'SP':
        return [int(i) for i in np.unique(v['sp'][z]) if i != 0 and _regiao_valida(v['sp'][z] == i, v['corpo'][z])]
    if tipo == 'SAM':
        return list(range(len(_cache_sam(mod, pid).get(z, []))))
    raise ValueError(tipo)


def mascara_regiao(mod, tipo, ep):
    if tipo == 'SP':
        m = volume(mod, ep['paciente'])['sp'][ep['fatia']] == ep['regiao']
    else:
        m = _cache_sam(mod, ep['paciente'])[ep['fatia']][ep['regiao']]
    assert m.shape == (256, 256) and m.any()
    return m


def montar_plano(mod, tipo, passos, semente, excluir=True):
    """Sorteio dos episódios: fatia uniforme entre as que têm região, depois região uniforme na fatia."""
    cfg = MODALIDADES[mod]
    candidatas = []
    for pid in cfg['ajuste']:
        proibidas = set()
        if excluir:
            for orgao in cfg['orgaos']:
                proibidas |= set(fatias_com(mod, orgao, pid))
        for z in range(0, volume(mod, pid)['img'].shape[0], cfg['passo_fatias']):
            if z in proibidas:
                continue
            ids = regioes_da_fatia(mod, tipo, pid, z)
            if ids:
                candidatas.append((pid, z, ids))
    assert len(candidatas) >= 50, f'Só {len(candidatas)} fatias com região em {mod}/{tipo}.'
    rng = np.random.default_rng(semente)
    plano = []
    for _ in range(passos):
        pid, z, ids = candidatas[int(rng.integers(len(candidatas)))]
        plano.append({'paciente': pid, 'fatia': z, 'regiao': ids[int(rng.integers(len(ids)))]})
    return plano, {'fatias': len(candidatas), 'regioes': sum(len(c[2]) for c in candidatas)}


# ---------------------------------------------------------------- ajuste

def episodio(mod, tipo, entrada, ep, aug, semente_passo, lado=LADO):
    """(suporte, máscara, query, alvo) a `lado`, na GPU."""
    s = tensor_fatia(mod, ep['paciente'], ep['fatia'], entrada, lado)
    fg = _mascara_672(mascara_regiao(mod, tipo, ep), lado)
    if aug == 'afim':
        g = torch.Generator().manual_seed(semente_passo)
        u = torch.rand(6, generator=g).tolist()
        ang = math.radians((u[0] * 2 - 1) * 20)
        escala = 0.85 + u[1] * 0.30
        espelho = -1.0 if u[4] < 0.5 else 1.0
        theta = torch.tensor([[espelho * escala * math.cos(ang), -escala * math.sin(ang), (u[2] * 2 - 1) * 0.1],
                              [espelho * escala * math.sin(ang), escala * math.cos(ang), (u[3] * 2 - 1) * 0.1]],
                             dtype=torch.float32, device='cuda')[None]
        grade = F.affine_grid(theta, list(s.shape), align_corners=False)
        q = F.grid_sample(s, grade, mode='bilinear', padding_mode='border', align_corners=False)
        alvo = F.grid_sample(fg[None], grade, mode='nearest', padding_mode='zeros', align_corners=False)[0]
        if alvo.sum() >= 64 * (lado / LADO) ** 2:
            lo, hi = q.min(), q.max()
            q = ((q - lo) / (hi - lo + 1e-6)) ** (0.7 + u[5] * 0.8) * (hi - lo) + lo
            return s, fg, q, alvo.long()
    else:
        assert aug == 'flip', aug
    return s, fg, torch.flip(s, dims=[-1]), torch.flip(fg, dims=[-1]).long()


def treinar(cfg, pasta=None, max_passos=None):
    """cfg: dict do YAML (configs/abdome). Mesma receita e mesmo laço de protovm.eft_regioes."""
    fixar_numerica()
    mod, tipo = cfg['modalidade'], cfg['tipo']
    cfg = dict(cfg)
    cfg.setdefault('entrada', MODALIDADES[mod]['entrada'])
    cfg.setdefault('aug', 'flip')
    cfg.setdefault('excluir', True)
    assert cfg['entrada'] in ENTRADAS
    total = cfg['passos']
    marcos = set(cfg['checkpoints']) | {0}
    # Opcionais (Exp. 12E). 'congelar' só entra no config gravado quando usado, para as retomadas antigas valerem.
    lado = cfg.pop('resolucao', None) or LADO
    congelar = bool(cfg.pop('congelar', False))
    registro_cfg = dict(cfg, resolucao=lado, loss='CE + alinhamento; pesos 1 e 1', otimizador='AdamW')
    if congelar:
        registro_cfg['congelar'] = 'patch_embed + estágios 0 e 1'
    if pasta is None:
        pasta = RAIZ / 'runs' / f"{agora()}_abdtreino_{cfg['nome']}"
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    if (pasta / 'config.json').exists():
        salvo = ler_json(pasta / 'config.json')
        assert {k: salvo.get(k) for k in registro_cfg} == registro_cfg, 'A configuração mudou; retomada recusada.'
        plano = ler_json(pasta / 'plano.json')['episodios']
    else:
        plano, contagem = montar_plano(mod, tipo, total, cfg['semente'], cfg['excluir'])
        gravar_json(pasta / 'plano.json', {'contagem': contagem, 'episodios': plano})
        gravar_json(pasta / 'config.json', dict(registro_cfg, git=git_commit(), ambiente=ambiente()))
        print('Plano:', contagem, flush=True)
    assert len(plano) == total
    print('TREINO:', pasta, flush=True)

    alp = construir_alp(cfg['modelo'], cfg.get('estagio') or 3, cfg.get('norm') or 'checkpoint', lado,
                        grade=cfg.get('grade', 32), treino=True, semente=cfg['semente'])
    if congelar:
        alp.encoder.patch_embed.requires_grad_(False)
        for i in (0, 1):
            alp.encoder.layers[i].requires_grad_(False)
    estado_align = instalar_alinhamento_seguro(alp)
    otimizador = torch.optim.AdamW(alp.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=cfg['weight_decay'])
    latest = pasta / 'checkpoint_latest.pt'

    def salvar(n):
        tmp = latest.with_suffix('.tmp')
        torch.save({'formato': 1, 'passo': n, 'modelo': alp.state_dict(), 'otimizador': otimizador.state_dict(),
                    'rng': estado_rng(), 'config': registro_cfg}, tmp)
        os.replace(tmp, latest)
        destino = pasta / f'checkpoint_{n:04d}.pt'
        if n in marcos and not destino.exists():
            try:
                os.link(latest, destino)
            except OSError:
                shutil.copy2(latest, destino)

    passo = 0
    if latest.exists():
        salvo = torch.load(latest, map_location='cpu', weights_only=True)
        assert salvo['config'] == registro_cfg
        alp.load_state_dict(salvo['modelo'], strict=True)
        otimizador.load_state_dict(salvo['otimizador'])
        passo = int(salvo['passo'])
        restaurar_rng(salvo['rng'])
        del salvo
        print(f'Retomada após o passo {passo}.', flush=True)
    else:
        salvar(0)

    alp.train()
    inicio, primeiro = time.perf_counter(), passo
    fim = min(total, max_passos or total)
    with (pasta / f'passos_{time.time_ns()}.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for n in range(passo + 1, fim + 1):
            ep = plano[n - 1]
            s, fg, q, alvo = episodio(mod, tipo, cfg['entrada'], ep, cfg['aug'], cfg['semente'] * 100003 + n, lado)
            otimizador.zero_grad(set_to_none=True)
            estado_align['ultimo'] = None
            logits, alinhamento = alp([[s]], [[fg]], [[1 - fg]], [q], isval=False, val_wsize=2, show_viz=False)[:2]
            assert tuple(logits.shape) == (1, 2, lado, lado) and torch.isfinite(logits).all()
            ce = F.cross_entropy(logits, alvo)
            loss = ce + alinhamento
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in alp.encoder.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            otimizador.step()
            total_passo, ce_passo = loss.item(), ce.item()
            log.write(json.dumps({'passo': n, **ep, 'ce': ce_passo, 'total': total_passo,
                                  'alinhamento_excecao': estado_align['ultimo']}, ensure_ascii=False) + '\n')
            del logits, alinhamento, ce, loss, grads, s, fg, q, alvo
            if n % cfg['salvar_a_cada'] == 0 or n in marcos or n == fim:
                salvar(n)
            if n == primeiro + 1 or n % 50 == 0:
                media = (time.perf_counter() - inicio) / (n - primeiro)
                print(f'{n}/{total} | loss={total_passo:.5f} | CE={ce_passo:.5f} | {media:.2f}s/passo', flush=True)
    if fim == total:
        gravar_json(pasta / 'concluido.json', {'passos': total})
        print('TREINO CONCLUIDO', flush=True)
    return pasta


# ---------------------------------------------------------------- avaliação

def _dice(pred, gt):
    tp = int((pred & gt).sum())
    return 2 * tp / max(int(pred.sum()) + int(gt.sum()), 1)


def _para_256(t):
    t = t.detach().float()
    while t.dim() < 4:
        t = t[None]
    return F.interpolate(t, size=(256, 256), mode='nearest')[0, 0].cpu().numpy() > 0.5


def _adicionar(linha):
    linhas = []
    if REGISTRO.exists():
        with REGISTRO.open(encoding='utf-8', newline='') as f:
            leitor = csv.DictReader(f)
            antigo = leitor.fieldnames
            linhas = list(leitor)
        if antigo != COLUNAS:  # arquivo anterior à coluna 'mascara': regrava com a coluna vazia
            assert antigo == COLUNAS[:len(antigo)], 'Cabeçalho de resultados/abdome.csv diferente de abdome.COLUNAS.'
            tmp = REGISTRO.with_suffix('.tmp')
            with tmp.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLUNAS)
                w.writeheader()
                w.writerows({c: l.get(c, '') for c in COLUNAS} for l in linhas)
            os.replace(tmp, REGISTRO)
    assert not set(linha) - set(COLUNAS), set(linha) - set(COLUNAS)
    linha = {c: linha.get(c, '') for c in COLUNAS}
    linha['id'] = f'A{len(linhas) + 1:04d}'
    linha['data'] = date.today().isoformat()
    novo = not REGISTRO.exists()
    REGISTRO.parent.mkdir(exist_ok=True)
    with REGISTRO.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        if novo:
            w.writeheader()
        w.writerow(linha)


def usar_maior_score(protosam):
    """Faz o ProtoSAM oficial (que usa sempre a máscara de índice 0) usar a de maior score do SAM."""
    original = protosam.predictor.predict

    def predict(*a, **k):
        mask, score, baixa = original(*a, **k)
        i = int(np.argmax(score))
        ordem = [i] + [j for j in range(len(score)) if j != i]
        return mask[ordem], score[ordem], baixa[ordem]
    protosam.predictor.predict = predict
    protosam.cfg_sam = dict(protosam.cfg_sam, mascara='maior_score')


def avaliar(alp, sam, mod, entrada, metadados, fatias_max=8, suportes=None, rotulo='', registrar=True):
    """Avalia um modelo em todos os suportes da validação de `mod`. sam = construir_protosam(alp, cca)."""
    from models.ProtoSAM import ALPNetInput
    git_inicio = git_commit()
    fixar_numerica()
    semear(42)
    cfg = MODALIDADES[mod]
    wrapper, protosam = sam
    cca = int(protosam.cfg_sam['use_cca'])
    mascara = protosam.cfg_sam.get('mascara', 'indice0')
    pasta = RAIZ / 'runs' / f'{agora()}_abd_{mod}_{rotulo}'
    pasta.mkdir(parents=True, exist_ok=False)
    gravar_json(pasta / 'config.json', {'modalidade': mod, 'entrada': entrada, 'alp': alp.descricao, 'cca': cca,
                                        'mascara': mascara,
                                        'fatias_max': fatias_max, 'metadados': metadados, 'git': git_inicio,
                                        'ambiente': ambiente()})
    print('AVALIAÇÃO:', pasta, flush=True)
    fatias = []
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for sup in (suportes or cfg['suportes']):
            medias = {}
            n = 0
            for orgao, (_, classe) in cfg['orgaos'].items():
                zs = fatias_com(mod, orgao, sup)
                z_sup = zs[len(zs) // 2]
                sup_img = tensor_fatia(mod, sup, z_sup, entrada)
                sup_mask = _mascara_672(volume(mod, sup)['rotulo'][z_sup] == classe)
                assert sup_mask.sum() > 0
                por_paciente = []
                for pid in cfg['validacao']:
                    if pid == sup:
                        continue
                    zq = fatias_com(mod, orgao, pid)
                    escolhidas = [zq[i] for i in sorted(set(np.linspace(0, len(zq) - 1, min(len(zq), fatias_max))
                                                            .round().astype(int).tolist()))]
                    dices = []
                    for z in escolhidas:
                        gt = volume(mod, pid)['rotulo'][z] == classe
                        assert gt.any()
                        qry = tensor_fatia(mod, pid, z, entrada)

                        def entrada_alp():
                            return ALPNetInput(support_images=[sup_img], support_labels=[sup_mask], query_images=qry,
                                               isval=True, val_wsize=2, show_viz=False, supp_fts=None)
                        logits = wrapper(entrada_alp())
                        assert logits.shape[1] == 2 and torch.isfinite(logits).all()
                        d_alp = _dice(_para_256(logits.argmax(1)), gt)
                        saida, _ = protosam(query_image=qry, coarse_model_input=entrada_alp(), degrees_rotate=0)
                        d_sam = _dice(_para_256(torch.as_tensor(saida)), gt)
                        dices.append((d_alp, d_sam))
                        fatias.append({'suporte': sup, 'orgao': orgao, 'paciente': pid, 'fatia': z,
                                       'dice_alp': d_alp, 'dice_sam': d_sam})
                        n += 1
                    por_paciente.append(np.mean(dices, axis=0))
                medias[orgao] = np.mean(por_paciente, axis=0)
            alp_m = float(np.mean([m[0] for m in medias.values()]))
            sam_m = float(np.mean([m[1] for m in medias.values()]))
            print(f'{mod} suporte {sup}: ' + ' | '.join(f'{o} ALP={m[0]:.4f} SAM={m[1]:.4f}' for o, m in medias.items())
                  + f' | média ALP={alp_m:.4f} SAM={sam_m:.4f}', flush=True)
            if registrar:
                d = alp.descricao
                _adicionar({'estado': estado_do_codigo(git_inicio, git_commit()), 'modalidade': mod,
                            'modelo': d['modelo'], 'entrada': entrada, 'aval_estagio': d.get('estagio') or '',
                            'aval_norm': d.get('norm') or '', 'cca': cca, 'mascara': mascara, 'suporte': f'P{sup:02d}', 'n': n,
                            **{f'dice_{e}_{o}': f'{m[i]:.6f}' for o, m in medias.items()
                               for i, e in enumerate(('alp', 'sam'))},
                            'dice_alp': f'{alp_m:.6f}', 'dice_sam': f'{sam_m:.6f}',
                            'commit': (git_inicio['commit'] or '')[:10],
                            'origem': str(pasta.relative_to(RAIZ).as_posix()), **metadados})
    with (pasta / 'fatias.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fatias[0]))
        w.writeheader()
        w.writerows(fatias)
    return pasta


def resumo():
    """Média e desvio entre suportes de cada configuração, e diferença pareada contra o mesmo modelo
    pré-treinado lido do mesmo jeito (mesma modalidade, entrada, estágio, norm e CCA)."""
    with REGISTRO.open(encoding='utf-8', newline='') as f:
        linhas = [dict(l, mascara=l.get('mascara') or 'indice0') for l in csv.DictReader(f) if l['estado'] == 'novo']
    chaves = ['modalidade', 'modelo', 'entrada', 'aval_estagio', 'aval_norm', 'cca', 'mascara', 'pseudo', 'aug',
              'lr', 'excluir', 'pesos', 'passo']
    grupos = {}
    for l in linhas:
        grupos.setdefault(tuple(l[c] for c in chaves), {})[l['suporte']] = l
    saida = []
    for k, g in grupos.items():
        base = grupos.get(k[:7] + ('', '', '', '', 'pre_treinado', ''), {})
        alp = [float(l['dice_alp']) for l in g.values()]
        sam = [float(l['dice_sam']) for l in g.values()]
        dif = [float(g[s]['dice_sam']) - float(base[s]['dice_sam']) for s in g if s in base]
        nome = f"{k[0]} {k[1]} {k[2]} e{k[3] or '-'} cca{k[5]} {k[6]} | " + (
            'pré-treinado' if k[11] == 'pre_treinado' else f"{k[11].split('_abdtreino_')[-1]} passo {k[12]}")
        saida.append(f"{nome} | n={len(g)} ALP {np.mean(alp):.3f}±{np.std(alp, ddof=1) if len(alp) > 1 else 0:.3f} "
                     f"SAM {np.mean(sam):.3f}±{np.std(sam, ddof=1) if len(sam) > 1 else 0:.3f}"
                     + (f" | Δ {np.mean(dif):+.3f} ganha {sum(d > 0 for d in dif)}/{len(dif)}" if dif else ''))
    return saida
