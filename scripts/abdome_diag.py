"""Diagnóstico do protocolo do abdome (Exp. 12b). Não grava em resultados/abdome.csv.

    python scripts/abdome_diag.py medir --modalidade CT --modelo dinov2_l14
    python scripts/abdome_diag.py medir --modalidade MR --modelo vmamba_tiny --estagio 2 --norm sem_norm
    python scripts/abdome_diag.py resumir

Para cada (suporte, rim, paciente de query) mede TODAS as fatias com o rim, com o modelo pré-treinado:
  variante 'meio'   -> suporte = fatia central do rim (protocolo do Exp. 12)
  variante 'partes' -> suporte da mesma terça parte do rim (código oficial: 3 partes, 1/6, 1/2 e 5/6)
  etapa 'alp', 'sam' (CCA desligada), 'sam_cca' (CCA ligada)
  etapa 'oraculo'   -> SAM com prompts tirados da máscara verdadeira (teto da etapa do SAM)
Grava uma linha por (fatia, variante, etapa) em rascunho/E12b_diag/<MOD>_<modelo>.csv, com a área
prevista, o acerto no rim-alvo e a sobreposição com o OUTRO rim.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import RAIZ, abdome  # noqa: E402
from protovm.abdome import LADO, MODALIDADES, _mascara_672, _para_256, fatias_com, tensor_fatia, volume  # noqa: E402

DESTINO = RAIZ / 'rascunho' / 'E12b_diag'
PCTS = [1 / 6, 1 / 2, 5 / 6]
CAMPOS = ['mod', 'modelo', 'sup', 'orgao', 'pid', 'z', 'i', 'n', 'parte', 'sub8', 'gt', 'variante', 'z_sup',
          'etapa', 'pred', 'tp', 'tp_outro', 'dice']


def parte(z, zs):
    """Terça parte do rim a que a fatia pertence (fórmula de dataloaders/common.py)."""
    z0, z1 = min(zs), max(zs)
    p = int((z - z0) // ((z1 - z0) / 3)) if z1 > z0 else 0
    return min(max(p, 0), 2)


def medir(args):
    from models.ProtoSAM import ALPNetInput
    from protovm.avaliar import construir_protosam, exigir_kernel_cuda
    from protovm.encoder import construir_alp
    from protovm.util import fixar_numerica, semear
    fixar_numerica()
    semear(42)
    mod = args.modalidade
    cfg = MODALIDADES[mod]
    entrada = cfg['entrada']
    if args.modelo.startswith('vmamba'):
        exigir_kernel_cuda()
    alp = construir_alp(args.modelo, args.estagio or 3, args.norm or 'checkpoint', LADO)
    wrapper, protosam = construir_protosam(alp, cca=False)
    DESTINO.mkdir(parents=True, exist_ok=True)
    arquivo = DESTINO / f'{mod}_{args.modelo}{args.sufixo}.csv'
    oraculo_feito, n_fatias = set(), 0
    with arquivo.open('w', newline='', encoding='utf-8') as f, torch.no_grad(), torch.autocast('cuda', enabled=False):
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for sup in cfg['suportes'][:args.suportes]:
            for orgao, (_, classe) in cfg['orgaos'].items():
                zs = fatias_com(mod, orgao, sup)
                z_meio = zs[len(zs) // 2]
                z_partes = [zs[int(p * len(zs))] for p in PCTS]
                sups = {z: (tensor_fatia(mod, sup, z, entrada), _mascara_672(volume(mod, sup)['rotulo'][z] == classe))
                        for z in {z_meio, *z_partes}}
                for pid in cfg['validacao']:
                    if pid == sup:
                        continue
                    zq = fatias_com(mod, orgao, pid)
                    sub8 = set(np.linspace(0, len(zq) - 1, min(len(zq), 8)).round().astype(int).tolist())
                    for i, z in enumerate(zq):
                        if args.limite and n_fatias >= args.limite:
                            break
                        n_fatias += 1
                        rot = volume(mod, pid)['rotulo'][z]
                        gt, outro = rot == classe, rot == (5 - classe)
                        qry = tensor_fatia(mod, pid, z, entrada)
                        p = parte(z, zq)
                        base = {'mod': mod, 'modelo': args.modelo, 'sup': sup, 'orgao': orgao, 'pid': pid, 'z': z,
                                'i': i, 'n': len(zq), 'parte': p, 'sub8': int(i in sub8), 'gt': int(gt.sum())}

                        def grava(variante, z_sup, etapa, pred):
                            tp = int((pred & gt).sum())
                            w.writerow(dict(base, variante=variante, z_sup=z_sup, etapa=etapa, pred=int(pred.sum()),
                                            tp=tp, tp_outro=int((pred & outro).sum()),
                                            dice=f'{2 * tp / max(int(pred.sum()) + int(gt.sum()), 1):.6f}'))

                        feitos = {}
                        for variante, z_sup in (('meio', z_meio), ('partes', z_partes[p])):
                            if z_sup not in feitos:
                                s_img, s_mask = sups[z_sup]

                                def entrada_alp():
                                    return ALPNetInput(support_images=[s_img], support_labels=[s_mask],
                                                       query_images=qry, isval=True, val_wsize=2, show_viz=False,
                                                       supp_fts=None)
                                r = {'alp': _para_256(wrapper(entrada_alp()).argmax(1))}
                                for etapa, cca in (('sam', False), ('sam_cca', True)):
                                    protosam.use_cca = cca
                                    saida, _ = protosam(query_image=qry, coarse_model_input=entrada_alp(),
                                                        degrees_rotate=0)
                                    r[etapa] = _para_256(torch.as_tensor(saida))
                                protosam.use_cca = False
                                feitos[z_sup] = r
                            for etapa, pred in feitos[z_sup].items():
                                grava(variante, z_sup, etapa, pred)
                        if (orgao, pid, z) not in oraculo_feito:
                            oraculo_feito.add((orgao, pid, z))
                            m = _mascara_672(gt)[None]
                            logits_gt = torch.cat([(1 - m) * 10, m * 10], dim=1)
                            tinha = 'coarse_segmentation_model' in protosam.__dict__
                            real = protosam.__dict__.get('coarse_segmentation_model')
                            protosam.__dict__['coarse_segmentation_model'] = lambda _: logits_gt
                            try:
                                s_img, s_mask = sups[z_meio]
                                saida, _ = protosam(query_image=qry, coarse_model_input=ALPNetInput(
                                    support_images=[s_img], support_labels=[s_mask], query_images=qry, isval=True,
                                    val_wsize=2, show_viz=False, supp_fts=None), degrees_rotate=0)
                            finally:
                                if tinha:
                                    protosam.__dict__['coarse_segmentation_model'] = real
                                else:
                                    del protosam.__dict__['coarse_segmentation_model']
                            grava('-', '', 'oraculo', _para_256(torch.as_tensor(saida)))
                    f.flush()
                print(f'{mod} {args.modelo} suporte {sup} {orgao}: {n_fatias} fatias', flush=True)
    print('GRAVADO', arquivo, flush=True)


def _media(v):
    return float(np.mean(v)) if len(v) else float('nan')


def resumir(_args):
    for arquivo in sorted(DESTINO.glob('*.csv')):
        with arquivo.open(encoding='utf-8', newline='') as f:
            linhas = list(csv.DictReader(f))
        for l in linhas:
            for c in ('sup', 'pid', 'z', 'i', 'n', 'parte', 'sub8', 'gt', 'pred', 'tp', 'tp_outro'):
                l[c] = int(l[c])
            l['dice'] = float(l['dice'])
        print(f'\n===== {arquivo.name} ({len(linhas)} linhas, suportes {sorted({l["sup"] for l in linhas})})')
        print('variante etapa   | fatias | Dice fatia | Dice sub8/paciente | Dice volume | erra tudo | área no alvo | '
              'área no outro rim | pred/gt')
        grupos = defaultdict(list)
        for l in linhas:
            grupos[(l['variante'], l['etapa'])].append(l)
        for (variante, etapa), g in sorted(grupos.items()):
            pac, vol = defaultdict(list), defaultdict(lambda: [0, 0, 0])
            for l in g:
                k = (l['sup'], l['orgao'], l['pid'])
                if l['sub8']:
                    pac[k].append(l['dice'])
                vol[k][0] += l['tp']
                vol[k][1] += l['pred']
                vol[k][2] += l['gt']
            por_sup = defaultdict(lambda: defaultdict(list))
            for (s, o, _), d in pac.items():
                por_sup[s][o].append(np.mean(d))
            sub8 = _media([_media([_media(v) for v in orgs.values()]) for orgs in por_sup.values()])
            pred, tp, outro, gt = (sum(l[c] for l in g) for c in ('pred', 'tp', 'tp_outro', 'gt'))
            print(f'{variante:8s} {etapa:7s} | {len(g):6d} | {_media([l["dice"] for l in g]):.3f}      | {sub8:.3f}'
                  f'              | {_media([2 * a / max(b + c, 1) for a, b, c in vol.values()]):.3f}       | '
                  f'{_media([l["tp"] == 0 for l in g]):.2f}      | {tp / max(pred, 1):.2f}         | '
                  f'{outro / max(pred, 1):.2f}              | {pred / max(gt, 1):.2f}')
        print('Dice por fatia, por terça parte do rim (0 = início, 1 = meio, 2 = fim) e por rim:')
        for (variante, etapa), g in sorted(grupos.items()):
            partes = ' '.join(f'{_media([l["dice"] for l in g if l["parte"] == p]):.3f}' for p in range(3))
            orgs = ' '.join(f'{o}={_media([l["dice"] for l in g if l["orgao"] == o]):.3f}'
                            for o in sorted({l['orgao'] for l in g}))
            areas = ' '.join(f'{_media([l["dice"] for l in g if a <= l["gt"] < b]):.3f}'
                             for a, b in ((0, 100), (100, 300), (300, 10 ** 9)))
            print(f'{variante:8s} {etapa:7s} | partes {partes} | {orgs} | área gt <100, 100-300, >=300 px: {areas}')


def outro_rim(args):
    """Fração da área prevista pelo SAM que cai no outro rim, a partir do fatias.csv de avaliações."""
    for pasta in args.pastas:
        with (Path(pasta) / 'fatias.csv').open(encoding='utf-8', newline='') as f:
            linhas = list(csv.DictReader(f))
        sups = sorted({int(l['suporte']) for l in linhas})
        for nome, usar in (('3 primeiros suportes', sups[:3]), ('todos', sups)):
            g = [l for l in linhas if int(l['suporte']) in usar]
            pred, outro = sum(int(l['pred_sam']) for l in g), sum(int(l['outro_sam']) for l in g)
            print(f"{Path(pasta).name} | {nome} | fatias {len(g)} | Dice SAM por fatia "
                  f"{_media([float(l['dice_sam']) for l in g]):.3f} | área no outro rim {outro / max(pred, 1):.3f}")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest='comando', required=True)
m = sub.add_parser('medir')
m.add_argument('--modalidade', choices=list(MODALIDADES), required=True)
m.add_argument('--modelo', required=True)
m.add_argument('--estagio', type=int, default=None)
m.add_argument('--norm', default=None)
m.add_argument('--suportes', type=int, default=3, help='quantos pacientes-suporte (os primeiros da lista)')
m.add_argument('--limite', type=int, default=None, help='ensaio: para depois de N fatias')
m.add_argument('--sufixo', default='')
sub.add_parser('resumir')
o = sub.add_parser('outro')
o.add_argument('pastas', nargs='+')
args = p.parse_args()
{'medir': medir, 'resumir': resumir, 'outro': outro_rim}[args.comando](args)
