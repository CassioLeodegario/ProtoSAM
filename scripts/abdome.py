"""Abdome (Exp. 12): regiões do SAM, ajuste auto-supervisionado e avaliação nos rins, em RM e CT.

    python scripts/abdome.py regioes MR                       # gera as regiões do SAM das fatias de ajuste
    python scripts/abdome.py treinar configs/abdome/mr_sam.yaml
    python scripts/abdome.py avaliar --modalidade MR --modelo vmamba_tiny --estagio 2 --norm sem_norm
    python scripts/abdome.py avaliar --treino runs/<pasta> --passos 500 1000 --estagio 2 --norm sem_norm
    python scripts/abdome.py resumo

Cada (modelo, suporte) vira uma linha em resultados/abdome.csv. Ver protovm/abdome.py para o protocolo.
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import RAIZ, abdome, config  # noqa: E402

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest='comando', required=True)
r = sub.add_parser('regioes')
r.add_argument('modalidade', choices=list(abdome.MODALIDADES))
t = sub.add_parser('treinar')
t.add_argument('config')
t.add_argument('--run', default=None)
t.add_argument('--max-passos', type=int, default=None)
a = sub.add_parser('avaliar')
a.add_argument('--modalidade', choices=list(abdome.MODALIDADES), default=None)
a.add_argument('--modelo', default=None)
a.add_argument('--treino', default=None, help='pasta runs/<...> de um ajuste abdominal')
a.add_argument('--passos', type=int, nargs='*', default=[])
a.add_argument('--estagio', type=int, default=None)
a.add_argument('--norm', default=None)
a.add_argument('--entrada', choices=abdome.ENTRADAS, default=None)
a.add_argument('--cca', action='store_true')
a.add_argument('--mascara', choices=abdome.MASCARAS, default='indice0')
a.add_argument('--fatias-max', type=int, default=8)
a.add_argument('--suportes', type=int, nargs='*', default=None, help='ensaio: só estes pacientes-suporte')
a.add_argument('--sem-registro', action='store_true', help='ensaio: não grava em resultados/abdome.csv')
a.add_argument('--obs', default='')
sub.add_parser('resumo')
args = p.parse_args()

if args.comando == 'regioes':
    abdome.gerar_regioes_sam(args.modalidade)
elif args.comando == 'treinar':
    abdome.treinar(config.carregar(args.config), pasta=args.run, max_passos=args.max_passos)
elif args.comando == 'resumo':
    print('\n'.join(abdome.resumo()))
else:
    from protovm.avaliar import construir_protosam, exigir_kernel_cuda
    from protovm.encoder import construir_alp
    from protovm.util import ler_json
    if args.treino:
        treino = (RAIZ / args.treino).resolve()
        cfg = ler_json(treino / 'config.json')
        assert args.passos, 'Informe --passos com o treino.'
        mod, modelo = cfg['modalidade'], cfg['modelo']
        entrada = args.entrada or cfg['entrada']
        alvos = [(treino / f'checkpoint_{n:04d}.pt', n) for n in args.passos]
        base = {'pesos': f'eft:{treino.name}', 'pseudo': cfg['tipo'], 'aug': cfg['aug'], 'lr': cfg['lr'],
                'excluir': int(cfg['excluir'])}
        padrao = (cfg.get('estagio') or 3, cfg.get('norm') or 'checkpoint')
    else:
        assert args.modalidade and args.modelo, 'Informe --modalidade e --modelo, ou --treino.'
        mod, modelo = args.modalidade, args.modelo
        entrada = args.entrada or abdome.MODALIDADES[mod]['entrada']
        alvos, base, padrao = [(None, '')], {'pesos': 'pre_treinado'}, (3, 'checkpoint')
    if modelo.startswith('vmamba'):
        exigir_kernel_cuda()
    estagio, norm = args.estagio or padrao[0], args.norm or padrao[1]
    sam = None
    for caminho, passo in alvos:
        alp = construir_alp(modelo, estagio, norm, abdome.LADO, pesos=caminho)
        if sam is None:
            sam = construir_protosam(alp, cca=args.cca)
            if args.mascara == 'maior_score':
                abdome.usar_maior_score(sam[1])
        else:
            sam[0].model = alp
        abdome.avaliar(alp, sam, mod, entrada, dict(base, passo=passo, obs=args.obs), fatias_max=args.fatias_max,
                       suportes=args.suportes, registrar=not args.sem_registro,
                       rotulo=f"{modelo}_{passo or 'pre'}_e{estagio}{'_cca' if args.cca else ''}"
                              f"{'_score' if args.mascara == 'maior_score' else ''}")
        del alp
        gc.collect()
        torch.cuda.empty_cache()
