"""Avaliação ALP + SAM-H. Cada (checkpoint, suporte) vira uma linha em resultados/registro.csv.

Exemplos:
    # VMamba-Tiny pré-treinado em V
    python scripts/avaliar.py --modelo vmamba_tiny
    # EFT: passos 1000 a 3000 de um treino, avaliados a 512
    python scripts/avaliar.py --treino runs/<pasta> --passos 1000 2000 3000 --resolucao 512
    # 20 suportes
    python scripts/avaliar.py --treino runs/<pasta> --passos 2000 --suportes todos
    # DINOv2-L (referência); dinov2_b14 e dinov2_s14 também existem
    python scripts/avaliar.py --modelo dinov2_l14
    # Com a CCA oficial ligada no SAM (coluna cca=1 no registro)
    python scripts/avaliar.py --modelo dinov2_l14 --cca
    # Penúltimo estágio do VMamba com o mapa nativo de 42 em vez de 32
    python scripts/avaliar.py --modelo vmamba_tiny --estagio 2 --norm sem_norm --grade 42
    # Checkpoint avulso (ex.: os do ProtoSAM-repro), informando como foi treinado
    python scripts/avaliar.py --modelo vmamba_tiny --pesos <...>/checkpoint_2000.pt --treino-desc 3 checkpoint 256
    # Ensaio rápido: só as 10 primeiras imagens, fora do registro
    python scripts/avaliar.py --modelo vmamba_tiny --limite 10
    # Teste final (T): só para um item já commitado de protocolo/finais.yaml
    python scripts/avaliar.py --treino runs/<pasta> --passos 2000 --conjunto T --final <nome> --suportes todos

Sem --estagio/--norm, usa os do treino (3/checkpoint no pré-treinado). Sem --resolucao, avalia a 672.
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import RAIZ, protocolo  # noqa: E402
from protovm.avaliar import Datasets, avaliar, construir_protosam  # noqa: E402
from protovm.encoder import construir_alp  # noqa: E402
from protovm.util import git_commit, ler_json, nome_do_treino  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--modelo', default=None,
               help='vmamba_tiny|vmamba_small|vmamba_base|dinov2_s14|dinov2_b14|dinov2_l14 (sem --treino)')
p.add_argument('--treino', default=None, help='pasta runs/<...> de um treino EFT')
p.add_argument('--passos', type=int, nargs='*', default=[], help='checkpoints do treino a avaliar')
p.add_argument('--pesos', default=None, help='checkpoint avulso (chave "modelo"), com --modelo')
p.add_argument('--treino-desc', nargs=3, metavar=('ESTAGIO', 'NORM', 'RESOLUCAO'), default=None,
               help='como o checkpoint avulso foi treinado (vai para o registro)')
p.add_argument('--estagio', type=int, default=None)
p.add_argument('--norm', default=None)
p.add_argument('--resolucao', type=int, default=None)
p.add_argument('--grade', type=int, default=None, help='lado do mapa entregue ao ALP no VMamba (padrão 32)')
p.add_argument('--cca', action='store_true', help='liga a análise de componentes conexos oficial no SAM')
p.add_argument('--conjunto', default='V', choices=['V', 'T'])
p.add_argument('--final', default=None, help='item de protocolo/finais.yaml; obrigatório com --conjunto T')
p.add_argument('--suportes', nargs='*', default=['S00'], help="ids (S00..S19) ou 'todos'")
p.add_argument('--limite', type=int, default=None, help='só as N primeiras imagens; não entra no registro')
p.add_argument('--obs', default='')
p.add_argument('--sem-mascaras', action='store_true', help='não salvar PNGs das máscaras')


def principal(a):
    if a.conjunto in protocolo.TRAVADOS:
        protocolo.item_final(a.final, git_commit())  # recusa antes de carregar os modelos
    suportes = protocolo.ids_suportes() if a.suportes == ['todos'] else a.suportes

    if a.treino:
        treino = (RAIZ / a.treino).resolve()
        cfg = ler_json(treino / 'config.json')
        assert a.passos, 'Informe --passos com o treino.'
        alvos = [(treino / f'checkpoint_{n:04d}.pt', n) for n in a.passos]
        modelo = cfg['modelo']
        base = {'pesos': f'eft:{treino.name}', 'treino_estagio': cfg['estagio'], 'treino_norm': cfg['norm'],
                'treino_resolucao': cfg['resolucao']}
        padrao = (cfg['estagio'], cfg['norm'], 672)  # resolução de avaliação padrão do protocolo
    elif a.pesos:
        assert a.modelo and a.treino_desc, 'Checkpoint avulso exige --modelo e --treino-desc.'
        caminho = Path(a.pesos).resolve()
        passo = int(torch.load(caminho, map_location='cpu', weights_only=True)['passo'])
        alvos, modelo = [(caminho, passo)], a.modelo
        est, nrm, res = a.treino_desc
        base = {'pesos': f'eft:{nome_do_treino(caminho)}', 'treino_estagio': int(est), 'treino_norm': nrm,
                'treino_resolucao': int(res)}
        padrao = (int(est), nrm, 672)
    else:
        assert a.modelo, 'Informe --modelo ou --treino.'
        alvos, modelo, base, padrao = [(None, '')], a.modelo, {'pesos': 'pre_treinado'}, (3, 'checkpoint', 672)

    estagio = a.estagio or padrao[0]
    norm = a.norm or padrao[1]
    resolucao = a.resolucao or padrao[2]
    assert a.grade is None or not modelo.startswith('dinov2'), 'No DINOv2 a grade é resolução // 14.'
    grade = a.grade or (cfg.get('grade', 32) if a.treino else 32)
    variacao = (f'_g{grade}' if a.grade else '') + ('_cca' if a.cca else '')
    datasets = Datasets()
    sam = None
    for caminho, passo in alvos:
        alp = construir_alp(modelo, estagio, norm, resolucao, grade=grade, pesos=caminho)
        # O SAM é o mesmo para todos os checkpoints: só troca o ALP dentro do wrapper.
        if sam is None:
            sam = construir_protosam(alp, cca=a.cca)
        else:
            sam[0].model = alp
        treino_curto = base['pesos'].removeprefix('eft:').replace('/', '__')[-64:]
        for s in suportes:
            rotulo = f"{modelo}_{treino_curto}_{passo or 'pre'}_r{resolucao}{variacao}"
            avaliar(alp, a.conjunto, s, rotulo=rotulo, datasets=datasets, wrapper_sam=sam,
                    salvar_mascaras=not a.sem_mascaras, final=a.final, limite=a.limite,
                    metadados=dict(base, passo=passo, obs=a.obs))
        del alp
        gc.collect()
        torch.cuda.empty_cache()


try:
    principal(p.parse_args())
except protocolo.Trava as erro:
    sys.exit(f'TRAVA: {erro}')
