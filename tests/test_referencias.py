"""Números de referência: o avaliador tem de repetir, nas 10 primeiras imagens de V com o suporte
S00, o Dice das avaliações completas R0084 a R0086 (protocolo/referencias_v1.json).

    python tests/test_referencias.py     # cerca de 2 minutos de GPU

Rode antes de cada série de avaliações e depois de qualquer mudança em protovm/, models/ ou
dataloaders/. Os checkpoints são procurados em ../ProtoSAM-repro/reproducao (a variável
PROTOVM_REPRODUCAO troca a pasta). Um caso cujo checkpoint não existe na máquina é pulado e avisado.
"""
import csv
import gc
import os
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def conferir():
    """Reavalia cada caso de referência; devolve os nomes dos casos pulados."""
    import torch
    from protovm import protocolo
    from protovm.avaliar import Datasets, avaliar, construir_protosam
    from protovm.encoder import construir_alp

    ref = protocolo.referencias()['avaliacoes']
    n, tolerancia = len(ref['imagens']), ref['tolerancia_media']
    reproducao = Path(os.environ.get('PROTOVM_REPRODUCAO', RAIZ.parent / 'ProtoSAM-repro' / 'reproducao'))
    datasets = Datasets()
    falhas, pulados = [], []
    for caso in ref['casos']:
        pesos = reproducao / caso['checkpoint'] if caso['checkpoint'] else None
        if pesos is not None and not pesos.exists():
            pulados.append(caso['nome'])
            print(f"PULADO {caso['nome']}: não achei {pesos}", flush=True)
            continue
        alp = construir_alp(caso['modelo'], caso['estagio'] or 3, caso['norm'] or 'checkpoint',
                            caso['resolucao'], pesos=pesos)
        assert alp.descricao['pesos_sha256'] == caso['checkpoint_sha256'], \
            f"{caso['nome']}: o checkpoint encontrado não é o de referência"
        sam = construir_protosam(alp)
        pasta = avaliar(alp, ref['conjunto'], ref['suporte'], rotulo='referencia_' + caso['nome'],
                        datasets=datasets, wrapper_sam=sam, salvar_mascaras=False, limite=n)
        with (pasta / 'por_imagem.csv').open(encoding='utf-8', newline='') as f:
            linhas = list(csv.DictReader(f))
        assert [l['query'] for l in linhas] == ref['imagens'], f"{caso['nome']}: imagens fora da ordem de referência"
        passou = True
        for etapa, coluna in (('alp', 'ALP_Dice'), ('sam', 'SAM_Dice')):
            obtido = [float(l[coluna]) for l in linhas]
            media_ref, media = sum(caso[etapa]) / n, sum(obtido) / n
            maior = max(abs(o - r) for o, r in zip(obtido, caso[etapa]))
            certo = abs(media - media_ref) <= tolerancia
            passou = passou and certo
            print(f"{'ok    ' if certo else 'FALHOU'} {caso['nome']} {etapa.upper()}: referência {media_ref:.6f}, "
                  f"obtido {media:.6f}, maior diferença em uma imagem {maior:.6f}", flush=True)
        if passou:
            shutil.rmtree(pasta)  # pasta criada por este teste; fica só quando falha, para inspeção
        else:
            falhas.append(f"{caso['nome']} (ver {pasta})")
        del alp, sam
        gc.collect()
        torch.cuda.empty_cache()
    assert not falhas, 'Números de referência não reproduzidos: ' + '; '.join(falhas)
    assert len(pulados) < len(ref['casos']), 'Nenhum caso de referência pôde ser avaliado.'
    return pulados


def test_numeros_de_referencia():
    conferir()


if __name__ == '__main__':
    pulados = conferir()
    print('OK: números de referência reproduzidos.' + (f" Pulados: {', '.join(pulados)}." if pulados else ''))
