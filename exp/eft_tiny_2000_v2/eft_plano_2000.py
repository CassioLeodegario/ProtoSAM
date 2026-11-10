"""Plano determinístico; preserva exatamente os primeiros 200 episódios."""
import numpy as np

def estender(treino, piloto, validacao, total=2000):
    treino = sorted(treino, key=lambda r: r['indice'])
    ti = {r['indice'] for r in treino}
    vi = {r['indice'] for r in validacao}
    assert len(treino) == len(ti) == 1304
    assert len(validacao) == len(vi) == 145 and not ti & vi
    assert all(r['split'] == 'treino' and r['candidatos'] for r in treino)
    rng = np.random.default_rng(1234)
    plano = []
    while len(plano) < total:
        for idx in rng.permutation(len(treino)):
            r = treino[int(idx)]
            plano.append(dict(r, regiao=int(rng.choice(r['candidatos']))))
            if len(plano) == total:
                break
    assert plano[:200] == piloto, 'Primeiros 200 episódios diferem do piloto; interrompido.'
    assert len({r['indice'] for r in plano[:1304]}) == 1304
    return plano
