"""Tabelas geradas a partir de resultados/registro.csv. Nenhum número é digitado.

Uma tabela é descrita por um YAML (ver experimentos/MODELO/tabela.yaml):

    titulo: ...
    conjunto: V                 # só linhas deste conjunto
    cca: 0                      # e desta opção do SAM
    metricas: [dice_alp, dice_sam]
    incluir_importados: false   # true aceita linhas de out/2026 (estado=importado)
    referencia: <nome>          # opcional: diferença pareada de cada linha contra esta
    linhas:
      - nome: DINOv2-L
        filtro: {modelo: dinov2_l14, pesos: pre_treinado}

Cada linha da tabela resume uma configuração sobre os suportes: média, desvio padrão e número de
suportes. A diferença pareada usa só os suportes comuns e traz um intervalo de 95% por bootstrap
(reamostragem dos suportes, semente fixa). Linhas 'rascunho' nunca entram. Se um filtro casar com
mais de uma avaliação do mesmo suporte, a geração para: o filtro precisa ser mais específico.
"""
import random
import statistics

from protovm import registro

REAMOSTRAS = 10000
SEMENTE = 20261111


class TabelaInvalida(ValueError):
    pass


def selecionar(linhas, filtro, conjunto, cca, incluir_importados=False):
    """{suporte: linha} das avaliações que casam com o filtro (colunas do registro, comparadas como texto)."""
    estados = ('novo', 'importado') if incluir_importados else ('novo',)
    desconhecidas = set(filtro) - set(registro.COLUNAS)
    if desconhecidas:
        raise TabelaInvalida(f'Colunas inexistentes no filtro: {sorted(desconhecidas)}')
    por_suporte = {}
    for l in linhas:
        if l['estado'] not in estados or l['conjunto'] != str(conjunto) or l['cca'] != str(cca):
            continue
        if any(l[c] != str(v) for c, v in filtro.items()):
            continue
        if l['suporte'] in por_suporte:
            raise TabelaInvalida(f"Filtro {filtro} casa com {por_suporte[l['suporte']]['id']} e {l['id']} "
                                 f"no suporte {l['suporte']}; acrescente uma coluna ao filtro.")
        por_suporte[l['suporte']] = l
    if not por_suporte:
        raise TabelaInvalida(f'Nenhuma linha do registro casa com {filtro} (conjunto {conjunto}, cca {cca}).')
    return por_suporte


def resumo(valores):
    return {'media': statistics.fmean(valores), 'desvio': statistics.stdev(valores) if len(valores) > 1 else 0.0,
            'n': len(valores)}


def diferenca_pareada(a, b):
    """a e b: {suporte: valor}. Média de a - b nos suportes comuns, com IC de 95% por bootstrap."""
    comuns = sorted(set(a) & set(b))
    if not comuns:
        raise TabelaInvalida('Sem suportes em comum para a diferença pareada.')
    d = [a[s] - b[s] for s in comuns]
    rng = random.Random(SEMENTE)
    medias = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(REAMOSTRAS))
    return {'media': statistics.fmean(d), 'ic': (medias[int(0.025 * REAMOSTRAS)], medias[int(0.975 * REAMOSTRAS) - 1]),
            'n': len(d), 'positivos': sum(x > 0 for x in d)}


def _br(x, casas=3, sinal=False):
    return f'{x:+.{casas}f}'.replace('.', ',') if sinal else f'{x:.{casas}f}'.replace('.', ',')


def gerar(spec, linhas=None):
    """Markdown da tabela descrita por spec (dict do YAML)."""
    linhas = registro.ler() if linhas is None else linhas
    metricas = spec.get('metricas', ['dice_alp', 'dice_sam'])
    conjunto, cca = spec.get('conjunto', 'V'), spec.get('cca', 0)
    dados = {}
    for item in spec['linhas']:
        if item['nome'] in dados:
            raise TabelaInvalida(f"Nome repetido: {item['nome']}")
        dados[item['nome']] = selecionar(linhas, item['filtro'], conjunto, cca, spec.get('incluir_importados', False))
    ref = spec.get('referencia')
    if ref and ref not in dados:
        raise TabelaInvalida(f"A referência '{ref}' não é uma das linhas.")

    cab = ['Configuração', 'Suportes'] + [f'{m} (média ± dp)' for m in metricas]
    if ref:
        cab += [f'Δ {m} vs. {ref} [IC 95%]' for m in metricas] + ['Suportes em que ganha']
    saida = [f"# {spec['titulo']}", '',
             f"Conjunto {conjunto}, CCA {'ligada' if str(cca) == '1' else 'desligada'}. "
             f"Gerado de resultados/registro.csv; não edite à mão.", '',
             '| ' + ' | '.join(cab) + ' |', '|' + '---|' * len(cab)]
    for nome, por_suporte in dados.items():
        celulas = [nome, str(len(por_suporte))]
        valores = {m: {s: float(l[m]) for s, l in por_suporte.items()} for m in metricas}
        for m in metricas:
            r = resumo(list(valores[m].values()))
            celulas.append(f"{_br(r['media'])} ± {_br(r['desvio'])}")
        if ref:
            ganha = '—'
            for m in metricas:
                if nome == ref:
                    celulas.append('—')
                    continue
                base = {s: float(l[m]) for s, l in dados[ref].items()}
                d = diferenca_pareada(valores[m], base)
                celulas.append(f"{_br(d['media'], sinal=True)} [{_br(d['ic'][0], sinal=True)}; {_br(d['ic'][1], sinal=True)}]")
                ganha = f"{d['positivos']}/{d['n']}"  # da última métrica listada
            celulas.append(ganha)
        saida.append('| ' + ' | '.join(celulas) + ' |')
    ids = sorted(l['id'] for por_suporte in dados.values() for l in por_suporte.values())
    saida += ['', f'Registros usados: {ids[0]} a {ids[-1]} ({len(ids)} linhas).' if ids else '']
    if ref:
        saida.append(f'Diferença pareada nos suportes comuns; IC por bootstrap ({REAMOSTRAS} reamostras). '
                     f'"Suportes em que ganha" refere-se a {metricas[-1]}.')
    return '\n'.join(saida) + '\n'
