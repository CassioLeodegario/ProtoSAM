# Exp. 12d — O espelho na query atrapalha o ajuste no abdome?

Escrita em 15/11/2026, antes de rodar. Pedido do Cássio em 15/11.

- **Pergunta**: todos os treinos do Exp. 12 usaram espelho horizontal na query (sempre em `flip`, em metade
  dos episódios em `afim`). No abdome o lado é informação: rim direito e rim esquerdo só se distinguem pela
  posição. Sem o espelho, o ajuste auto-supervisionado melhora o VMamba-T?
- **Hipótese que testa**: apoio à H1 fora de pólipos; verifica uma ressalva do resultado negativo dos
  Exp. 12 e 12c.
- **O que muda**: uma coisa só. `aug: afim_se` é a query deformada do Exp. 12C (rotação ±20°, escala
  0,85–1,15, translação ±10%, gama 0,7–1,5) com os mesmos sorteios e sem espelho. Quando a região
  deformada fica pequena demais, sorteia de novo (até 8 vezes) em vez de cair no espelho.
- **O que fica fixo**: VMamba-T, mesma receita, semente 1234 e mesmo plano de episódios do Exp. 12C;
  avaliação no protocolo oficial do Exp. 12c (CCA, 3 fatias de suporte, todas as fatias, 5 suportes),
  penúltima camada.
- **O que é comparado**: `configs/abdome/{mr,ct}_{sam,sp}_afim_se.yaml`, treinados até o passo 1000
  (`--max-passos 1000`) e medidos nos passos 500 e 1000, contra (a) o VMamba-T pré-treinado e (b) os
  mesmos pares com espelho, ambos já medidos no Exp. 12c. 4 treinos, 8 avaliações.
- **Medida secundária**: fração da área prevista pelo SAM que cai no outro rim
  (`scripts/abdome_diag.py outro`), no pré-treinado (do Exp. 12b, 3 suportes), sem espelho (passo 500) e
  com espelho (passo 500; os 4 pares do Exp. 12C são reavaliados sem registro só para isso). 4 avaliações
  a mais, por último na fila.
- **Métrica**: Dice SAM, média e desvio nos 5 suportes; diferença pareada por suporte contra o
  pré-treinado.
- **Regra de decisão**: a mesma. Um par **melhora** se o Dice SAM subir pelo menos 0,03 sobre o
  pré-treinado e ganhar em pelo menos 4 dos 5 suportes nos passos 500 e 1000. Se nenhum par cumprir, o
  resultado negativo do abdome fica mantido também sem espelho. Digo que o espelho **atrapalhava** se, sem
  ele, o Dice SAM for pelo menos 0,03 maior que o do mesmo par com espelho nos dois passos, em pelo menos
  2 dos 4 pares. Um par que cumprir a regra é repetido com outra semente antes de entrar em qualquer texto.
- **Máximo de execuções**: 4 treinos (cerca de 40 min) e 12 avaliações (cerca de 8,5 h). Pacientes
  reservados não são lidos.

## Resultado

A preencher.

## Decisão

A preencher.
