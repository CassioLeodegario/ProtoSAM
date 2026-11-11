# Exp. N — título curto

Copie a pasta `MODELO` para `experimentos/ENN_nome/`, preencha até "Regra de decisão" e faça o
commit **antes** da primeira execução. As duas últimas seções são preenchidas depois.

- **Pergunta**: uma frase, respondível com sim, não ou um número.
- **Hipótese que testa**: H1, H2, H3 ou nenhuma (experimento de apoio).
- **O que é comparado**: as configurações, uma por linha, com o comando ou o YAML de cada uma.
- **O que fica fixo**: conjunto, suportes, resolução, grade, CCA, sementes.
- **Métrica**: por exemplo, Dice do ALP, média nos 20 suportes, diferença pareada com IC de 95%.
- **Regra de decisão**: o que faço com cada resultado possível, escrito antes de ver o número.
- **Máximo de execuções**: quantas avaliações e treinos este experimento pode gastar.

## Resultado

Tabela gerada por `python scripts/gerar_tabelas.py experimentos/ENN_nome` (não digite números aqui;
aponte para o arquivo `tabela.md`). Registros usados: R.... a R.....

## Decisão

O que a regra acima mandou fazer, em uma ou duas frases, com a data.
