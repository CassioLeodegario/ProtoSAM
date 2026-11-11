# Exp. 2b — DINOv2-B e DINOv2-S nos 20 suportes

- **Pergunta**: quanto o Dice cai, na média dos 20 suportes, quando o DINOv2-L é trocado pelo B e pelo S?
- **Hipótese que testa**: nenhuma diretamente (apoio). Responde à banca sobre as alternativas eficientes:
  o DINOv2-S é o controle de custo do VMamba-Tiny.
- **O que é comparado**: DINOv2-B/14 e DINOv2-S/14 pré-treinados, contra o DINOv2-L do Exp. 2a.
  `python scripts/avaliar.py --modelo dinov2_b14 --suportes todos --sem-mascaras`
  `python scripts/avaliar.py --modelo dinov2_s14 --suportes todos --sem-mascaras`
- **O que fica fixo**: validação V (145 imagens), suportes S00 a S19, encoder a 672, grade 48, CCA desligada, protocolo v1.
- **Métrica**: Dice do ALP e do SAM, média e desvio nos 20 suportes; diferença pareada contra o DINOv2-L com IC de 95%.
- **Regra de decisão**: se o IC da diferença do DINOv2-S contra o L no SAM contiver zero ou ficar acima de −0,02,
  o gatilho do dia 12 do plano dispara: a conclusão passa a ser que o encoder pequeno com atenção já basta,
  e o roteiro do texto se ajusta, sem novos experimentos. Caso contrário, o plano segue como está.
- **Máximo de execuções**: 40 avaliações (cerca de 1 h 40 de GPU).

## Resultado

Ver [tabela.md](tabela.md), gerada depois das avaliações.

## Decisão

A preencher.
