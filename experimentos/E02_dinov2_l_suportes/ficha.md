# Exp. 2a — DINOv2-L nos 20 suportes: média e variância da referência

- **Pergunta**: qual é a média e o desvio padrão do Dice do DINOv2-L na validação quando o suporte varia entre S00 e S19?
- **Hipótese que testa**: nenhuma (apoio). É a linha de referência das tabelas de H1 e H3; em out/2026 só S00 e S01 tinham sido avaliados.
- **O que é comparado**: uma configuração só, DINOv2-L/14 pré-treinado.
  `python scripts/avaliar.py --modelo dinov2_l14 --suportes S01 ... S19 --sem-mascaras`
  (S00 já está no registro como R0085, mesma GPU e mesmo código de avaliação).
- **O que fica fixo**: validação V (145 imagens), encoder a 672, grade 48, CCA desligada, protocolo v1.
- **Métrica**: Dice do ALP e do SAM por suporte; média e desvio padrão nos 20 suportes.
- **Regra de decisão**: experimento descritivo, sem escolha a fazer. O resultado vira a linha de referência
  das comparações pareadas. Se o desvio entre suportes passar de 0,10 no SAM, as conclusões da dissertação
  usam sempre os 20 suportes, nunca S00 sozinho.
- **Máximo de execuções**: 19 avaliações (cerca de 50 minutos de GPU).

## Resultado

Ver [tabela.md](tabela.md), gerada depois das avaliações.

## Decisão

A preencher.
