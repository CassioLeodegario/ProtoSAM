# Exp. 15b — Linhas que faltam para o quadro-resumo DINOv2 × VMamba nos 20 suportes

- **Pergunta**: nenhuma nova. Completa o quadro pedido em 11/11/2026: melhor resultado de cada encoder
  (DINOv2-L, DINOv2-S, VMamba-T) por tipo de pseudo-máscara (congelado, regiões do SAM, Felzenszwalb)
  e CCA ligada/desligada, sempre nos 20 suportes.
- **Hipótese que testa**: nenhuma diretamente (apoio às tabelas de H1 e ao Exp. 15).
- **O que é comparado** (só o que ainda não está no registro):
  1. VMamba-T pré-treinado, lido como o adaptado (estágio 2, sem norm, 672, grade 32), 20 suportes;
  2. VMamba-T adaptado com Felzenszwalb (treino_SP256 de out/2026): passos 1000, 2000, 4000, 7000 e 10000
     em S00; o de maior Dice SAM vai para os outros 19 suportes — mesma regra do Exp. 15;
  3. DINOv2-S e DINOv2-L adaptados com regiões do SAM, passo escolhido no Exp. 15, com CCA ligada, 20 suportes.
- **O que fica fixo**: validação V (145 imagens), suportes S00 a S19, encoder a 672, protocolo v1.
  Nenhum treino novo.
- **Métrica**: Dice do ALP e do SAM, média e desvio nos 20 suportes.
- **Regra de decisão**: nenhuma; as decisões são as das fichas E01 (CCA) e E15 (adaptação do DINOv2).
  A CCA não é avaliada nos modelos adaptados com Felzenszwalb: se o quadro mostrar que eles ficam
  muito abaixo dos demais, a linha não muda nenhuma conclusão.
- **Máximo de execuções**: 84 avaliações (cerca de 4 h 30 de GPU).
- **Adendo de 12/11/2026** (escrito antes de rodar, depois de ver o quadro sem CCA): a pedido, a linha
  VMamba-T com Felzenszwalb (treino_SP256, passo 1000) é medida também com CCA ligada nos 20 suportes,
  para o quadro ficar completo. Mais 20 avaliações (máximo 104). Não muda nenhuma regra de decisão.
- **Segundo adendo de 12/11/2026** (escrito antes de rodar): o quadro deve trazer o melhor caso do
  VMamba-T com Felzenszwalb. Pelo histórico de outubro é o treino `eft_ultima672_10000_wd001` de 18/10,
  passo 500, lido na penúltima camada (0,532 ± 0,143 nos 20 suportes, código antigo). Ele é remedido
  pelo código novo nos 20 suportes, com CCA desligada e ligada. O passo não é reescolhido: vale o de
  outubro. Mais 40 avaliações (máximo 144).

## Resultado

A preencher.

## Decisão

A preencher.
