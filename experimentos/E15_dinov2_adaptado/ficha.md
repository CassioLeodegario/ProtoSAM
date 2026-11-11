# Exp. 15 — a mesma adaptação auto-supervisionada aplicada ao DINOv2

- **Pergunta**: quando o DINOv2 recebe a mesma adaptação que levou o VMamba-Tiny a 0,700, ele melhora tanto quanto?
  Dito de outro modo: o ganho do VMamba vem do encoder ou da receita?
- **Hipótese que testa**: apoio direto a H1 e resposta à banca ("o resultado vem do SSM ou da receita de adaptação?").
- **O que é comparado**: quatro treinos novos, com a receita e os episódios do par que já existe para o VMamba
  (`eft_SAM_vs_SP256_1231`: 1.231 imagens sem rótulo, 10.000 episódios).

  | Encoder | Regiões do SAM | Superpixels de Felzenszwalb |
  |---|---|---|
  | VMamba-Tiny | já treinado em out/2026 | já treinado em out/2026 |
  | DINOv2-S | `configs/eft_regioes/dinov2_s14_sam.yaml` | `configs/eft_regioes/dinov2_s14_sp256.yaml` |
  | DINOv2-L | `configs/eft_regioes/dinov2_l14_sam.yaml` | `configs/eft_regioes/dinov2_l14_sp256.yaml` |

- **O que fica fixo**: 672 px, encoder inteiro ajustado, AdamW com lr 1e-5 e weight decay 0,01, semente 1234, sem AMP,
  perda CE + alinhamento. Nenhum hiperparâmetro é afinado para o DINOv2. Avaliação: validação V, CCA conforme o
  Exp. 1, DINOv2 lido como no congelado (grade 48).
- **Diferenças que não dá para igualar**: o VMamba treina com DropPath 0,2 e o DINOv2 do hub com 0; o VMamba é lido
  na penúltima camada na avaliação e o DINOv2 na última.
- **Escolha do passo**: como foi feito para o VMamba, pelo Dice do SAM com o suporte S00 entre os passos 1000, 2000,
  4000, 7000 e 10000. O passo escolhido é então avaliado nos 20 suportes.
- **Métrica**: Dice do ALP e do SAM, média nos 20 suportes; diferença pareada de cada modelo adaptado contra o mesmo
  modelo congelado (Exp. 2a e 2b) e contra o VMamba adaptado, com IC de 95%.
- **Regra de decisão**:
  - Se o DINOv2-S adaptado ficar acima do VMamba-Tiny adaptado no ALP (IC da diferença inteiro acima de zero), a
    dissertação afirma que o ganho vem da receita e que, com custo parecido, a atenção ainda vence; H1 é discutida
    com esse resultado contra ela.
  - Se o IC contiver zero, os dois ficam empatados em qualidade e a comparação passa a ser decidida pelo custo (H2).
  - Se o VMamba ficar acima, o resultado sustenta H1 para este domínio.
  - Em qualquer caso, o ganho do DINOv2 com a adaptação (adaptado − congelado) é reportado ao lado do ganho do VMamba.
- **Máximo de execuções**: 4 treinos (cerca de 13 horas de GPU), 20 avaliações com S00 e 80 nos 20 suportes.

## Resultado

A preencher com as tabelas geradas.

## Decisão

A preencher.
