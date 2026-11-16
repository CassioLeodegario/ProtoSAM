# Exp. 12d — Gargalos de entrada e de representação do VMamba-T congelado (sem treino)

Escrita em 15/11/2026, antes de rodar. Plano do Cássio de 15/11.

- **Pergunta**: quanto do déficit do VMamba-T congelado no protocolo oficial (Dice SAM 0,504 em RM e 0,350
  em CT no Exp. 12c) vem da redução da grade (42 → 32), da falta de posição 2D (`posembed=False`) e da
  normalização da entrada em CT e RM? E o treino `ct_sam_d4_penultima`, já salvo, melhora no protocolo
  oficial?
- **Hipótese que testa**: apoio à H1. Só mudam o encoder (`f_theta`) e a normalização da entrada; ALP, CCA,
  prompts e SAM-H ficam intocados.
- **O que é comparado** (`scripts/abdome.py avaliar --protocolo oficial`, 5 suportes, um fator por vez
  sobre a referência do Exp. 12c: VMamba-T, penúltima camada, `sem_norm`, 672 px, grade 32, entrada
  `zscore` em RM e `imagenet` em CT):
  1. **Resgate, sem treino novo**:
     - `runs/*_abdtreino_ct_sam_d4_penultima`, passos 250 e 500, em CT;
     - VMamba-T congelado em RM com `--entrada imagenet`.
  2. **Entrada limpa**:
     - CT, `--entrada imagenet_fixo`: sem o corte por percentil de cada paciente; escala fixa
       (mínimo e máximo do volume, que já vem janelado) para 0–255, depois ImageNet;
     - RM, `--entrada zscore_corpo` e `--entrada imagenet_corpo`: média, desvio e percentis 0,5 e 99,5
       calculados só nos voxels dentro do corpo (`fgmask > 0`);
     - as três também no DINOv2-L congelado, como controle. Para o DINOv2-L em RM mede-se também
       `--entrada imagenet`, para a comparação de `imagenet_corpo` ter a mesma referência do VMamba-T.
  3. **Grade nativa**: `--grade 42` a 672 px (sem reduzir 42 × 42 para 32 × 32) e
     `--resolucao 768 --grade 48` (a grade do DINOv2-L, sem interpolação), em RM e CT.
  4. **Posição 2D canônica** (`--pos-can 0.20`): 4 canais determinísticos
     `0,20 · [cos(πx), sin(πx), cos(πy), sin(πy)]`, com x e y em [−1, 1] relativos à caixa do corpo,
     concatenados aos tokens normalizados (L2) em `get_features`. RM e CT.
  5. **Multi-escala assimétrica** (`--estagio 12 --grade 84 --peso-e1 0.15`): estágios 1 e 2
     normalizados separadamente (`ln_unitaria` + L2), F2 interpolado para 84 × 84 e concatenado como
     `[√0,85 · F2, √0,15 · F1]`. RM e CT.
  6. **Linha combinada**: só os fatores de 2 a 5 que cumprirem a regra individual, em RM e CT.
- **O que fica fixo**: pacientes (reservados RM 0–4 e CT 0–6 nunca lidos; validação RM 5–9 e CT 7–14;
  suportes RM 5–9 e CT 7–11), protocolo oficial (CCA, 3 fatias de suporte, todas as fatias com rim),
  SAM-H com pontos + caixa, máscara de índice 0. Toda opção nova tem padrão igual ao comportamento atual.
- **Métrica**: `dice_alp` e `dice_sam` (2D, por fatia) e `dice_vol` (3D, por paciente), média e desvio
  nos 5 suportes; diferença pareada por suporte contra a referência do Exp. 12c da mesma modalidade.
  Nos itens 2 e 3 do DINOv2-L, contra o DINOv2-L do Exp. 12c.
- **Regra de decisão**: um fator entra na base do Exp. 12e se subir pelo menos 0,02 em `dice_sam` (ou
  pelo menos 0,03 em `dice_alp`) e ganhar em pelo menos 4 dos 5 suportes, sem piorar a outra modalidade
  (queda de `dice_sam` maior que 0,01). Fatores de entrada valem só para a sua modalidade. O treino
  `d4` conta como resgatado se cumprir a regra do Exp. 12 (+0,03 e 4 de 5 suportes nos dois passos).
  Se a entrada limpa também subir o DINOv2-L, a mudança é de protocolo e vale para os dois modelos; se
  subir só o VMamba-T, fica registrada como ganho do VMamba-T. A linha combinada só é a nova base se
  for pelo menos tão boa quanto o melhor fator sozinho. Como vários fatores são escolhidos na mesma
  validação, o número final da base é remedido no Exp. 12e como referência, não comparado com o 12c.
- **Máximo de execuções**: 10 avaliações em RM e 13 em CT (cerca de 11 h de GPU). Nenhum treino.

## Resultado

A preencher.

## Decisão

A preencher.
