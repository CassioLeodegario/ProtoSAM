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

Medido em 15 e 16/11/2026 (`runs/E12d_representacao.log`, 10 avaliações em RM e 6 em CT). Dice SAM 2D,
média nos 5 suportes; Δ é a diferença pareada contra a referência do Exp. 12c do mesmo modelo, com os
suportes em que ganha; ΔALP é a mesma diferença no Dice do ALP; 3D é o `dice_vol`.

**VMamba-T, RM** (referência 0,504; ALP 0,385)

| Fator | SAM | Δ | ΔALP | 3D |
|---|---|---|---|---|
| Entrada `imagenet` | 0,515 | +0,012 (3/5) | −0,001 (2/5) | 0,542 |
| Entrada `zscore_corpo` | 0,507 | +0,003 (2/5) | −0,001 (2/5) | 0,549 |
| Entrada `imagenet_corpo` | 0,506 | +0,002 (3/5) | −0,005 (4/5) | 0,525 |
| Grade 42 | 0,512 | +0,009 (3/5) | −0,016 (0/5) | 0,549 |
| 768 px, grade 48 | 0,520 | +0,016 (3/5) | −0,027 (0/5) | 0,556 |
| **Posição 2D (0,20)** | **0,728** | **+0,224 (5/5)** | **+0,166 (5/5)** | 0,769 |
| Multi-escala (estágios 1 e 2) | 0,483 | −0,020 (2/5) | −0,043 (0/5) | 0,496 |

**VMamba-T, CT** (referência 0,350; ALP 0,239)

| Fator | SAM | Δ | ΔALP | 3D |
|---|---|---|---|---|
| Treino `ct_sam_d4_penultima`, passo 250 | 0,365 | +0,016 (3/5) | +0,038 (4/5) | 0,383 |
| Treino `ct_sam_d4_penultima`, passo 500 | 0,330 | −0,019 (3/5) | +0,021 (4/5) | 0,355 |
| Grade 42 | 0,325 | −0,024 (1/5) | −0,012 (0/5) | 0,307 |
| 768 px, grade 48 | 0,307 | −0,043 (0/5) | −0,026 (0/5) | 0,280 |
| **Posição 2D (0,20)** | **0,511** | **+0,161 (5/5)** | **+0,091 (5/5)** | 0,499 |
| Multi-escala (estágios 1 e 2) | 0,242 | −0,107 (1/5) | −0,062 (1/5) | 0,197 |

**DINOv2-L, RM** (referência 0,747): `imagenet` 0,720 (−0,027; 1/5), `zscore_corpo` 0,736 (−0,011; 1/5),
`imagenet_corpo` 0,727 (−0,020; 1/5).

**Entrada limpa em CT não foi medida**: nos pacientes 7 a 14 (e 15 e 20), o volume já está em 0–255 e os
percentis 0,5 e 99,5 são exatamente 0 e 255. O corte por percentil não altera nada; `imagenet_fixo` é
idêntico a `imagenet`. Por isso foram 6 avaliações em CT, não 13.

## Adendo (16/11/2026, escrito antes de rodar): controle da posição 2D no DINOv2-L

A posição 2D não vem do encoder: são 4 canais calculados da caixa do corpo. Para saber se o ganho é do
VMamba-T ou do pipeline, mede-se o mesmo fator no DINOv2-L congelado (`--modelo dinov2_l14 --pos-can 0.20`,
protocolo oficial, 5 suportes, RM e CT). 2 avaliações além do máximo original da ficha. Leitura: se o
DINOv2-L subir pelo menos 0,02 em 4 de 5 suportes, a posição 2D é mudança de pipeline e vale para os dois
modelos (a comparação VMamba × DINOv2 passa a ser feita com ela nos dois); se não subir, fica registrada
como ganho do VMamba-T.

Resultado do adendo: a preencher.

## Decisão

- **Posição 2D canônica (0,20)**: cumpre a regra nas duas modalidades e entra na base do Exp. 12e. O
  VMamba-T congelado passa de 0,504 para 0,728 em RM e de 0,350 para 0,511 em CT.
- **Entrada limpa, grade 42, 768 px com grade 48 e multi-escala**: nenhum cumpre a regra; ficam fora. A
  grade maior baixa o ALP em todos os suportes e piora o CT. A entrada de RM continua `zscore` (o DINOv2-L
  piora com as três alternativas).
- **Treino `ct_sam_d4_penultima`**: não resgatado (+0,016 e −0,019 no SAM; a regra pede +0,03 nos dois
  passos). O ALP sobe nos dois passos (+0,038 e +0,021, 4 de 5 suportes), mas o ganho não chega ao SAM.
- **Linha combinada**: só um fator cumpriu a regra, então a linha combinada é a própria posição 2D e não
  foi rodada.
- **Base do Exp. 12e**: VMamba-T, penúltima camada, `sem_norm`, 672 px, grade 32, `--pos-can 0.20`,
  entrada `zscore` em RM e `imagenet` em CT.
