# Resultados ProtoSAM-repro — Resumo dos Experimentos

> Gerado em 2026-11-10. Objetivo: explicar, de forma acessível (mesmo para quem
> não conhece o projeto), **o que foi testado**, **como ler os números** e
> **quais os melhores resultados até aqui**.

---

## 1. O problema, em uma frase

Segmentar estruturas em imagens médicas — órgãos abdominais em **Tomografia (TC)**
e **Ressonância (RM)**, e **pólipos** em colonoscopia — usando **pouquíssimos
exemplos rotulados**. No caso limite, **um único exemplo** por classe
(*one-shot*, chamado aqui de **K1**).

Por que isso importa: rotular imagem médica é caro e exige especialista. Um método
que aprende a segmentar vendo 1 exemplo economiza muito trabalho.

---

## 2. Como o modelo funciona (ProtoSAM)

O pipeline tem **3 blocos** em sequência:

1. **Encoder** (extrai características da imagem). Baseline: **DINOv2-Large**.
   Alternativa testada no projeto: **VMamba** (nos tamanhos *tiny*, *small*, *base*).
2. **ALPNet (protótipos)**: a partir do exemplo de suporte, cria "protótipos" da
   classe e compara com a imagem de consulta, produzindo uma **primeira máscara**.
   -> métrica **`Dice_ALP`**.
3. **SAM-Huge (refinamento)**: usa a máscara do ALP como *prompt* e refina os
   contornos. -> métrica **`Dice_SAM`** (resultado final do pipeline).

**Dice** mede a sobreposição entre a máscara prevista e a correta:
**0% = nada em comum, 100% = perfeito**. É a métrica principal.

> **Regra observada em todos os experimentos:** o SAM quase sempre *melhora* o
> ALP. Ex.: 53,8% (ALP) -> 62,5% (SAM). O ganho do refino costuma ser ~8-10 pontos.

---

## 3. Vocabulário rápido

| Termo | Significado |
|---|---|
| **K1** | *One-shot*: 1 único exemplo de suporte por classe |
| **ALP / SAM** | Dice da etapa de protótipos / Dice após refino do SAM |
| **EFT** | *Encoder Fine-Tuning*: ajustar o encoder ao domínio médico (auto-supervisão por superpixels), em vez de usá-lo congelado |
| **última / penúltima camada** | de qual camada do encoder as características são lidas |
| **normalização** | como as características são normalizadas (`do_checkpoint`, `sem_norm`, `ln_unitaria`) |
| **passo** | nº de passos de treino do fine-tuning (checkpoint avaliado) |
| **TC=SABS / RM=CHAOS** | datasets abdominais; órgãos: rins (label 0), fígado+baço (label 1) |

---

## 4. Linha do tempo dos experimentos (a ablação)

Cada notebook em `exp/` responde a **uma pergunta**. Em ordem:

| Notebook | Pergunta investigada |
|---|---|
| `00/01_baseline` | Baseline oficial: **DINOv2-L + ALP + SAM-H**, K1 |
| `02_vmamba_congelado_k1` | E se trocar o encoder por **VMamba congelado**? |
| `03_vmamba_small_base_k1` | VMamba **Small** e **Base** congelados (672 px) |
| `04_eft_dados_e_episodios` | Montar dados/episódios para o **fine-tuning (EFT)** |
| `05_eft_vmamba_tiny` | **EFT** no VMamba *tiny* |
| `12_camadas_normalizacao` | Qual **camada** + **normalização** rende mais? |
| `13/14_eft_penultima` | Treinar/avaliar lendo a **penúltima camada** |
| `15_avaliacao_multissuporte` | E com **mais de 1 suporte** (K>1)? |
| `16_dinov2_multissuporte` | Multi-suporte no DINOv2 |
| `17_eft_tiny_512` | **EFT512**: treino direto pela penúltima camada @512 px |
| `18_eft_sam_pseudolabels` | Usar máscaras do SAM como **pseudo-rótulos** |
| `19_eft_ct_mri` | Transferência **TC -> RM** (cross-modalidade) |
| `20_eft_mri_chaos` | **EFT** em RM (CHAOS) |
| `21_baseline_dino_mri` | Baseline DINOv2 em RM |
| `22_vmamba_672_mri` | VMamba 672 px em RM |

---

## 5. Melhores resultados até aqui

> **Destaque (protocolo TC abdominal):** o melhor resultado neste cenário é **Dice_SAM = 62,5%**
> (Dice_ALP = 53,8%), obtido com **EFT no VMamba tiny, 2000 passos de treino,
> lendo a última camada, normalização "do checkpoint"**. O experimento
> **EFT512** (penúltima camada @512 px) empata nisso (~62,6% SAM).

### 5a. Ajuste fino do encoder (EFT) vs. congelado + camada/normalização
*(notebook 12 — `avaliacao_camadas_norm_tiny_2000`)*

| Encoder | Camada | Normalização | Dice_ALP | Dice_SAM |
|---|---|---|---:|---:|
| **EFT (2000)** | **última** | **do_checkpoint** | 53,8 | **62,5** |
| EFT (2000) | penúltima | sem_norm | 50,3 | 57,6 |
| EFT (2000) | penúltima | ln_unitaria | 50,1 | 57,0 |
| EFT (2000) | última | ln_unitaria | 47,5 | 51,9 |
| pré-treinado | penúltima | ln_unitaria | 45,0 | 51,6 |
| pré-treinado | penúltima | sem_norm | 44,9 | 50,9 |
| pré-treinado | última | do_checkpoint | 39,9 | 48,1 |
| EFT (2000) | última | sem_norm | 35,6 | 31,7 |

Leitura: **ajustar o encoder (EFT) supera usá-lo congelado** (62,5 vs 48,1 na
mesma config). Mas a combinação importa: **última camada + do_checkpoint** é a
melhor; **última camada sem normalização** quebra o modelo (31,7).

### 5b. Quanto treinar? (efeito do nº de passos)
*(EFT tiny — `avaliacao_marcos_tiny_5000`)*

| Passo | Dice_ALP | Dice_SAM |
|---:|---:|---:|
| 0 (sem EFT) | 39,9 | 48,1 |
| 200 | 44,9 | 56,5 |
| 1000 | 45,2 | 49,6 |
| 1500 | 50,0 | 58,4 |
| **2000** | **53,8** | **62,5** |
| 3000 | 53,3 | 61,3 |
| 4000 | 48,3 | 55,5 |
| 5000 | 51,0 | 56,6 |

Leitura: existe um **ponto ótimo por volta de 2000 passos**. Treinar mais
(3000-5000) **piora** — sinal de *overfitting*.

---

## 6. Principais lições (até aqui)

1. **Fine-tuning do encoder (EFT) é o maior ganho:** +14 pontos de Dice_SAM
   sobre o encoder congelado (48,1 -> 62,5).
2. **Mais treino não é melhor:** o ótimo fica ~2000 passos; além disso, cai.
3. **O refino do SAM agrega ~+9 pontos** sobre a máscara do ALP, de forma
   consistente.
4. **Camada + normalização têm efeito grande:** última camada + `do_checkpoint`
   é a melhor receita; `sem_norm` na última camada é a pior.
5. **Resolução 512 alcança o mesmo topo (~62,6%)** com menos custo que 672.

---

## 7. Ressalvas e próximos passos

**Escopo dos números:** as tabelas acima são do cenário **TC abdominal, VMamba
tiny**. Os experimentos de **RM (CHAOS), TC->RM, multi-suporte (K>1),
pseudo-rótulos e pólipos** (notebooks 15, 16, 18, 19-22) ainda estão em forma
**bruta** (sem tabela consolidada `comparacao`), por isso seus Dice **não** foram
incluídos aqui — comparar datasets/resoluções diferentes seria enganoso.

**Onde ficam os dados:** ~83 mil JSONs brutos em `reproducao/` (não versionados).
As comparações confiáveis são os arquivos `reproducao/avaliacao_*.json`.

**Sugestões:**
- Consolidar RM/pólipos no mesmo formato `comparacao` para comparação justa.
- Comparar VMamba **small/base + EFT** contra o **tiny** (hoje só há tiny com EFT).
- Fechar a avaliação **multi-suporte (K>1)**.

---
*Metodologia: Dice médio sobre as consultas, K1. Números extraídos de
`reproducao/avaliacao_*.json` via scripts em `_claude_tools/` (fora do versionamento).*

---

## 8. Consolidação por modalidade — RM, TC (rins), pólipos e multi-suporte

Esta seção reúne os experimentos que antes estavam só em forma bruta (notebooks
15, 16, 18, 19-22). Métrica = **Dice (%)**, 1-shot salvo indicado. **Atenção:**
datasets diferentes **não** são comparáveis diretamente entre si.

### 8a. Melhor resultado por dataset

| Dataset | Alvo | Melhor configuração | **Dice_SAM** |
|---|---|---|---:|
| **Pólipos (Kvasir)** | pólipo | **DINOv2-L (multi-suporte)** | **74,6** |
| TC abdominal (SABS) | órgão | EFT VMamba *tiny*, 2000 passos | 62,5 |
| **RM abdominal (CHAOS)** | rim direito | **DINOv2-L @672 px (congelado)** | **60,7** |

> **O melhor Dice absoluto até agora é em pólipos com DINOv2 (74,6%)** — não no
> protocolo TC. Cada domínio tem seu campeão (ver abaixo).

### 8b. Ressonância (CHAOS) — rins, fold 0

| Backbone / config | Rim D. (RK) | Rim E. (LK) |
|---|---:|---:|
| DINOv2-L @256 px (congelado) | 44,8 | 47,4 |
| **DINOv2-L @672 px (congelado)** | **60,7** | 48,8 |
| VMamba *tiny* + EFT (CHAOS, 10k passos) | 28,5 | 34,4 |
| VMamba + EFT (CT felzenszwalb, fold 0) | 30,9 | 27,0 |

**Leitura:** em RM, aumentar a **resolução para 672 px** dá um salto no DINOv2
(44,8 -> 60,7 no rim direito). Já o **EFT no VMamba fica bem abaixo** (28-34%): a
receita que venceu no TC **não transfere** para a RM.

### 8c. Pólipos (Kvasir)

| Backbone | Dice_SAM |
|---|---:|
| **DINOv2-L (multi-suporte, melhor caso)** | **74,6** |
| DINOv2-L (outro suporte) | 70,7 |
| VMamba *tiny* + EFT | ~56,5 |
| VMamba *small* + EFT | 41,2 *(piorou vs. 48,7 congelado)* |
| VMamba *base* + EFT | 39,9 *(piorou vs. 44,6 congelado)* |

**Leitura:** **DINOv2 domina em pólipos.** O EFT só ajuda o VMamba *tiny*; nos
modelos *small*/*base* o fine-tuning **piorou** o resultado.

### 8d. Sensibilidade ao suporte (multi-suporte, VMamba *tiny*, Kvasir)

Variando qual exemplo de suporte é usado (18 casos), o Dice_SAM oscilou de
**~21% a ~62%**. **Conclusão:** em one-shot, **a escolha do exemplo de suporte
tem impacto enorme** na qualidade — resultado instável por natureza.

### Ressalvas de dados (ler antes de citar números)

1. **Baseline "antes" suspeito:** vários `resumo.json` compartilham exatamente o
   mesmo valor "antes" (48,1 / 51,6), igual ao baseline do TC. Indício de que o
   "antes" foi **herdado do protocolo TC**, não recomputado no dataset alvo.
   Confie no **"depois"** (modelo avaliado); trate os *deltas* antes->depois de
   pólipos com cautela e re-valide.
2. **Agregações diferentes:** RM/CT usam média **por paciente**
   (`resumo_geral.csv`); pólipos usam média **por imagem** (`resumo.json`).
3. Números extraídos dos arquivos `resumo_geral.csv` e `resumo.json` de cada
   rodada (scripts em `_claude_tools/`).
