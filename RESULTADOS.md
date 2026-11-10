# Resultados ProtoSAM-repro — Resumo dos Experimentos

> Gerado em 2026-11-10 (revisado após auditoria dos datasets). Objetivo: explicar
> de forma acessível **o que foi testado**, **como ler os números** e **os
> melhores resultados até aqui**.

> **Correção importante desta revisão:** a maior parte da ablação (ajuste fino,
> passos, camada, resolução) foi feita em **PÓLIPOS** (datasets `polyp_raw` e
> Kvasir), e **não** em TC abdominal como uma versão anterior deste documento
> sugeria. Isso foi confirmado lendo os caminhos reais das imagens em cada
> rodada (ex.: `polyp_raw/TrainDataset/.../111.png`).

---

## 1. O problema, em uma frase

Segmentar estruturas em imagens médicas usando **pouquíssimos exemplos
rotulados** (no limite, **um único** — *one-shot*, "K1"). Os experimentos cobrem
dois domínios:
- **Pólipos** (colonoscopia): datasets `polyp_raw` (estilo PraNet, *TrainDataset*)
  e **Kvasir** (*TestDataset*). **É o foco principal.**
- **Abdome** (TC e RM): **CHAOS** (RM) e um conjunto CT — alvo: **rins**.
  Conjunto menor de experimentos, mais recente.

---

## 2. Como o modelo funciona (ProtoSAM)

Pipeline em **3 blocos**:
1. **Encoder** (extrai características). Baseline: **DINOv2-Large**. Alternativa
   testada: **VMamba** (*tiny*, *small*, *base*).
2. **ALPNet (protótipos)**: a partir do exemplo de suporte, gera a **primeira
   máscara**. -> métrica **`Dice_ALP`**.
3. **SAM-Huge (refinamento)**: refina a máscara do ALP. -> **`Dice_SAM`** (final).

**Dice** = sobreposição com a máscara correta (0% a 100%). **O SAM quase sempre
melhora o ALP** (ex.: 53,8% -> 62,5%).

---

## 3. Vocabulário rápido

| Termo | Significado |
|---|---|
| **K1** | *One-shot*: 1 exemplo de suporte |
| **ALP / SAM** | Dice dos protótipos / Dice após refino do SAM |
| **EFT** | *Encoder Fine-Tuning*: ajustar o encoder (auto-supervisão por superpixels) |
| **antes / depois** | modelo **congelado** (pré-EFT) vs **após** o EFT |
| **última / penúltima** | de qual camada do encoder se leem as características |
| **passo** | nº de passos de treino do EFT (checkpoint avaliado) |
| **polyp_raw / Kvasir** | datasets de pólipos | 
| **CHAOS** | dataset de RM abdominal (rins) |

---

## 4. Linha do tempo dos experimentos

| Notebook | Dataset | Pergunta |
|---|---|---|
| `00/01_baseline` | Pólipos (Kvasir) | Baseline **DINOv2-L + ALP + SAM-H**, K1 |
| `02_vmamba_congelado_k1` | Pólipos | Trocar encoder por **VMamba congelado** |
| `03_vmamba_small_base_k1` | Pólipos | VMamba **Small/Base** congelados (672 px) |
| `04/05_eft...` | Pólipos | Montar e rodar o **EFT** (VMamba *tiny*) |
| `12_camadas_normalizacao` | Pólipos | Melhor **camada + normalização** |
| `13/14_eft_penultima` | Pólipos | Ler/treinar pela **penúltima camada** |
| `15/16_multissuporte` | Pólipos (Kvasir) | Efeito de **múltiplos suportes** (K>1) |
| `17_eft_tiny_512` | Pólipos | **EFT512** (penúltima @512 px) |
| `18_eft_sam_pseudolabels` | Pólipos | Pseudo-rótulos do SAM |
| `19_eft_ct_mri` | TC/RM | Transferência **TC -> RM** |
| `20_eft_mri_chaos` | RM (CHAOS) | **EFT** em RM |
| `21_baseline_dino_mri` | RM (CHAOS) | Baseline DINOv2 em RM |
| `22_vmamba_672_mri` | RM (CHAOS) | VMamba 672 px em RM |

---

## 5. Melhores resultados até aqui (por domínio)

| Domínio | Melhor configuração | **Dice_SAM** |
|---|---|---:|
| **Pólipos (Kvasir)** | **DINOv2-L (multi-suporte)** | **74,6** |
| **Pólipos (polyp_raw)** | EFT VMamba *tiny*, passo 2000, última camada | **62,5** |
| RM abdominal (CHAOS) — rim dir. | DINOv2-L @672 px (congelado) | 60,7 |
| TC abdominal — rim | EFT (CT felzenszwalb) | ~31 |

> **O melhor Dice absoluto é em pólipos** (DINOv2 74,6% no Kvasir; EFT-VMamba
> *tiny* 62,5% no polyp_raw). Em abdome os resultados são bem mais baixos.

### 5a. Pólipos (polyp_raw) — EFT vs congelado + camada/normalização
*(VMamba tiny; "antes" = congelado, "depois" = após EFT; n=145 imagens)*

| Modelo | Camada | Normalização | Dice_SAM |
|---|---|---|---:|
| **EFT (2000)** | **última** | **do_checkpoint** | **62,5** |
| EFT (2000) | penúltima | sem_norm | 57,6 |
| EFT (2000) | penúltima | ln_unitaria | 57,0 |
| EFT (2000) | última | ln_unitaria | 51,9 |
| congelado | — | — | 48,1 |
| EFT (2000) | última | sem_norm | 31,7 |

### 5b. Pólipos — efeito do nº de passos (EFT tiny)

| Passo | Dice_SAM |
|---:|---:|
| 0 (congelado) | 48,1 |
| 1000 | 49,6 |
| 1500 | 58,4 |
| **2000** | **62,5** |
| 3000 | 61,3 |
| 5000 | 56,6 |

Ótimo em **~2000 passos**; treinar mais **piora** (overfitting).

---

## 6. Principais lições

1. **Em pólipos, DINOv2 é o backbone mais forte** (74,6% com multi-suporte),
   acima do VMamba tiny com EFT (62,5%).
2. **EFT ajuda o VMamba *tiny*** (48,1 -> 62,5), mas **piorou o *small*/*base***
   em pólipos (ver seção 7b).
3. **Mais treino não é melhor:** ótimo ~2000 passos.
4. **O refino do SAM agrega ~+9 pontos** sobre o ALP.
5. **Camada + normalização importam:** última + `do_checkpoint` é a melhor;
   última + `sem_norm` é a pior (31,7).
6. **A receita de pólipos NÃO transfere para abdome** (ver seção 7).

---

## 7. Abdome (RM/CHAOS e TC) e multi-suporte

### 7a. RM (CHAOS) — rins, fold 0 (média por paciente)

| Config | Rim D. | Rim E. |
|---|---:|---:|
| DINOv2-L @256 (congelado) | 44,8 | 47,4 |
| **DINOv2-L @672 (congelado)** | **60,7** | 48,8 |
| VMamba *tiny* + EFT (10k passos) | 28,5 | 34,4 |
| VMamba + EFT (CT felzenszwalb) | 30,9 | 27,0 |

**Resolução 672 ajuda muito** o DINOv2 (44,8 -> 60,7). **EFT-VMamba fica bem
abaixo** (28-34%): a receita que venceu em pólipos **não transfere** para RM.

### 7b. Pólipos — comparação de backbones

| Backbone | Dice_SAM |
|---|---:|
| **DINOv2-L (Kvasir, multi-suporte)** | **74,6** / 70,7 |
| VMamba *tiny* + EFT (polyp_raw) | 62,5 |
| VMamba *small* + EFT | 41,2 *(piorou vs. 48,7 congelado)* |
| VMamba *base* + EFT | 39,9 *(piorou vs. 44,6 congelado)* |

### 7c. Sensibilidade ao suporte (VMamba tiny, Kvasir)
Trocando o exemplo de suporte (18 casos), Dice_SAM oscilou de **~21% a ~62%**.
**One-shot é instável:** a escolha do suporte pesa muito.

---

## 8. Nota sobre a integridade dos dados (investigada)

Numa primeira leitura, vários `resumo.json` compartilhavam o mesmo valor "antes"
(ex.: 48,1), o que parecia erro. **Investiguei e confirmei que NÃO é erro:**
- Recalculei a média direto dos 145 registros por imagem: bate exatamente
  (`0,4808002661`).
- As imagens são pólipos reais (`polyp_raw/TrainDataset/.../111.png`), e "antes"
  e "depois" usam as mesmas queries (há `assert` de SHA no código).
- O valor se repete porque é **o mesmo modelo tiny congelado avaliado no mesmo
  conjunto de pólipos** — logo o baseline é idêntico entre esses experimentos.
  Modelos diferentes (base/small) têm "antes" diferentes (44,6 / 48,7), como
  esperado.

*Metodologia: Dice extraído dos `registros/*.json` (média por imagem, pólipos) e
dos `resumo_geral.csv` (média por paciente, RM/CT). Scripts em `_claude_tools/`.*
