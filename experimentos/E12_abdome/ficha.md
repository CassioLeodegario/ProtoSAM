# Exp. 12 — Existe uma receita em que o ajuste auto-supervisionado melhora o VMamba em CT e RM?

Escrita em 12/11/2026, antes de qualquer execução. Antecipa o Exp. 12 do plano (semana 4).

- **Pergunta**: em rins (RM CHAOS-T2 e CT SABS), o VMamba-Tiny ajustado com pseudo-máscaras
  (superpixels de Felzenszwalb e regiões do SAM) supera o mesmo VMamba pré-treinado?
- **Hipótese que testa**: apoio à H1 fora de pólipos.
- **Ponto de partida** (out/2026, `ProtoSAM-repro`): ajuste com superpixels, 256 px, última camada,
  um único suporte (paciente 4) e 4 pacientes de query (RM) ou 6 (CT). Sem ganho confiável: o Dice
  oscilava entre checkpoints mais do que a diferença para o modelo congelado.
- **O que muda no protocolo** (`protovm/abdome.py`), para a medida deixar de ser ruído:
  - os pacientes de teste do fold 0 oficial e o suporte antigo ficam reservados (RM 0–4, CT 0–6);
  - validação de desenvolvimento: RM pacientes 5–9, CT 7–14; ajuste: RM 10–19, CT 15–29;
  - 5 pacientes-suporte por modalidade (RM 5–9, CT 7–11), cada um avaliado nos outros pacientes
    da validação; suporte = fatia central do rim; até 8 fatias com o rim por paciente de query;
  - fatias a 672 no encoder, SAM-H com pontos + caixa, como em pólipos.
- **O que é comparado** (uma coisa por vez; `scripts/abdome.py`):
  - **A. Sem treino** (CCA desligada), em RM e CT: VMamba-T pré-treinado lido na última camada e na
    penúltima; penúltima com a outra normalização de entrada (`zscore` × `imagenet`); DINOv2-L
    pré-treinado nas duas entradas (referência). 10 avaliações.
  - **B. Receita de pólipos** (672 px, encoder inteiro, lr 1e-5, wd 0,01, query espelhada, 4.000
    episódios, fatias com rim fora do ajuste): `configs/abdome/{mr,ct}_{sp,sam}.yaml`.
    Passos 500, 1000, 2000 e 4000, lidos na penúltima camada. 4 treinos, 16 avaliações.
  - **C. Query deformada** (`aug: afim`: rotação ±20°, escala 0,85–1,15, translação ±10%, espelho
    e gama 0,7–1,5): `configs/abdome/*_afim.yaml`. Motivo: em fatias quase simétricas o espelho
    pode ser uma tarefa fácil demais; o ALPNet original usa deformações. 4 treinos, 16 avaliações.
  - **D. Segunda rodada, só se B e C não cumprirem a regra**, um fator por vez sobre a melhor de B/C:
    lr 3e-6; fatias com rim dentro do ajuste (sem rótulo); entrada alternativa se A mostrar ganho
    de pelo menos 0,03; leitura na última camada; CCA ligada. No máximo 6 treinos e 30 avaliações.
- **O que fica fixo**: semente 1234; mesmas fatias e mesmos suportes em todas as linhas; as regiões
  do SAM usam os parâmetros do gerador de pólipos e o mesmo filtro de área dos superpixels
  (entre 0,5% e 80% do corpo, 95% dentro do corpo).
- **Métrica**: Dice do ALP e do SAM, média dos dois rins, média e desvio nos 5 suportes; diferença
  pareada por suporte contra o VMamba pré-treinado lido do mesmo jeito (`scripts/abdome.py resumo`).
- **Regra de decisão**: digo que a receita **melhora** uma modalidade com um tipo de pseudo-máscara se
  o Dice SAM médio subir pelo menos 0,03 sobre o pré-treinado, ganhar em pelo menos 4 dos 5 suportes
  e isso valer em **dois passos consecutivos** do mesmo treino (para não escolher um pico isolado).
  O passo reportado é o primeiro dos dois. Se nada cumprir a regra depois de D, o resultado do
  abdome é negativo e é reportado assim. Nenhum paciente reservado é lido em nenhum caso.
- **Máximo de execuções**: A+B+C: 8 treinos e 42 avaliações (cerca de 10 h de GPU). D: mais 6 treinos
  e 30 avaliações.
- **Limites conhecidos**: 5 pacientes de validação em RM e 8 em CT; só rins; um fold; uma semente.
- **Adendo de 13/11/2026** (escrito antes de rodar a rodada D; A, B e a parte de RM de C já lidas, os
  dois treinos de CT de C ainda rodando). O que foi visto: nenhuma linha cumpre a regra; o pico vem
  no passo 500 e o Dice cai com mais treino, nas duas modalidades e nos dois tipos de pseudo-máscara;
  a mais próxima é RM/Felzenszwalb/query deformada no passo 500 (+0,029, 5 de 5 suportes; +0,008 no
  passo 1000). A entrada alternativa não deu 0,03 em A e sai da rodada. A pedido, a rodada D fica
  maior que o previsto e vira uma busca de hiperparâmetros, sempre com query deformada, nos quatro
  pares (RM, CT) × (Felzenszwalb, regiões do SAM), um fator por vez sobre a receita de C:
  - **D1** lr 3e-6, 2.000 episódios, passos 250, 500, 1000 e 2000 (`*_d1_lr3e6.yaml`);
  - **D2** lr 1e-5 lido antes do pico: passos 100 e 250 (`*_d2_curto.yaml`; o passo 500 é o de C);
  - **D3** fatias com rim dentro do ajuste (sem rótulo), passos 250, 500 e 1000 (`*_d3_comrim.yaml`);
  - **D4** treino direto pela penúltima camada, passos 250, 500 e 1000 (`*_d4_penultima.yaml`).
  Tudo lido na penúltima camada, CCA desligada. 16 treinos e 48 avaliações (cerca de 7 h de GPU).
  A regra de decisão não muda; "dois passos consecutivos" vale na grade de passos de cada treino, e
  em D2 a sequência é 100, 250 e o 500 de C. Com 16 treinos a mais, um par que cumpra a regra por
  pouco é tratado como candidato e precisa ser repetido com outra semente antes de ser reportado.
  Leitura na última camada e CCA ficam para depois, só sobre a melhor linha de cada modalidade.
- **Segundo adendo de 13/11/2026** (escrito antes de rodar e antes de ler a rodada D). A pedido, as
  ablações feitas com o VMamba-Tiny em pólipos são repetidas no abdome, e entram o VMamba-Small e o
  VMamba-Base. Um fator por vez sobre a receita de C (query deformada, lr 1e-5, 672, wd 0,01), nos
  quatro pares (RM, CT) × (Felzenszwalb, regiões do SAM), passos 250, 500 e 1000, penúltima camada:
  - **E1–E3** resolução de treino 256, 512 e 1024 (`*_e1_res256`, `*_e2_res512`, `*_e3_res1024`);
    a avaliação continua a 672;
  - **E4** weight decay 0 (`*_e4_wd0`); **E5** `patch_embed` e estágios 0–1 congelados (`*_e5_congelar`);
  - **E6** VMamba-Small e **E7** VMamba-Base (`*_e6_small`, `*_e7_base`), comparados com o próprio
    Small/Base pré-treinado.
  - **F, sem treino novo**, em RM e CT: Small e Base pré-treinados (última e penúltima camada); Tiny
    pré-treinado com CCA e com a máscara de maior score do SAM; e os quatro treinos de C no passo 500
    lidos na última camada, com CCA e com a máscara de maior score.
  Já cobertas: lr 3e-6 (D1), treino pela penúltima (D4), passos (B, C, D), Felzenszwalb × regiões do
  SAM (todas). Não repetida: superpixels calculados a 512, porque os superpixels do abdome são os
  arquivos prontos do conjunto de dados (256 px). 28 treinos e 108 avaliações (cerca de 16 h de GPU).
  A regra de decisão é a mesma; CCA e máscara de maior score são comparadas com o pré-treinado lido
  do mesmo jeito. Com tantas linhas, qualquer par que cumpra a regra é candidato e precisa ser
  repetido com outra semente antes de ser reportado.
- **Terceiro adendo de 14/11/2026** (escrito depois de ler A–F e antes de rodar a repetição). Um único
  par cumpre a regra: RM, regiões do SAM, VMamba-Base (E7). Como previsto nos adendos, ele é repetido
  com outra semente (`configs/abdome/mr_sam_e7_base_s2.yaml`, semente 4321), mesmos passos e mesma
  leitura. 1 treino e 3 avaliações. O par só é reportado como ganho se a segunda semente também
  cumprir a regra.

## Resultado

Lido em 14/11/2026, de `resultados/abdome.csv` (`scripts/abdome.py resumo`). Dice SAM, média nos 5
suportes, CCA desligada e penúltima camada, salvo indicação. Δ é a diferença pareada contra o
pré-treinado lido do mesmo jeito; entre parênteses, suportes em que ganha.

**A. Sem treino.**

| Modelo | RM | CT |
|---|---|---|
| VMamba-T, última camada | 0,347 | 0,142 |
| VMamba-T, penúltima (referência) | 0,381 | 0,215 |
| VMamba-T, penúltima, outra entrada | 0,390 | 0,199 |
| VMamba-S, penúltima (última) | 0,388 (0,323) | 0,204 (0,155) |
| VMamba-B, penúltima (última) | 0,261 (0,263) | 0,176 (0,150) |
| VMamba-T, penúltima, CCA | 0,418 | 0,261 |
| VMamba-T, penúltima, máscara de maior score | 0,364 | 0,185 |
| DINOv2-L | 0,584 | 0,395 |

**B–E. VMamba-T ajustado: melhor passo de cada variação (Δ).**

| Variação | RM Felz. | RM SAM | CT Felz. | CT SAM |
|---|---|---|---|---|
| B query espelhada | −0,007 (1/5) | +0,010 (4/5) | −0,004 (3/5) | +0,016 (3/5) |
| C query deformada | **+0,029 (5/5)** | +0,002 (3/5) | −0,039 (0/5) | −0,005 (3/5) |
| D1 lr 3e-6 | +0,003 (4/5) | +0,001 (2/5) | −0,009 (2/5) | +0,008 (3/5) |
| D2 passos 100 e 250 | −0,014 (1/5) | +0,007 (3/5) | −0,022 (1/5) | −0,001 (1/5) |
| D3 fatias com rim no ajuste | +0,008 (3/5) | −0,001 (2/5) | +0,015 (4/5) | +0,003 (3/5) |
| D4 treino pela penúltima | +0,008 (4/5) | −0,039 (0/5) | +0,006 (4/5) | +0,018 (4/5) |
| E1 treino a 256 | +0,024 (5/5) | +0,012 (2/5) | +0,001 (3/5) | −0,016 (2/5) |
| E2 treino a 512 | −0,016 (1/5) | +0,002 (2/5) | −0,002 (4/5) | +0,011 (3/5) |
| E3 treino a 1024 | +0,006 (3/5) | −0,004 (3/5) | +0,010 (3/5) | +0,002 (2/5) |
| E4 weight decay 0 | −0,004 (4/5) | −0,002 (3/5) | −0,018 (2/5) | −0,003 (3/5) |
| E5 estágios iniciais congelados | +0,010 (4/5) | −0,003 (2/5) | −0,016 (2/5) | +0,004 (3/5) |

Nenhuma linha do VMamba-T cumpre a regra. A mais próxima é RM/Felzenszwalb/query deformada no passo
500 (0,410; +0,029, 5/5), que cai para +0,008 no passo 1000; o treino a 256 repete o padrão (+0,024
no passo 500, −0,033 no 1000). O ganho, quando existe, aparece até o passo 500 e some com mais treino.

**E6–E7. Small e Base ajustados (Δ contra o próprio pré-treinado; passos 250 / 500 / 1000).**

| Modelo | RM Felz. | RM SAM | CT Felz. | CT SAM |
|---|---|---|---|---|
| Small | −0,093 / −0,068 / −0,029 | −0,060 / −0,048 / −0,081 | −0,105 / −0,099 / −0,031 | +0,007 / −0,006 / −0,031 |
| Base | −0,096 / −0,055 / +0,015 | **+0,089 / +0,087 / +0,082 (4/5)** | −0,084 / −0,075 / −0,068 | −0,062 / −0,033 / −0,007 |

RM/regiões do SAM/Base cumpre a regra nos três passos (0,351 no passo 250). Ressalva: o Base
pré-treinado é muito fraco em RM (0,261), e o Base ajustado continua abaixo do Tiny pré-treinado
(0,381). É uma recuperação de um ponto de partida ruim, não um resultado melhor em valor absoluto.

**F. Treinos de C no passo 500 lidos de outros jeitos (Δ contra o pré-treinado lido igual).**

| Leitura | RM Felz. | RM SAM | CT Felz. | CT SAM |
|---|---|---|---|---|
| Última camada | +0,028 (3/5) | +0,025 (3/5) | −0,014 (3/5) | +0,031 (4/5) |
| CCA | +0,017 (3/5); 0,436 | +0,013 (4/5); 0,432 | −0,019 (1/5); 0,241 | +0,029 (4/5); 0,290 |
| Máscara de maior score | +0,036 (5/5); 0,400 | +0,022 (4/5); 0,386 | −0,031 (1/5); 0,154 | +0,011 (3/5); 0,197 |

São leituras de um único passo, então não podem cumprir a regra de dois passos. A CCA dá os maiores
valores absolutos do VMamba-T (RM 0,436; CT 0,290), mas a maior parte vem da própria CCA (pré-treinado
com CCA: 0,418 e 0,261). A máscara de maior score piora o valor absoluto em todas as linhas.

## Decisão

Pela regra, o resultado do abdome para o VMamba-Tiny é **negativo** nas duas modalidades e nos dois
tipos de pseudo-máscara: em 44 treinos, nenhum passo com +0,03 em 4 de 5 suportes por dois passos
seguidos. O único par que cumpre a regra é RM/regiões do SAM com o VMamba-Base, tratado como
candidato até a repetição com a segunda semente (terceiro adendo), e com a ressalva de que fica abaixo
do Tiny sem treino. Resultado da segunda semente: a preencher.
