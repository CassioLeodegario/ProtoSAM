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

## Resultado

A preencher.

## Decisão

A preencher.
