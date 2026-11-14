# Exp. 12b — Por que o Dice do abdome é baixo até para o DINOv2-L?

Escrita em 14/11/2026, antes de rodar.

- **Pergunta**: no protocolo do Exp. 12 o DINOv2-L pré-treinado dá Dice SAM 0,584 em RM e 0,395 em CT
  (rins), bem abaixo do que o artigo relata. O que no protocolo causa isso?
- **Hipótese que testa**: nenhuma da dissertação; é uma verificação do protocolo.
- **Diferenças já lidas no código oficial** (`validation_protosam.py`, `run_protosam.sh`,
  `dataloaders/`), ainda não medidas:
  1. o oficial liga a CCA no abdome; o Exp. 12 mede com CCA desligada, e sem CCA todos os componentes
     da máscara do ALP viram prompts e as máscaras do SAM são somadas;
  2. o oficial usa 3 fatias de suporte (1/6, 1/2 e 5/6 do órgão) e cada fatia de query usa a da sua
     terça parte; o Exp. 12 usa só a fatia central;
  3. o oficial mede todas as fatias com o órgão; o Exp. 12 mede até 8 igualmente espaçadas, o que dá
     mais peso às pontas do rim;
  4. o rim direito e o esquerdo são classes separadas com aparência igual: a predição pode cair no
     outro rim.
- **O que é medido** (`scripts/abdome_diag.py`, sem treino, sem registro; saída em `rascunho/E12b_diag/`):
  DINOv2-L pré-treinado, em RM e CT, 3 primeiros suportes, todas as fatias com rim. Por fatia: suporte
  central × suporte por terça parte; máscara do ALP, SAM sem CCA e SAM com CCA; e um teto da etapa do
  SAM (prompts tirados da máscara verdadeira). Grava área prevista, acerto no rim-alvo e sobreposição
  com o outro rim. Depois, o mesmo para o VMamba-T pré-treinado (penúltima camada).
- **Leituras**: Dice por fatia, Dice no subconjunto de 8 fatias (deve reproduzir o registro), Dice por
  volume, por terça parte do rim, por área do rim, fração de fatias com erro total e fração da área
  prevista que cai no outro rim.
- **Regra de decisão**: nenhuma sobre a dissertação. Se uma das quatro diferenças explicar a maior parte
  da queda, proponho a mudança de protocolo ao Cássio antes de qualquer nova medida de EFT; o protocolo
  do Exp. 12 não é alterado aqui.
- **Máximo de execuções**: 4 medições (2 modelos × 2 modalidades), cerca de 3 h de GPU. Pacientes
  reservados não são lidos.

## Resultado

A preencher.

## Decisão

A preencher.
