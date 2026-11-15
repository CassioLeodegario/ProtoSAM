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

Medido em 14-15/11/2026 (`runs/E12b_diag.log`), modelos pré-treinados, 3 suportes, todas as fatias com rim
(RM 367 fatias, CT 1169). Dice SAM; a primeira coluna é o subconjunto de 8 fatias por paciente, que
reproduz o protocolo do Exp. 12.

| Modelo, modalidade | Exp. 12 (8 fatias, média por paciente) | todas as fatias | + CCA | + suporte por terça parte (oficial) |
|---|---|---|---|---|
| DINOv2-L, RM | 0,553 | 0,574 | 0,646 | 0,738 |
| DINOv2-L, CT | 0,367 | 0,414 | 0,483 | 0,568 |
| VMamba-T, RM | 0,364 | 0,377 | 0,412 | 0,507 |
| VMamba-T, CT | 0,209 | 0,241 | 0,269 | 0,337 |

- Teto da etapa do SAM (prompts tirados da máscara verdadeira): 0,927 em RM e 0,865 em CT. O erro está
  na máscara do ALP, não no SAM.
- Por terça parte do rim (DINOv2-L, protocolo oficial): RM 0,691 / 0,819 / 0,714; CT 0,503 / 0,616 / 0,587.
- Por área do rim na fatia (DINOv2-L, oficial; <100, 100-300, >=300 px): RM 0,000 / 0,304 / 0,785;
  CT 0,091 / 0,284 / 0,659. As fatias com rim pequeno continuam quase sem acerto em qualquer variante.
- Sem CCA a área prevista é 1,7 a 2,0 vezes a do rim (DINOv2-L) e 2,4 a 4,1 vezes (VMamba-T); com CCA e
  suporte por terça parte cai para 1,3 (DINOv2-L) e 1,5 a 2,6 (VMamba-T).
- Outro rim: sem CCA, 9% a 11% da área prevista pelo DINOv2-L cai no outro rim (13% a 21% no VMamba-T);
  com CCA, 0% a 1% (DINOv2-L) e 6% a 12% (VMamba-T). A CCA aumenta as fatias com erro total
  (DINOv2-L: 5% para 11% em RM, 9% para 22% em CT; VMamba-T: até 44%).

## Decisão

As três diferenças de protocolo explicam +0,185 (RM) e +0,201 (CT) de Dice SAM no DINOv2-L: suporte por
terça parte (+0,09), CCA (+0,07) e medir todas as fatias (+0,02 a +0,05). A confusão com o outro rim é
secundária. O Dice baixo do Exp. 12 vinha do protocolo, não do DINOv2. Em 15/11 o Cássio decidiu usar o
protocolo oficial; a remedida fica no Exp. 12c. O que ainda separa estes números do artigo não foi medido
aqui (pacientes de suporte e de query diferentes, normalização da entrada em CT, fatias com rim pequeno).
