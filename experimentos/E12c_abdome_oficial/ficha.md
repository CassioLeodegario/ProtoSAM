# Exp. 12c — As conclusões do Exp. 12 se mantêm no protocolo oficial do abdome?

Escrita em 15/11/2026, antes de rodar. Decisão do Cássio em 15/11: usar o protocolo oficial.

- **Pergunta**: medido como o código oficial mede o abdome, o ajuste auto-supervisionado melhora o VMamba
  em rins (RM e CT)? As linhas principais do Exp. 12 são remedidas; nenhum treino novo.
- **Hipótese que testa**: apoio à H1 fora de pólipos.
- **O que muda na medida** (`scripts/abdome.py avaliar --protocolo oficial`; motivo no Exp. 12b):
  - CCA ligada;
  - 3 fatias de suporte por rim (a 1/6, 1/2 e 5/6 das fatias com o rim); cada fatia de query usa a da
    sua terça parte;
  - todas as fatias com o rim de cada paciente de query; média do Dice por fatia, por rim, depois média
    dos dois rins.
- **O que fica igual ao Exp. 12**: pacientes (reservados RM 0–4 e CT 0–6, nunca lidos; validação RM 5–9
  e CT 7–14), 5 pacientes-suporte por modalidade, 672 px, penúltima camada no VMamba, entrada `zscore`
  em RM e `imagenet` em CT, máscara de índice 0 do SAM. Diferença que resta para o oficial: lá há um
  único paciente-suporte e os pacientes de teste do fold; aqui são 5 suportes na validação de
  desenvolvimento.
- **O que é medido**:
  - **A. Sem treino**: DINOv2-L e VMamba-T em RM e CT; VMamba-Base em RM; DINOv2-L em CT com a entrada
    `zscore` (o oficial normaliza o CT por média e desvio). 6 avaliações.
  - **B. Ajustados** (treinos já feitos, passos 500 e 1000):
    - VMamba-T com query deformada, os 4 pares: `mr_sam_afim`, `mr_sp_afim`, `ct_sam_afim`, `ct_sp_afim`;
    - VMamba-Base, RM, regiões do SAM, as duas sementes: `mr_sam_e7_base`, `mr_sam_e7_base_s2`.
    12 avaliações.
- **Métrica**: Dice do ALP e do SAM, média e desvio nos 5 suportes; diferença pareada por suporte contra o
  mesmo modelo pré-treinado medido no mesmo protocolo (`scripts/abdome.py resumo`).
- **Regra de decisão**: a mesma do Exp. 12. Um par (modalidade, pseudo-máscara) **melhora** se o Dice SAM
  subir pelo menos 0,03 sobre o pré-treinado e ganhar em pelo menos 4 dos 5 suportes nos passos 500 e
  1000. Se nenhum par do VMamba-T cumprir, o resultado negativo do Exp. 12 fica mantido no protocolo
  oficial. Para o Base vale a ressalva do Exp. 12: só conta como ganho útil se o Base ajustado passar o
  Tiny pré-treinado. A entrada do CT só muda se `zscore` der pelo menos +0,03 no DINOv2-L; nesse caso
  proponho ao Cássio antes de remedir.
- **Máximo de execuções**: 18 avaliações, cerca de 14 h de GPU (RM ~25 min e CT ~80 min por avaliação,
  estimado). Nenhum treino. Pacientes reservados não são lidos.

## Resultado

A preencher.

## Decisão

A preencher.
