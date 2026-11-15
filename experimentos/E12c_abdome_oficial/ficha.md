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

Medido em 15/11/2026 (`runs/E12c_oficial.log`, 18 avaliações, cerca de 7 h). Dice SAM, média ± desvio nos
5 suportes; Δ é a diferença pareada contra o mesmo modelo pré-treinado, com os suportes em que ganha.

**A. Sem treino**

| Modelo | RM | CT |
|---|---|---|
| DINOv2-L | 0,747 ± 0,028 | 0,603 ± 0,090 |
| VMamba-T | 0,504 ± 0,081 | 0,350 ± 0,082 |
| VMamba-Base | 0,296 ± 0,079 | não medido |

DINOv2-L em CT com a entrada `zscore`: 0,597 (−0,006). A entrada do CT fica `imagenet`.

**B. Ajustados**

| Par | Passo 500 | Passo 1000 |
|---|---|---|
| VMamba-T, RM, regiões do SAM | 0,479 (−0,024; 0/5) | 0,476 (−0,027; 0/5) |
| VMamba-T, RM, superpixels | 0,504 (0,000; 2/5) | 0,495 (−0,008; 2/5) |
| VMamba-T, CT, regiões do SAM | 0,348 (−0,001; 3/5) | 0,333 (−0,017; 2/5) |
| VMamba-T, CT, superpixels | 0,309 (−0,041; 1/5) | 0,306 (−0,044; 1/5) |
| VMamba-Base, RM, regiões do SAM, semente 1234 | 0,386 (+0,090; 5/5) | 0,374 (+0,079; 5/5) |
| VMamba-Base, RM, regiões do SAM, semente 4321 | 0,362 (+0,066; 4/5) | 0,397 (+0,101; 5/5) |

As linhas de CT com superpixels foram calculadas a partir das linhas por suporte do log.

## Decisão

Nenhum par do VMamba-T cumpre a regra: o resultado negativo do Exp. 12 fica mantido no protocolo oficial,
nas duas modalidades e nos dois tipos de pseudo-máscara. O VMamba-Base em RM com regiões do SAM cumpre a
regra nas duas sementes, mas o Base ajustado (0,36 a 0,40) fica abaixo do Tiny pré-treinado (0,504); pela
ressalva, não conta como ganho útil. O protocolo oficial passa a ser o do abdome daqui em diante.
