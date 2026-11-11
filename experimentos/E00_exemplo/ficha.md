# Exp. 0 — exemplo: o EFT-SP de 256 px ajuda em média nos 20 suportes?

Exemplo de ficha e de tabela, feito com avaliações de out/2026 que já estavam no registro
(R0042 a R0081, RTX 4090). Serve de modelo e de teste do gerador; não é resultado da dissertação.

- **Pergunta**: o VMamba-Tiny com EFT-SP (256 px, passo 2000) supera o pré-treinado na média dos 20 suportes?
- **Hipótese que testa**: nenhuma (apoio).
- **O que é comparado**: VMamba-Tiny pré-treinado e o mesmo modelo com EFT-SP no passo 2000, último estágio, norma do checkpoint.
- **O que fica fixo**: validação V, suportes S00 a S19, encoder a 672, grade 32, CCA desligada.
- **Métrica**: Dice do ALP e do SAM, média nos 20 suportes, diferença pareada com IC de 95%.
- **Regra de decisão**: se o IC da diferença no SAM excluir zero, o EFT segue como linha principal.
- **Máximo de execuções**: 40 avaliações (já feitas).

## Resultado

Ver [tabela.md](tabela.md).

## Decisão

Exemplo; sem decisão.
