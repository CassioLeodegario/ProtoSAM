# Exp. 1 — CCA ligada ou desligada no protocolo

- **Pergunta**: a análise de componentes conexos do código oficial (`--cca`) muda o Dice do SAM na média dos 20 suportes?
- **Hipótese que testa**: nenhuma (apoio). Decide uma opção do protocolo antes de ele ser congelado.
- **O que é comparado**: cada modelo com a CCA desligada e ligada.
  - DINOv2-L: desligada no Exp. 2a (R0085 e S01 a S19); ligada com
    `python scripts/avaliar.py --modelo dinov2_l14 --cca --suportes todos --sem-mascaras`
  - VMamba-Tiny, melhor checkpoint (EFT com regiões do SAM, passo 4000, penúltimo estágio, `sem_norm`, grade 32):
    desligada e ligada nos 20 suportes (o R0086, de S00, foi gravado com o nome antigo do treino e não entra), com
    `python scripts/avaliar.py --modelo vmamba_tiny --treino-desc 3 checkpoint 672 --pesos <...>/treino_SAM/checkpoint_4000.pt --estagio 2 --norm sem_norm`
- **O que fica fixo**: validação V (145 imagens), suportes S00 a S19, encoder a 672, protocolo v1.
- **Métrica**: Dice do SAM, média nos 20 suportes; diferença pareada (ligada − desligada) com IC de 95%.
  O Dice do ALP não muda com a CCA e serve de conferência: tem de sair idêntico.
- **Regra de decisão**: a escolha é feita **só pelo DINOv2-L**, o modelo de referência, para não depender do
  que favorece o VMamba. A CCA ligada é a configuração oficial dos autores e passa a ser a do protocolo,
  a menos que o IC da diferença no DINOv2-L fique inteiro abaixo de zero; nesse caso o protocolo segue sem CCA.
  O resultado do VMamba é reportado, mas não entra na decisão. Decidido isso, todos os modelos comparados
  na dissertação rodam com a mesma opção.
- **Máximo de execuções**: 60 avaliações (cerca de 5 horas de GPU).

## Resultado

Ver [tabela_dinov2_l.md](tabela_dinov2_l.md) e [tabela_vmamba.md](tabela_vmamba.md), geradas depois das avaliações.

## Decisão

A preencher.
