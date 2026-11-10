# Tiny: continuação até 2.000 atualizações
Execute com kernel **Python (protosam)** recém-iniciado. Encerre os kernels anteriores para liberar a GPU.
Este notebook restaura pesos, AdamW e RNG do piloto de 200 passos e executa mais 1.800 passos em uma nova pasta.
Mantém normalização RGB, último estágio 3, outnorm3 treinável, ALP e perdas do piloto. Não testa outras camadas ou normalizações.
Cada ciclo percorre as 1.304 imagens de treino em ordem sorteada; a região é sorteada novamente e pode coincidir com a anterior.
Validação fixa de pseudo-máscaras a cada 250 passos. Checkpoint de retomada a cada 50. Checkpoints 500/1000/1500/2000 preservados para avaliação posterior ALP + SAM. Não seleciona automaticamente o melhor modelo.

Instalação: extraia a pasta eft_tiny_2000_v1 em /workspace/ProtoSAM-repro/exp/. Abra 09_tiny_2000.ipynb e execute as células em ordem. Os dados preparados e as fontes do piloto devem permanecer nos caminhos originais. Nenhum arquivo do repositório oficial nem do piloto será alterado.

Ao terminar, envie o resumo JSON e o caminho RESULTADOS. A comparação anotada ALP + SAM dos checkpoints é a próxima etapa; este notebook não executa SAM nem lê máscaras manuais.

Validação local do pacote: sintaxe e consistência da sequência sintética. Execução CUDA depende do Runpod e não foi realizada no ambiente de preparação.
