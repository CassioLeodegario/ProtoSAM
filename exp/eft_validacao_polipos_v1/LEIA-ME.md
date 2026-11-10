# Validação anotada de transferência do piloto EFT

Extraia esta pasta em `/workspace/ProtoSAM-repro/exp/`, obtendo `exp/eft_validacao_polipos_v1/`. Mantenha os dois módulos Python junto dos três notebooks. Encerre os kernels anteriores. Abra primeiro `08a_validar_polipos_tiny.ipynb`, selecione Python (protosam) e execute suas células em ordem. Os caminhos dos três pilotos estão preenchidos a partir dos registros fornecidos. Não reexecute os notebooks antigos.

## Protocolo

Usa somente as145 imagens da validação fixa dos pilotos. Mapeia as imagens e máscaras por caminho completo no manifesto; indexa as queries no dataset oficial que também fornece o suporte (pool original de1450 imagens). A factory oficial instancia metadados de teste, mas o objeto de teste não é indexado nem avaliado. A máscara manual da query é usada somente no cálculo das métricas; o ALP recebe a máscara do suporte K1, como no protocolo one-shot. Suporte reservado fora da adaptação: cju3xtufwiv9c0818djsc4cqd.png, com51940 pixels FG no canvas1024.

A condição antes usa a mesma inicialização do piloto (classifier.norm→outnorm3). A condição depois carrega strict=True o estado inteiro salvo após200 atualizações. Modelos sempre em eval, sem gradientes. Não usa a variante de penúltimo estágio e não usa outnorm inicial escala1/viés0 como baseline. Fonte original não é editada.

Encoder672, último estágio21×21, interpolado para32×32. ALP com isval=True e val_wsize=2, conforme notebook de avaliação original. O kernel configurado é4×4; o comportamento de inferência com val_wsize2 é preservado no código oficial. No piloto de treino, isval era False: não confundir os dois contextos.

SAM-H congelado, canvas1024, caixa+ponto de confiança+centroide, CCAFalse, sem máscara como prompt e sem pontos negativos. Usa forward oficial e seleção original alternativa0 por componente. Não incorpora a ablação de maior score.

Dice, IoU, precisão, recall e contagens TP/FP/FN são calculados no canvas1024, como no notebook fornecido, e resumidos por média por imagem. Não são métricas em resolução física/nativa e não incluem HD95 nesta etapa. ALP e ALP+SAM são medidos separadamente. Os tempos deste caminho não servem para benchmark, porque ALP é executado duas vezes.

A validação anotada é uma adição explícita ao protocolo, para verificar transferência de superpixels para pólipos. Pode orientar desenvolvimento, mas não deve ser apresentada como teste independente. Separação por paciente segue não comprovada.

## Registros e retomada

A pasta AVALIAÇÃO é impressa após verificações e hashes, antes de carregar os modelos. Primeira execução usa RETOMAR=None. Após interrupção, use o mesmo caminho em RETOMAR e execute de novo no kernel correto. Os JSON de conclusão por imagem são gravados depois das máscaras, e a retomada verifica seus hashes. Uma imagem parcialmente gravada é refeita; não é ignorada. Condição antes é concluída antes de carregar pesos depois.

Guarda config.json, cópias das fontes, antes/depois com máscaras alp/sam/gt e registros individuais, por_imagem.csv, resumo.json e traceback em falha dentro do loop. Hashes cobrem dados, pesos e fontes utilizadas. As imagens pré-processadas e GT são conferidas entre condições. Mesmo suporte e todas as145 queries são usados antes/depois.

## Verificação desta entrega

Sintaxe Python/células verificada localmente, métricas binárias verificadas em casos sintéticos, forward e alinhamento preservados por herança. Configuração SAM e argumentos de inferência derivados do notebook anexado. Execução com os datasets e checkpoints reais/CUDA depende do Runpod.
