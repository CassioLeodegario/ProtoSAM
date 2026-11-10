# Piloto EFT-SP VMamba — v1

## Instalação e sequência

Extraia a pasta do ZIP em `/workspace/ProtoSAM-repro/exp/`. Os módulos e os quatro notebooks devem ficar juntos em `exp/eft_sp_piloto_v1/`.

1. Encerre o kernel do diagnóstico Base. Selecione Python (protosam) nos notebooks novos.
2. Execute `04_preparar_pool_piloto.ipynb`. A preparação usa CPU e imprime progresso a cada 25 imagens.
3. Copie o caminho DADOS para `PASTA_DADOS` no `05_piloto_tiny.ipynb`.
4. Execute o Tiny e envie o resumo final. Small/Base ficam preparados para uso posterior, após análise do Tiny.

Se houver imagens sem região elegível, a preparação registra as exclusões e informa as contagens efetivas. Não substitui nem move imagens entre divisões. Se o pool de treino elegível tiver menos de 200 imagens ou a validação ficar vazia, a preparação para. A proporção 1.304/145 descreve a reserva antes dessas exclusões.

A preparação ordena o pool por nome/caminho e usa NumPy seed 1234. O treino do piloto seleciona 200 imagens distintas por permutação, uma região uniforme dentre as elegíveis de cada imagem. A validação tem uma região fixa por imagem elegível. Os planos e caches são iguais entre variantes. Esta é uma escolha experimental para um piloto, ainda não um sampler final de treino longo.

## O que foi preservado

Filtros finais do notebook 04: Felzenszwalb scale100/sigma1/min_size400; max RGB>10; maior componente; preencher buracos; erosão3; região interna >=100 pixels; cobertura interna >=95%; FG sobrevive a nearest32; BG local >0.95 em support e query. Query é flip horizontal da mesma imagem/região. FG global é permitido. Regiões não são rótulos de pólipos. Não lemos os arquivos de máscaras manuais.

Modelo baseado no diagnóstico aprovado: último estágio Tiny/Small768 canais, Base1024; entrada256→grade8×8→interpolação32×32; ALP kernel4; transferência classifier.norm→outnorm3; carregamento strict=True. Forward e alignLoss herdados do ALP. SAM não é carregado.

200 atualizações com batch1, sem acumulação, full fine-tuning, float32, AdamW lr1e-5, betas0.9/0.999, eps1e-8, weight_decay0. CE sem pesos + alinhamento, coeficientes1. Sem scheduler, clipping ou AMP. Mantém drop path específico de cada encoder. Não é reprodução exata do artigo.

## Validação

Antes e depois nos mesmos episódios reservados. eval() desativa drop path e alinhamento no código oficial. isval=False mantém o kernel ALP de treino4×4. CE média por imagem e Dice de pseudo-foreground por imagem; não somamos a loss de alinhamento na validação. As 738 queries de teste ficam fora do piloto. A divisão por imagem não garante separação por paciente.

## Retomada

Guarde o caminho EXECUÇÃO, impresso antes da construção do modelo. No mesmo notebook e com o mesmo código, coloque `RETOMAR = Path("caminho da execução")` e execute. Se houve OOM/erro de CUDA, reinicie o kernel correto antes de retomar.

checkpoint_latest.pt contém modelo, AdamW, scheduler=None, passo, RNG Python/NumPy/PyTorch/CUDA, configuração e baseline. Gravação atômica no passo0 e a cada25 passos, incluindo200; mantém só a última versão. Durante a gravação, o arquivo temporário pode exigir espaço para outra cópia. Não alteramos os checkpoints pré-treinados. Logs pós-checkpoint de uma tentativa interrompida permanecem registrados; uma nova tentativa tem log próprio e passo inicial explícito. Não concatene tentativas sem considerar o passo de retomada. Não garantimos determinismo bit a bit de kernels CUDA.

Se interromper antes do primeiro checkpoint, retomar pela mesma pasta reconstrói o modelo original e repete a validação inicial. Se interromper depois do passo200, retoma apenas a validação final. Uma pasta com resumo.json é considerada concluída e não treina novamente. Cada execução nova deve usar RETOMAR=None.

Para retomar apenas a preparação dos dados, informe DESTINO no notebook04 com a pasta DADOS já criada. Os caches existentes são verificados por hashes. Se dados/configuração/código/ambiente mudarem, não ignorar as verificações: criar execução nova e registrar a mudança.

## Artefatos

Dados: config.json, divisao.json, cache NPZ/JSON por imagem, planos.json, excluidos_sem_regiao.json.
Execução: config.json, fontes/, planos.json, ambiente GPU/git, tentativa_*.json, passos_*.jsonl, validacao_inicial.json, validacao_final.json, checkpoint_latest.pt, resumo.json e erro_*.txt em caso de falha. Tempos registrados incluem verificações e não são benchmark de custo.

## Validação da entrega

Verificação local de sintaxe Python e células dos quatro notebooks. Testes da divisão fixa (contagens, disjunção e invariância à ordem), equivalência da área útil ao notebook original em entradas sintéticas e extração dos filtros finais. Sem execução de treinamento/CUDA neste ambiente; apenas os diagnósticos de 12 passos anteriores foram confirmados pelo usuário no Runpod. Esta entrega precisa ser executada lá.
