# Diagnóstico compartilhado EFT-SP / VMamba — v1

Extraia esta pasta em `/workspace/ProtoSAM-repro/exp/`. Resultado:
`/workspace/ProtoSAM-repro/exp/eft_sp_diagnostico_v1/`.

Arquivos novos; não substitua o notebook Tiny original. O módulo precisa ficar junto dos três notebooks.

1. Encerre pelo Jupyter o kernel do notebook Tiny original; fechar a aba não basta.
2. Abra `06_eft_vmamba_small_compartilhado.ipynb`, selecione o kernel do ambiente protosam e execute as células em ordem.
3. Envie as 12 linhas e a mensagem final. Se houver erro, envie o traceback e não inicie Base ainda.
4. Depois de Small passar, encerre seu kernel e execute `07_eft_vmamba_base_compartilhado.ipynb`.
5. O notebook Tiny compartilhado está disponível para uso futuro; não é necessário repetir o teste agora.

O módulo usa as configurações Tiny/Small/Base do handoff e a construção/extração do notebook Tiny validado. Modifica apenas os métodos get_encoder/get_cls/get_features; forward e alignLoss continuam herdados do ALP. O carregamento é strict=True após transferir classifier.norm para outnorm3 e remover classifier.head.

Último estágio: Tiny/Small 768 canais, Base 1024. Entrada 256 → grade 8×8 → interpolação bilinear para 32×32. ALP kernel 4×4. A experiência com penúltimo estágio permanece separada.

Política idêntica: 12 atualizações, seis episódios em ordem repetidos duas vezes, seed 1234, float32, sem autocast, batch 1, AdamW lr 1e-5 e weight_decay 0. CE sem pesos + alinhamento, ambos com peso 1. Drop path próprio de cada backbone mantido. Pesos iniciais copiados para RAM e restaurados no finally; não há persistência de pesos treinados. Interrupção abrupta do processo/GPU pode impedir o finally; os checkpoints de origem nunca são escritos.

Cada execução cria nova pasta em reproducao/, com config.json, config_episodios.json, passos.jsonl, resumo.json, fontes usadas, hashes de checkpoint/episódios/código e informações de ambiente. Falhas são registradas em erro.txt e propagadas. Não existe captura silenciosa para pular episódios. O SHA256 do VMamba é conferido contra o handoff.

Não é treinamento definitivo nem reprodução exata de EFT do artigo. Não implementa ainda split treino/val, novas augmentations, LoRA, orçamento final ou retomada. SAM não é carregado. Custo de diagnóstico com verificações e sincronizações não é benchmark.

Validação desta entrega: sintaxe Python, células de notebooks e preservação por AST dos métodos forward/alignLoss (sem overrides). Sem execução CUDA neste ambiente; Small/Base e a versão refatorada dependem da execução no Runpod. Tiny original passou forward/backward e 12 atualizações conforme saídas fornecidas pelo usuário.
