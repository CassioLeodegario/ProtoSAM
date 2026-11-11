# Protocolo de avaliação — pólipos, one-shot

Este arquivo define o que os números do `resultados/registro.csv` significam. Os nomes de arquivo
de cada conjunto estão congelados em `protocolo/polipos_v1.json`; nada aqui depende de caminhos absolutos.

## Tarefa

Segmentação binária one-shot: uma imagem de suporte com máscara, uma query sem máscara.
Pipeline ProtoSAM oficial (commit `64868c9`): encoder → ALP (máscara grossa) → prompts → SAM-H.

## Conjuntos

| Conjunto | Imagens | Origem | Uso |
|---|---|---|---|
| **V** | 145 | Reservadas do pool de treino (semente 1234) | Desenvolvimento: escolher camada, norm, resolução, passo |
| **T** | 100 | Teste oficial Kvasir (`TestDataset/test`) | **Só no final**, uma vez por configuração escolhida |
| Pool EFT | 1449 (1304 treino + 145 V) | 1450 imagens de treino menos o suporte K1 | Treino auto-supervisionado; máscaras manuais nunca usadas |

As 738 imagens dos quatro conjuntos de teste (Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS) ficam fora do pool.
A divisão é por imagem; não há informação de paciente.

## Suportes

| Id | Imagem | Observação |
|---|---|---|
| **S00** (= K1) | `cju3xtufwiv9c0818djsc4cqd.png` | Índice 927 do dataset de treino; 51.940 px de objeto no canvas 1024. Suporte padrão. |
| S01–S19 | ver `protocolo/polipos_v1.json` | Sorteados do treino (semente 20261014). Medem robustez à escolha do suporte. |

S00 é o melhor dos 20 suportes para o VMamba-Tiny EFT (exp. 15). **Resultado só com S00 é otimista**;
para conclusões, reporte também a média nos 20.

## Inferência

- Dataset e pré-processamento oficiais (`get_polyp_dataset`, `ResizeLongestSide(1024)`), canvas 1024×1024.
- ALP: `isval=True`, `val_wsize=2`, proto_grid 8. Grade de features 32 para VMamba (kernel 4×4), 48 para DINOv2 a 672.
  `--grade` troca a grade do VMamba (por exemplo 42, o mapa nativo do penúltimo estágio a 672).
- SAM-H: 1 ponto de confiança + centróide + caixa por componente (`point_mode=both`, `use_bbox=True`), sem CCA, sem pontos negativos.
- `--cca` liga a análise de componentes conexos do código oficial (`do_cca=True` no `run_protosam.sh`): só o
  componente mais confiante da máscara do ALP gera prompts e o SAM devolve uma máscara única. Vale para o
  caminho do SAM; `dice_alp` continua sendo a máscara bruta do ALP. A coluna `cca` do registro diz qual foi usada.
- FP32; `allow_tf32` desligado em matmul e ligado em cuDNN; sementes 42 na avaliação e 1234 no treino.
- VMamba sempre com o kernel CUDA `selective_scan_cuda` (o fallback em PyTorch é recusado).

## Métricas

Dice, IoU, precisão e recall por imagem no canvas 1024 (inclui o padding), depois média entre imagens.
`dice_alp` é a máscara grossa do ALP; `dice_sam` é a saída final do ProtoSAM.

## Comparabilidade

- Só compare linhas com o mesmo `conjunto`, o mesmo `suporte` e a mesma `cca`.
- A mesma configuração pode variar alguns milésimos entre GPUs/drivers; o SAM às vezes escolhe outro componente
  e muda muito uma imagem isolada. Séries de uma ablação devem rodar na mesma GPU (coluna `gpu`).
- Linhas `estado=importado` vêm do ProtoSAM-repro (out/2026). Linhas `novo` vêm do protovm com o código
  commitado. Linhas `rascunho` foram geradas com alteração sem commit e não entram em tabela.

## Travas

Regras conferidas pelo código, não por quem roda. `tests/test_travas.py` mostra cada uma.

- **Conjunto de teste**: `--conjunto T` só roda com `--final <nome>`, um item de `protocolo/finais.yaml`
  já commitado e igual à configuração avaliada (modelo, estágio, norma, resolução, grade, CCA e hash do checkpoint).
- **Código sem commit**: alteração em `protovm/`, `models/`, `dataloaders/`, `util/`, `scripts/`, `protocolo/`
  ou `configs/` faz a linha entrar no registro como `rascunho`.
- **Entrada**: toda avaliação compara a imagem do suporte K1, como ela chega ao encoder, com as estatísticas
  de `protocolo/referencias_v1.json`. Se o pré-processamento mudar, a avaliação não começa.
- **Números de referência**: `python scripts/checar.py` reavalia três configurações nas 10 primeiras imagens
  de V e compara com as avaliações completas R0084 a R0086 (tolerância de 0,002 na média).
- **Ensaio**: `--limite N` avalia só as N primeiras imagens e não grava no registro.
- **Checkpoint**: o caminho e o SHA-256 do arquivo avaliado ficam no `config.json` do run.
