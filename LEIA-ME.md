# ProtoSAM-vmamba

Fork do ProtoSAM oficial (commit `64868c9`) com o VMamba como encoder do ALP e o EFT-SP
(ajuste auto-supervisionado com superpixels). O código oficial não é alterado; tudo o que é nosso
fica em `protovm/`. Veja o protocolo em [PROTOCOLO.md](PROTOCOLO.md).

## Estrutura

```
protovm/            código nosso, uma cópia só
  encoder.py        construir_alp(modelo, estagio, norm, resolucao, grade, pesos)
  eft_dados.py      superpixels, cache, planos de episódios
  eft_treino.py     laço de treino EFT-SP (retomável)
  avaliar.py        ALP + SAM-H em V ou T, um suporte por chamada
  registro.py       resultados/registro.csv
  protocolo.py      conjuntos e suportes congelados; travas (ver PROTOCOLO.md)
  tabelas.py        tabelas e diferença pareada a partir do registro
configs/eft/        um YAML por experimento de treino (herdam de _base.yaml)
scripts/            preparar_dados.py, treinar_eft.py, avaliar.py, gerar_tabelas.py, checar.py, ligar_dados.sh
protocolo/          polipos_v1.json (nomes dos arquivos de V, T, pool e suportes),
                    finais.yaml (configurações autorizadas no teste), referencias_v1.json
resultados/         registro.csv — uma linha por avaliação
notebooks/          só análise: leem o registro e as pastas de runs/
experimentos/       uma pasta por experimento: ficha.md, tabela.yaml e tabela.md (gerada)
rascunho/           testes livres; fora do git e fora do texto
DIARIO.md           três linhas por sessão
tests/              verificações automáticas (código oficial, travas, tabelas, números de referência)
migracao/           scripts usados uma vez para trazer o ProtoSAM-repro
runs/               saídas (fora do git)
dados_eft/          caches de episódios (fora do git)
```

## Ambiente (servidor com RTX 5090)

```bash
conda create -n protosam python=3.10 -y && conda activate protosam
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
grep -vE "^(torch|torchvision|triton|dinov2|nvidia-.*)==" requirements.txt > /tmp/req.txt
pip install "setuptools<70" wheel && pip install --no-build-isolation -r /tmp/req.txt
pip install pyyaml pandas pytest
conda install -c conda-forge git -y

# Kernel do VMamba: o models/vmamba.py importa `selective_scan_cuda`, que vem do mamba-ssm
# (no pod: mamba-ssm 2.3.1). Na 5090 (sm_120) precisa compilar com CUDA 12.8:
conda install -c nvidia cuda-toolkit=12.8 -y
export TORCH_CUDA_ARCH_LIST="12.0" MAMBA_FORCE_BUILD=TRUE
pip install mamba-ssm==2.3.1 --no-build-isolation      # demora; rode num screen
python -c "import torch, selective_scan_cuda; print('kernel ok')"   # o torch tem de vir antes
```

Dados e pesos (uma vez, depois de clonar):

```bash
bash scripts/ligar_dados.sh ~/cassio_pereira       # pasta com ProtoSAM/ e polyp_raw/
python scripts/checar.py                           # código oficial, travas e números de referência
```

Os pesos do DINOv2 ficam em `pretrained_model/hub/checkpoints/` (`dinov2_vitl14_pretrain.pth`,
`dinov2_vitb14_pretrain.pth`, `dinov2_vits14_pretrain.pth`) e vêm de
`https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` (troque `vits14` por
`vitb14` ou `vitl14`). O código do hub fica em `pretrained_model/hub/facebookresearch_dinov2_main`.

## Fluxo de trabalho

```bash
# 0. checagem, antes de cada série de avaliações e depois de mexer no código (2 min de GPU)
python scripts/checar.py
# 1. dados (CPU, uma vez por resolução)
python scripts/preparar_dados.py --resolucao 512 --superpixels-em 256
# 2. treino (num screen)
python scripts/treinar_eft.py configs/eft/tiny_512.yaml
# 3. avaliação em V; cada checkpoint/suporte vira uma linha do registro
python scripts/avaliar.py --treino runs/<pasta> --passos 1000 2000 3000                 # a 672
python scripts/avaliar.py --treino runs/<pasta> --passos 1000 2000 3000 --resolucao 512
# 4. análise
jupyter lab notebooks/01_resultados.ipynb
```

Não copie código para mudar um número: se faltar um parâmetro, acrescente-o em `protovm/` (com
valor padrão igual ao comportamento atual). Treinos e avaliações longos rodam pelos scripts, dentro
de um `screen`.

## Passo a passo de cada experimento

| # | Passo | O que fazer | Pronto quando |
|---|---|---|---|
| 1 | Ficha | Copiar `experimentos/MODELO` para `experimentos/ENN_nome` e preencher `ficha.md` até a regra de decisão | Commit da ficha feito, antes de rodar |
| 2 | Configuração | Um YAML por treino em `configs/`; parâmetro que faltar entra em `protovm/` | Commit feito |
| 3 | Checagem | `python scripts/checar.py` | Termina com `CHECAGEM OK` |
| 4 | Treino, se houver | Num `screen`: `python scripts/treinar_eft.py configs/<arquivo>.yaml` | Checkpoints em `runs/` |
| 5 | Avaliação | `python scripts/avaliar.py ... --suportes todos` (validação) | Linhas `novo` em `resultados/registro.csv` |
| 6 | Tabela | Preencher `tabela.yaml` e rodar `python scripts/gerar_tabelas.py experimentos/ENN_nome` | `tabela.md` gerado; nenhum número digitado |
| 7 | Decisão | Preencher "Resultado" e "Decisão" na ficha, pela regra do passo 1 | Ficha fechada |
| 8 | Fim do dia | Três linhas no `DIARIO.md`; `git add`, `git commit`, `git push` | Push feito |
| 9 | Texto | Escrever o parágrafo na dissertação apontando para a tabela gerada | Parágrafo escrito |

Duas exceções:

- **Ideia rápida**: vai em `rascunho/` (ver `rascunho/LEIA-ME.md`), ou roda com `--limite N`, que
  não grava no registro. Número de rascunho não entra no texto; se a ideia der certo, volte ao passo 1.
- **Conjunto de teste**: fica fora desse ciclo. Roda uma vez, no fim, só para itens commitados de
  `protocolo/finais.yaml` (ver PROTOCOLO.md, seção Travas).

`experimentos/E00_exemplo` mostra uma ficha e uma tabela prontas.

## Validação da migração

Antes de confiar na estrutura nova, reproduza o melhor resultado antigo (registro `R0027`,
V/S00, ALP 0,538 / SAM 0,625 no passo 2000), em duas etapas:

```bash
# A. Avaliador: o checkpoint ANTIGO avaliado pelo código novo (isola o avaliador)
python scripts/avaliar.py --modelo vmamba_tiny --treino-desc 3 checkpoint 256 \
  --pesos ~/cassio_pereira/ProtoSAM-repro/reproducao/eft_tiny_2000_v2_20261011T204725195685Z/checkpoint_2000.pt

# B. Treino: dados + EFT do zero com o código novo
python scripts/preparar_dados.py --resolucao 256
python scripts/treinar_eft.py configs/eft/tiny_256.yaml
python scripts/avaliar.py --treino runs/<pasta> --passos 2000
```

A etapa A deve bater com o 0,625 quase exatamente. Na B, diferenças de alguns milésimos são
esperadas (GPU, driver e torch 2.7 em vez de 2.0).

**Já verificado em CPU** (`migracao/verificar_dados_256.py`): com Pillow 10.4.0 e scikit-image 0.24.0,
o cache gerado pelo protovm é idêntico ao de out/2026 em todas as 1449 imagens (rgb, superpixels,
área útil, candidatos), e os planos de validação, piloto (200) e treino (5000) também.
O cache 512 com superpixels a 256 gera exatamente o mesmo plano.
Com Pillow 13 duas imagens mudam 1 nível de cor no redimensionamento, o que altera o plano a partir
do episódio 79: **mantenha `pillow==10.4.0`** (já fixado no requirements.txt).
