# ProtoSAM — Mapa do Projeto

## O que é

**ProtoSAM** é um framework de segmentação médica *few-shot* (one-shot) que combina três modelos:

1. **DINOv2** — extração de features visuais
2. **ALPNet** — segmentação grosseira via protótipos adaptativos
3. **SAM** (Segment Anything Model) — refinamento da máscara final

O pipeline é: *imagem de suporte + máscara → protótipos → máscara grosseira → prompts → SAM → máscara refinada*

---

## Estrutura de Diretórios

```
ProtoSAM/
├── models/                        # Coração do projeto
│   ├── ProtoSAM.py               # Orquestrador principal (wrapper)
│   ├── ProtoMedSAM.py            # Variante com rotação para imagens médicas
│   ├── grid_proto_fewshot.py     # FewShotSeg — backbone da segmentação few-shot
│   ├── alpmodule.py              # MultiProtoAsConv — módulo ALP (protótipos)
│   ├── SamWrapper.py             # Wrapper do SAM
│   └── backbone/                 # Encoders alternativos (ResNet-101)
│       └── segment_anything/     # Código oficial do SAM (Facebook)
│
├── dataloaders/                   # Carregamento e pré-processamento de dados
│   ├── dev_customized_med.py     # Composição few-shot (episódios suporte/query)
│   ├── GenericSuperDatasetv2.py  # Dataset com superpixels
│   ├── PolypDataset.py           # Dataset de pólipos
│   ├── COCODataset.py            # Dataset COCO (multi-classe)
│   └── common.py                 # Classes base
│
├── util/
│   ├── utils.py                  # CCA, rotação, visualização
│   ├── metric.py                 # Dice, Precisão, Recall
│   ├── consts.py                 # Constantes globais (IMG_SIZE, etc.)
│   └── lora.py                   # Fine-tuning LoRA
│
├── config_ssl_upload.py          # Configuração central via Sacred
├── training.py                   # Script de treino
├── validation.py                 # Validação do backbone
├── validation_protosam.py        # Validação do pipeline completo
├── validation_multiclass_coco.py # Validação multi-classe COCO
├── backbone.sh                   # Execução do treino
└── run_protosam.sh               # Execução da inferência
```

---

## Principais Classes

### 1. `FewShotSeg` — `models/grid_proto_fewshot.py`

**Papel**: Backbone da segmentação few-shot. Extrai features via DINOv2 e delega a predição ao módulo ALP.

| Método | Função |
|---|---|
| `get_encoder()` | Instancia DINOv2 (ViT-L14 ou ViT-B14) ou ResNet-101 |
| `get_features(imgs)` | Passa imagens pelo backbone e retorna feature maps |
| `get_cls()` | Instancia o `MultiProtoAsConv` |
| `forward()` | Passa suporte + query → retorna predição + losses auxiliares |

**Output do forward**: `(query_pred, align_loss, sim_maps, assign_maps, proto_grid, supp_fts, qry_fts)`

---

### 2. `MultiProtoAsConv` — `models/alpmodule.py`

**Papel**: Implementa o **Adaptive Local Prototype Pooling (ALP)** — o diferencial técnico do paper. Em vez de um protótipo global, cria uma *grade de protótipos locais* a partir da imagem de suporte.

| Método | Função |
|---|---|
| `get_prototypes()` | Average pool nas features de suporte mascaradas → grade de protótipos |
| `get_prediction_from_prototypes()` | Similaridade cosseno entre features da query e protótipos |
| `forward()` | Protótipos → logits de segmentação grosseira `(B, 2, H', W')` |

**Modos**: `mask` (protótipo único), `gridconv` (grade local), `gridconv+` (grade com indexação)

---

### 3. `ProtoSAM` — `models/ProtoSAM.py`

**Papel**: Orquestrador do pipeline completo. Une a segmentação grosseira com o SAM.

| Método | Função |
|---|---|
| `get_sam()` | Carrega checkpoint do SAM (`sam_h`, `sam_b`, `medsam`) |
| `forward()` | Pipeline completo: grosseiro → prompts → SAM → máscara refinada |
| `get_points_from_pred()` | Extrai pontos de alta confiança ou centróides da predição grosseira |
| `get_bbox_from_pred()` | Extrai bounding box da predição grosseira |
| `get_connected_components()` | CCA para análise de componentes conexas |

**Modalidades de prompt**: pontos, bounding box, máscara (configurável)

---

### 4. `SamWrapper` — `models/SamWrapper.py`

**Papel**: Interface limpa com o SAM. Recebe prompts (pontos/bbox/máscara) e retorna máscara refinada.

---

### 5. `Metric` — `util/metric.py`

**Papel**: Acumula predições e calcula métricas ao final da validação.

| Método | Função |
|---|---|
| `record()` | Registra predição vs ground truth |
| `get_mDice()` | Dice por classe |
| `get_mPrecRecall()` | Precisão e Recall por classe |
| `reset()` | Limpa acumulador |

---

## Fluxo de Dados (Ponta a Ponta)

```
Entrada
  ├─ Imagem de suporte  (B, 3, H, W)
  ├─ Máscara de suporte (B, H, W)  ← ground truth da classe alvo
  └─ Imagem de query    (B, 3, H, W)
         │
         ▼
  [DINOv2] → Feature maps (B, C, H/14, W/14)
         │
         ▼
  [MultiProtoAsConv]
    - grade de protótipos da imagem de suporte
    - similaridade cosseno com features da query
    → Logits grosseiros (B, 2, H', W')
         │
         ▼
  [ProtoSAM - geração de prompts]
    - upsample para resolução original
    - CCA opcional
    - extrai: pontos / bbox / máscara
         │
         ▼
  [SAM]
    - encoder de imagem (ViT)
    - encoder de prompts
    - decoder de máscara
    → Máscara refinada (H, W)
         │
         ▼
  Saída: máscara binária final
```

---

## Sequência de Leitura Recomendada

Para entender o projeto de dentro para fora, leia nesta ordem:

| # | Arquivo | Por que ler |
|---|---|---|
| 1 | `util/consts.py` | Constantes globais — entende as dimensões usadas |
| 2 | `util/metric.py` | Como o projeto mede resultados (Dice, etc.) |
| 3 | `models/alpmodule.py` | **Coração do paper** — protótipos adaptativos locais |
| 4 | `models/grid_proto_fewshot.py` | Backbone few-shot completo com DINOv2 |
| 5 | `models/SamWrapper.py` | Como o SAM é integrado |
| 6 | `models/ProtoSAM.py` | Orquestrador do pipeline completo |
| 7 | `dataloaders/dev_customized_med.py` | Como episódios suporte/query são compostos |
| 8 | `config_ssl_upload.py` | Todos os hiperparâmetros e configurações |
| 9 | `training.py` | Loop de treino completo |
| 10 | `validation_protosam.py` | Loop de inferência/validação |

---

## Datasets Suportados

| Dataset | Domínio | Formato |
|---|---|---|
| SABS | CT abdominal (rins, fígado, baço) | NIfTI |
| CHAOS-T2 | MRI abdominal | NIfTI |
| Kvasir/CVC | Pólipos (endoscopia) | PNG/JPEG |
| COCO | Imagens naturais multi-classe | JSON + JPEG |

---

## Variáveis de Configuração Importantes

Todas definidas em `config_ssl_upload.py` via Sacred:

```python
modelname        = 'dinov2_l14'   # backbone DINOv2
proto_grid_size  = 8              # tamanho da grade de protótipos
protosam_sam_ver = 'sam_h'        # variante do SAM
use_bbox         = True           # usar bbox como prompt
use_points       = True           # usar pontos como prompt
do_cca           = True           # pós-processamento por CCA
point_mode       = 'conf'         # 'conf' | 'centroid' | 'both'
lora             = 0              # rank LoRA (0 = desabilitado)
```
