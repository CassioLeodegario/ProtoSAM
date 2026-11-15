# Exp. 1 — DINOv2-L com a CCA desligada e ligada (20 suportes, validação)

Conjunto V, CCA indicada em cada linha. Gerado de resultados/registro.csv; não edite à mão.

| Configuração | Suportes | dice_alp (média ± dp) | dice_sam (média ± dp) | Δ dice_alp vs. CCA desligada [IC 95%] | Δ dice_sam vs. CCA desligada [IC 95%] | Suportes em que ganha |
|---|---|---|---|---|---|---|
| CCA desligada | 20 | 0,587 ± 0,119 | 0,616 ± 0,189 | — | — | — |
| CCA ligada | 20 | 0,587 ± 0,119 | 0,668 ± 0,177 | +0,000 [+0,000; +0,000] | +0,052 [+0,038; +0,066] | 19/20 |

Registros usados: R0085 a R0165 (40 linhas).
Diferença pareada nos suportes comuns; IC por bootstrap (10000 reamostras). "Suportes em que ganha" refere-se a dice_sam.
