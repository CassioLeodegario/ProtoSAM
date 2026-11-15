# Exp. 2b — DINOv2-L, B e S nos 20 suportes (validação)

Conjunto V, CCA desligada. Gerado de resultados/registro.csv; não edite à mão.

| Configuração | Suportes | dice_alp (média ± dp) | dice_sam (média ± dp) | Δ dice_alp vs. DINOv2-L [IC 95%] | Δ dice_sam vs. DINOv2-L [IC 95%] | Suportes em que ganha |
|---|---|---|---|---|---|---|
| DINOv2-L | 20 | 0,587 ± 0,119 | 0,616 ± 0,189 | — | — | — |
| DINOv2-B | 20 | 0,566 ± 0,106 | 0,599 ± 0,185 | -0,021 [-0,033; -0,009] | -0,017 [-0,030; -0,003] | 5/20 |
| DINOv2-S | 20 | 0,552 ± 0,114 | 0,572 ± 0,190 | -0,035 [-0,056; -0,016] | -0,043 [-0,067; -0,024] | 4/20 |

Registros usados: R0085 a R0145 (60 linhas).
Diferença pareada nos suportes comuns; IC por bootstrap (10000 reamostras). "Suportes em que ganha" refere-se a dice_sam.
