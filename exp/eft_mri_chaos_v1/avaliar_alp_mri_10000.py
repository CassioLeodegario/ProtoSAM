
import os
import sys
import csv
import json
import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
MODULOS = ROOT / "exp/eft_tiny_512_v1"
DADOS = ROOT / "data/CHAOST2/chaos_MR_T2_normalized"
TREINO = ROOT / "reproducao/eft_mri_chaost2_fold0_10000_v1"
PILOTO = ROOT / "reproducao/eft_mri_chaost2_fold0_piloto200"

def arquivo_checkpoint(passo):
    if passo in (0, 200):
        return PILOTO / f"checkpoint_{passo:04d}.pt"
    return TREINO / f"checkpoint_{passo:05d}.pt"
SAIDA = TREINO / "avaliacao_alp_rins_positivas_f0_7ckpts"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULOS))

from eft_modelo_672 import construir_modelo

CHECKPOINTS = [0, 200, 500, 1000, 2000, 5000, 10000]
ORGAOS = [(2, "RK"), (3, "LK")]
QUERIES = list(range(4))
SUPORTE = 4

def ler_volume(pid):
    imagem = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"image_{pid}.nii.gz"))
    )
    label = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"label_{pid}.nii.gz"))
    )
    assert imagem.shape == label.shape
    assert imagem.shape[1:] == (256, 256)
    imagem = imagem.astype(np.float32)

    media_volume = float(imagem.mean())
    desvio_volume = float(imagem.std())

    assert np.isfinite(media_volume)
    assert np.isfinite(desvio_volume)
    assert desvio_volume > 1e-6

    imagem = (imagem - media_volume) / desvio_volume

    return imagem, label

def preparar_imagem(fatia):
    rgb = np.repeat(
        fatia.astype(np.float32)[..., None],
        3,
        axis=2
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).permute(2, 0, 1)[None].to(
        "cuda", dtype=torch.float32
    )

    assert tensor.shape == (1, 3, 256, 256)
    assert torch.isfinite(tensor).all().item()

    return tensor


def escrever_status(**kwargs):
    temp = SAIDA / "status.json.tmp"
    temp.write_text(json.dumps(kwargs, indent=2))
    os.replace(temp, SAIDA / "status.json")

def main():
    assert torch.cuda.is_available()
    SAIDA.mkdir(parents=True, exist_ok=True)

    csv_fatias = SAIDA / "metricas_fatias.csv"
    assert not csv_fatias.exists(), (
        "Avaliação existente. Não sobrescrever."
    )

    classmap = json.loads(
        (DADOS / "classmap_1.json").read_text()
    )

    # Verificar todos os checkpoints antes de iniciar.
    for passo in CHECKPOINTS:
        assert arquivo_checkpoint(passo).is_file()

    img_s, label_s = ler_volume(SUPORTE)

    suportes = {}

    for classe, nome in ORGAOS:
        indices = np.flatnonzero(
            np.any(label_s == classe, axis=(1, 2))
        )
        assert len(indices) > 0

        # Mesma política usada nas células 43 e 45.
        z = int(indices[len(indices) // 2])
        support = preparar_imagem(img_s[z])
        fg = torch.from_numpy(
            (label_s[z] == classe).astype(np.float32)
        )[None].to("cuda")

        suportes[classe] = (support, fg, z)
        print(f"Suporte {nome}: paciente 4, fatia {z}", flush=True)

    total_esperado = sum(
        len(classmap[nome][str(pid)])
        for _, nome in ORGAOS
        for pid in QUERIES
    ) * len(CHECKPOINTS)

    assert total_esperado == 784, total_esperado

    modelo = construir_modelo("tiny", repo=ROOT)
    modelo.to("cuda").eval().requires_grad_(False)

    campos = [
        "checkpoint", "orgao", "paciente", "fatia",
        "dice", "iou", "precisao", "recall",
        "tp", "fp", "fn", "pixels_gt", "pixels_pred"
    ]

    concluidos = 0
    inicio = time.perf_counter()

    with csv_fatias.open(
        "x", newline="", encoding="utf-8", buffering=1
    ) as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()

        for passo in CHECKPOINTS:
            checkpoint = torch.load(
                arquivo_checkpoint(passo),
                map_location="cpu",
                weights_only=False
            )
            assert checkpoint["passo"] == passo

            modelo.load_state_dict(
                checkpoint["modelo"], strict=True
            )
            del checkpoint
            modelo.eval()

            print(f"\nCHECKPOINT {passo}", flush=True)

            for classe, nome in ORGAOS:
                support, fg, _ = suportes[classe]

                for pid in QUERIES:
                    imagem, label = ler_volume(pid)
                    fatias = classmap[nome][str(pid)]

                    for z in fatias:
                        z = int(z)
                        gt = label[z] == classe
                        assert gt.any()

                        query = preparar_imagem(imagem[z])

                        with torch.inference_mode():
                            retorno = modelo(
                                [[support]], [[fg]], [[1 - fg]],
                                [query],
                                isval=True,
                                val_wsize=2,
                                show_viz=False
                            )
                            logits = retorno[0]
                            assert logits.shape == (1, 2, 256, 256)
                            assert torch.isfinite(logits).all().item()

                            pred = (
                                logits.argmax(dim=1)[0]
                                .cpu().numpy() == 1
                            )

                        tp = int((pred & gt).sum())
                        fp = int((pred & ~gt).sum())
                        fn = int((~pred & gt).sum())

                        registro = {
                            "checkpoint": passo,
                            "orgao": nome,
                            "paciente": pid,
                            "fatia": z,
                            "dice": 2 * tp / max(2*tp+fp+fn, 1),
                            "iou": tp / max(tp+fp+fn, 1),
                            "precisao": tp / max(tp+fp, 1),
                            "recall": tp / max(tp+fn, 1),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "pixels_gt": int(gt.sum()),
                            "pixels_pred": int(pred.sum())
                        }

                        escritor.writerow(registro)
                        concluidos += 1

                        if concluidos % 25 == 0:
                            escrever_status(
                                estado="executando",
                                concluidos=concluidos,
                                total=total_esperado,
                                checkpoint=passo,
                                orgao=nome,
                                paciente=pid
                            )
                            print(
                                f"{concluidos}/{total_esperado}",
                                flush=True
                            )

                        del query, retorno, logits

    # Resumir por paciente, depois entre os quatro pacientes.
    agrupado = defaultdict(list)

    with csv_fatias.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            chave = (
                int(r["checkpoint"]),
                r["orgao"],
                int(r["paciente"])
            )
            agrupado[chave].append(float(r["dice"]))

    linhas_pacientes = []

    for (passo, orgao, pid), valores in sorted(agrupado.items()):
        linhas_pacientes.append({
            "checkpoint": passo,
            "orgao": orgao,
            "paciente": pid,
            "n_fatias": len(valores),
            "dice_medio": float(np.mean(valores)),
            "dice_desvio_fatias": float(np.std(valores, ddof=1))
                if len(valores) > 1 else 0.0
        })

    with (SAIDA / "resumo_pacientes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        escritor = csv.DictWriter(
            f, fieldnames=list(linhas_pacientes[0])
        )
        escritor.writeheader()
        escritor.writerows(linhas_pacientes)

    print("\n=== DICE MÉDIO ± DP ENTRE PACIENTES ===", flush=True)

    for passo in CHECKPOINTS:
        for _, orgao in ORGAOS:
            valores = [
                r["dice_medio"]
                for r in linhas_pacientes
                if r["checkpoint"] == passo and r["orgao"] == orgao
            ]

            assert len(valores) == 4

            print(
                f"{orgao} | passo {passo:3d} | "
                f"{np.mean(valores):.4f} ± "
                f"{np.std(valores, ddof=1):.4f}",
                flush=True
            )

    escrever_status(
        estado="concluido",
        concluidos=concluidos,
        total=total_esperado,
        segundos=time.perf_counter() - inicio
    )
    print("\nAVALIAÇÃO CONCLUÍDA", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        erro = traceback.format_exc()
        print(erro, flush=True)
        SAIDA.mkdir(parents=True, exist_ok=True)
        escrever_status(estado="erro", traceback=erro)
        raise
