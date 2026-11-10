
import os
import sys
import csv
import json
import time
import fcntl
import traceback
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

import numpy as np
import torch
import SimpleITK as sitk

ROOT = Path("/home/1542521/cassio_pereira/ProtoSAM-repro")
MODULOS = ROOT / "exp/eft_tiny_512_v1"
DADOS = ROOT / "data/CHAOST2/chaos_MR_T2_normalized"
TREINO = ROOT / "reproducao/eft_mri_chaost2_fold0_10000_v1"
PILOTO = ROOT / "reproducao/eft_mri_chaost2_fold0_piloto200"

SAIDA = TREINO / "avaliacao_alp_sam_rins_f0_7ckpts"
CSV_SAM = SAIDA / "metricas_fatias.csv"
CSV_PACIENTES = SAIDA / "resumo_pacientes.csv"
CSV_GERAL = SAIDA / "resumo_geral.csv"
STATUS = SAIDA / "status.json"

REF_ALP = (
    TREINO / "avaliacao_alp_rins_positivas_f0_7ckpts"
    / "metricas_fatias.csv"
)

CFG_SAM = (
    ROOT / "reproducao"
    / "eft_tiny_passo_2000_validacao_20261011T221527472147Z"
    / "config.json"
)

CHECKPOINTS = [0, 200, 500, 1000, 2000, 5000, 10000]
ORGAOS = [(2, "RK"), (3, "LK")]
QUERIES = [0, 1, 2, 3]
SUPORTE = 4
TOTAL = 784

CAMPOS = [
    "checkpoint", "orgao", "paciente", "fatia",
    "dice_alp", "dice_sam", "iou_sam",
    "precisao_sam", "recall_sam",
    "tp_sam", "fp_sam", "fn_sam",
    "pixels_gt", "pixels_sam", "sam_executado"
]

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULOS))

from eft_modelo_672 import construir_modelo
from models.ProtoSAM import ProtoSAM, ALPNetWrapper, ALPNetInput

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def salvar_json(path, objeto):
    temporario = path.with_name(path.name + ".tmp")
    temporario.write_text(
        json.dumps(objeto, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(temporario, path)


def checkpoint(passo):
    if passo in (0, 200):
        return PILOTO / f"checkpoint_{passo:04d}.pt"
    return TREINO / f"checkpoint_{passo:05d}.pt"


@lru_cache(maxsize=5)
def volume(pid):
    img = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"image_{pid}.nii.gz"))
    ).astype(np.float32)

    gt = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"label_{pid}.nii.gz"))
    )

    assert img.shape == gt.shape
    assert img.shape[1:] == (256, 256)
    assert np.isfinite(img).all()

    media = float(img.mean())
    desvio = float(img.std())

    assert np.isfinite(desvio) and desvio > 1e-6

    # Mesmo z-score por volume usado no treinamento
    # e na avaliação ALP-MRI anterior.
    img = (img - media) / desvio

    return img, gt


def preparar(fatia):
    rgb = np.repeat(
        fatia.astype(np.float32)[..., None],
        3, axis=2
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).permute(2, 0, 1)[None].to(
        "cuda", dtype=torch.float32
    )

    assert tensor.shape == (1, 3, 256, 256)
    return tensor


def metricas(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precisao": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pixels_pred": int(pred.sum())
    }


def chave(registro):
    return (
        int(registro["checkpoint"]),
        registro["orgao"],
        int(registro["paciente"]),
        int(registro["fatia"])
    )


def resumir(registros):
    grupos = defaultdict(list)

    for r in registros:
        grupos[(
            int(r["checkpoint"]),
            r["orgao"],
            int(r["paciente"])
        )].append(r)

    pacientes = []

    for (passo, orgao, pid), valores in sorted(grupos.items()):
        alp = np.array(
            [float(r["dice_alp"]) for r in valores]
        )
        sam = np.array(
            [float(r["dice_sam"]) for r in valores]
        )

        pacientes.append({
            "checkpoint": passo,
            "orgao": orgao,
            "paciente": pid,
            "n_fatias": len(valores),
            "dice_alp_medio": float(alp.mean()),
            "dice_sam_medio": float(sam.mean()),
            "dice_sam_dp_fatias": (
                float(sam.std(ddof=1))
                if len(sam) > 1 else 0.0
            ),
            "delta_dice": float(sam.mean() - alp.mean()),
            "sem_prompt": sum(
                int(r["sam_executado"]) == 0
                for r in valores
            )
        })

    with CSV_PACIENTES.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(pacientes[0])
        )
        writer.writeheader()
        writer.writerows(pacientes)

    geral = []

    print("\n=== DICE MÉDIO ± DP ENTRE PACIENTES ===")

    for passo in CHECKPOINTS:
        for _, orgao in ORGAOS:
            grupo = [
                p for p in pacientes
                if p["checkpoint"] == passo
                and p["orgao"] == orgao
            ]

            assert len(grupo) == 4

            alp = np.array([
                p["dice_alp_medio"] for p in grupo
            ])
            sam = np.array([
                p["dice_sam_medio"] for p in grupo
            ])

            registro = {
                "checkpoint": passo,
                "orgao": orgao,
                "n_pacientes": 4,
                "dice_alp_media": float(alp.mean()),
                "dice_alp_dp": float(alp.std(ddof=1)),
                "dice_sam_media": float(sam.mean()),
                "dice_sam_dp": float(sam.std(ddof=1)),
                "delta_dice": float(sam.mean() - alp.mean()),
                "sem_prompt": sum(p["sem_prompt"] for p in grupo)
            }

            geral.append(registro)

            print(
                f"{orgao} | passo {passo:5d} | "
                f"ALP={registro['dice_alp_media']:.4f} | "
                f"SAM={registro['dice_sam_media']:.4f} ± "
                f"{registro['dice_sam_dp']:.4f} | "
                f"Delta={registro['delta_dice']:+.4f}",
                flush=True
            )

    with CSV_GERAL.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(geral[0])
        )
        writer.writeheader()
        writer.writerows(geral)


def main():
    assert torch.cuda.is_available()
    SAIDA.mkdir(parents=True, exist_ok=True)

    mapa = json.loads(
        (DADOS / "classmap_1.json").read_text()
    )

    # Construir a ordem completa e fixa de avaliação.
    esperadas = [
        (passo, nome, pid, int(z))
        for passo in CHECKPOINTS
        for _, nome in ORGAOS
        for pid in QUERIES
        for z in mapa[nome][str(pid)]
    ]

    assert len(esperadas) == TOTAL
    assert len(set(esperadas)) == TOTAL

    referencia = {}

    with REF_ALP.open(
        newline="", encoding="utf-8"
    ) as f:
        for r in csv.DictReader(f):
            k = chave(r)
            assert k not in referencia
            referencia[k] = r

    assert len(referencia) == TOTAL
    assert set(referencia) == set(esperadas)

    for passo in CHECKPOINTS:
        assert checkpoint(passo).is_file()

    # Retomada: aceitar apenas um prefixo completo
    # da ordem esperada, evitando duplicações.
    anteriores = []

    if CSV_SAM.exists():
        assert CSV_SAM.read_bytes().endswith(b"\n"), (
            "CSV interrompido no meio de uma linha. "
            "Requer verificação antes da retomada."
        )

        with CSV_SAM.open(
            newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CAMPOS
            anteriores = list(reader)

        assert len(anteriores) <= TOTAL

        for indice, registro in enumerate(anteriores):
            assert chave(registro) == esperadas[indice], (
                f"Ordem inconsistente na linha {indice+1}"
            )

    concluidos = len(anteriores)
    print(
        f"Retomada: {concluidos}/{TOTAL} inferências",
        flush=True
    )

    if concluidos == TOTAL:
        resumir(anteriores)
        salvar_json(STATUS, {
            "estado": "concluido",
            "concluidos": TOTAL,
            "total": TOTAL
        })
        return

    # Suporte fixo: paciente 4, fatia central positiva.
    img_s, gt_s = volume(SUPORTE)
    suportes = {}

    for classe, nome in ORGAOS:
        z_positivas = np.flatnonzero(
            np.any(gt_s == classe, axis=(1, 2))
        )

        assert len(z_positivas) > 0
        z = int(z_positivas[len(z_positivas) // 2])

        sup_img = preparar(img_s[z])
        sup_mask = torch.from_numpy(
            (gt_s[z] == classe).astype(np.float32)
        )[None].cuda()

        suportes[classe] = (sup_img, sup_mask)

        print(
            f"Suporte {nome}: paciente 4, fatia {z}",
            flush=True
        )

    cfg = dict(
        json.loads(CFG_SAM.read_text())["sam"]
    )

    cfg["image_size"] = [1024, 1024]
    cfg["alpnet_image_size"] = 256
    cfg["sam_pretrained_path"] = str(
        ROOT / "pretrained_model/sam_vit_h.pth"
    )

    assert cfg["use_bbox"] and cfg["use_points"]
    assert not cfg["use_mask"]

    salvar_json(SAIDA / "config_sam_utilizada.json", cfg)

    print("Construindo VMamba-Tiny + SAM-H...", flush=True)

    modelo = construir_modelo("tiny", repo=ROOT)
    modelo.cuda().eval().requires_grad_(False)

    protosam = ProtoSAM(
        coarse_segmentation_model=ALPNetWrapper(modelo),
        **cfg
    ).to("cuda", dtype=torch.float32)

    protosam.eval().requires_grad_(False)

    if not CSV_SAM.exists():
        with CSV_SAM.open(
            "x", newline="", encoding="utf-8"
        ) as f:
            csv.DictWriter(
                f, fieldnames=CAMPOS
            ).writeheader()

    inicio = time.perf_counter()
    checkpoint_atual = None

    salvar_json(STATUS, {
        "estado": "executando",
        "concluidos": concluidos,
        "total": TOTAL
    })

    with CSV_SAM.open(
        "a", newline="", encoding="utf-8", buffering=1
    ) as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)

        for passo, nome, pid, z in esperadas[concluidos:]:
            if passo != checkpoint_atual:
                cp = torch.load(
                    checkpoint(passo),
                    map_location="cpu",
                    weights_only=False
                )

                assert int(cp["passo"]) == passo
                modelo.load_state_dict(
                    cp["modelo"], strict=True
                )

                del cp
                modelo.eval()
                protosam.eval()
                checkpoint_atual = passo

                print(
                    f"\nCHECKPOINT {passo}",
                    flush=True
                )

            classe = 2 if nome == "RK" else 3
            sup_img, sup_mask = suportes[classe]

            imagens, labels = volume(pid)
            gt = labels[z] == classe
            assert gt.any()

            query = preparar(imagens[z])

            entrada = ALPNetInput(
                support_images=[sup_img],
                support_labels=[sup_mask],
                query_images=query,
                isval=True,
                val_wsize=2,
                show_viz=False,
                supp_fts=None
            )

            with (
                torch.no_grad(),
                torch.autocast("cuda", enabled=False)
            ):
                pred_sam, scores = protosam(
                    query_image=query,
                    coarse_model_input=entrada,
                    degrees_rotate=0
                )

                pred_sam = (
                    pred_sam.detach()
                    .cpu().numpy().squeeze()
                )

            sam_executado = True

            # Mesmo tratamento do avaliador CT para
            # o retorno antecipado sem foreground.
            if pred_sam.shape == (1024, 1024):
                assert not np.any(pred_sam)
                pred_sam = np.zeros(
                    (256, 256), dtype=np.uint8
                )
                sam_executado = False

            assert pred_sam.shape == (256, 256)
            assert np.isin(pred_sam, [0, 1]).all()

            m = metricas(pred_sam.astype(bool), gt)
            ref = referencia[(passo, nome, pid, z)]

            registro = {
                "checkpoint": passo,
                "orgao": nome,
                "paciente": pid,
                "fatia": z,
                "dice_alp": float(ref["dice"]),
                "dice_sam": m["dice"],
                "iou_sam": m["iou"],
                "precisao_sam": m["precisao"],
                "recall_sam": m["recall"],
                "tp_sam": m["tp"],
                "fp_sam": m["fp"],
                "fn_sam": m["fn"],
                "pixels_gt": int(gt.sum()),
                "pixels_sam": m["pixels_pred"],
                "sam_executado": int(sam_executado)
            }

            writer.writerow(registro)
            concluidos += 1

            if concluidos % 10 == 0 or concluidos == TOTAL:
                salvar_json(STATUS, {
                    "estado": "executando",
                    "concluidos": concluidos,
                    "total": TOTAL,
                    "checkpoint": passo,
                    "orgao": nome,
                    "paciente": pid,
                    "tempo_decorrido_s": (
                        time.perf_counter() - inicio
                    )
                })

            if concluidos % 25 == 0:
                print(
                    f"{concluidos}/{TOTAL}",
                    flush=True
                )

            del query, entrada, pred_sam, scores

    with CSV_SAM.open(
        newline="", encoding="utf-8"
    ) as f:
        registros = list(csv.DictReader(f))

    assert len(registros) == TOTAL
    assert [chave(r) for r in registros] == esperadas

    resumir(registros)

    salvar_json(STATUS, {
        "estado": "concluido",
        "concluidos": TOTAL,
        "total": TOTAL,
        "tempo_total_s": time.perf_counter() - inicio
    })

    print("\nAVALIAÇÃO ALP + SAM MRI CONCLUÍDA", flush=True)


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)

    with (SAIDA / "execucao.lock").open("w") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        try:
            main()
        except Exception:
            erro = traceback.format_exc()
            salvar_json(STATUS, {
                "estado": "erro",
                "traceback": erro
            })
            print(erro, flush=True)
            raise
