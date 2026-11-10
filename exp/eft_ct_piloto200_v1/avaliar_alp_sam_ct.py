
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
DADOS = ROOT / "data/SABS/sabs_CT_normalized"

PILOTO = ROOT / "reproducao/eft_ct_felzenszwalb_fold0_piloto200"
TREINO = ROOT / "reproducao/eft_ct_felzenszwalb_fold0_10000_v1"

REFERENCIA_ALP = (
    TREINO / "avaliacao_alp_rins_positivas_f0_7ckpts"
    / "metricas_fatias.csv"
)

SAIDA = TREINO / "avaliacao_alp_sam_rins_f0_4ckpts"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULOS))

from eft_modelo_672 import construir_modelo
from models.ProtoSAM import ProtoSAM, ALPNetWrapper, ALPNetInput

PASSOS = [0, 2000, 5000, 10000]
ORGAOS = [(2, "KID_R"), (3, "KID_l")]
QUERIES = list(range(6))

torch.backends.cuda.matmul.allow_tf32 = False


def salvar_json(arquivo, obj):
    temp = arquivo.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(temp, arquivo)


def checkpoint(passo):
    if passo == 0:
        return PILOTO / "checkpoint_0000.pt"
    return TREINO / f"checkpoint_{passo:05d}.pt"


def volume(pid):
    img = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"image_{pid}.nii.gz"))
    )
    gt = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"label_{pid}.nii.gz"))
    )
    assert img.shape == gt.shape
    assert img.shape[1:] == (256, 256)
    return img, gt


def preparar(fatia):
    rgb = np.repeat(
        fatia.astype(np.float32)[..., None], 3, axis=2
    )

    t = torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).permute(2, 0, 1)[None].cuda()

    media = t.new_tensor(
        [123.675, 116.28, 103.53]
    )[None, :, None, None]

    desvio = t.new_tensor(
        [58.395, 57.12, 57.375]
    )[None, :, None, None]

    return (t - media) / desvio


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


def main():
    assert torch.cuda.is_available()
    SAIDA.mkdir(parents=True, exist_ok=True)

    destino = SAIDA / "metricas_fatias.csv"
    assert not destino.exists(), "Avaliação já existe."

    for passo in PASSOS:
        assert checkpoint(passo).is_file()

    # Referência ALP já avaliada, nos mesmos pacientes.
    referencia = {}

    with REFERENCIA_ALP.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            chave = (
                int(r["checkpoint"]),
                r["orgao"],
                int(r["paciente"]),
                int(r["fatia"])
            )
            assert chave not in referencia
            referencia[chave] = r

    mapa = json.loads(
        (DADOS / "classmap_1.json").read_text()
    )

    # Carregar suporte fixo para os dois rins.
    img_s, gt_s = volume(6)

    suportes = {}

    for classe, nome in ORGAOS:
        fatias = np.flatnonzero(
            np.any(gt_s == classe, axis=(1, 2))
        )
        assert len(fatias) > 0
        z = int(fatias[len(fatias) // 2])

        sup_img = preparar(img_s[z])
        sup_mask = torch.from_numpy(
            (gt_s[z] == classe).astype(np.float32)
        )[None].cuda()

        suportes[classe] = (sup_img, sup_mask, z)
        print(f"Suporte {nome}: paciente 6, fatia {z}", flush=True)

    # Configuração SAM-H validada no teste anterior.
    cfg_path = (
        ROOT / "reproducao"
        / "eft_tiny_passo_2000_validacao_20261011T221527472147Z"
        / "config.json"
    )

    cfg = dict(json.loads(cfg_path.read_text())["sam"])
    cfg["image_size"] = [1024, 1024]
    cfg["alpnet_image_size"] = 256
    cfg["sam_pretrained_path"] = str(
        ROOT / "pretrained_model/sam_vit_h.pth"
    )

    assert cfg["use_bbox"] and cfg["use_points"]
    assert not cfg["use_mask"]
    assert Path(cfg["sam_pretrained_path"]).is_file()

    print("Construindo VMamba-Tiny e SAM-H...", flush=True)

    modelo = construir_modelo("tiny", repo=ROOT)
    modelo.cuda().eval().requires_grad_(False)

    wrapper = ALPNetWrapper(modelo)

    protosam = ProtoSAM(
        coarse_segmentation_model=wrapper,
        **cfg
    ).to("cuda", dtype=torch.float32)

    protosam.eval().requires_grad_(False)

    total = 425 * len(PASSOS)
    concluidos = 0
    inicio = time.perf_counter()

    campos = [
        "checkpoint", "orgao", "paciente", "fatia",
        "dice_alp", "dice_sam", "iou_sam",
        "precisao_sam", "recall_sam",
        "tp_sam", "fp_sam", "fn_sam",
        "pixels_gt", "pixels_sam"
    ]

    registros = []

    with destino.open(
        "x", newline="", encoding="utf-8", buffering=1
    ) as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()

        for passo in PASSOS:
            ckpt = torch.load(
                checkpoint(passo),
                map_location="cpu",
                weights_only=False
            )

            assert ckpt["passo"] == passo
            modelo.load_state_dict(
                ckpt["modelo"], strict=True
            )
            del ckpt

            modelo.eval()
            protosam.eval()

            print(f"\nCHECKPOINT {passo}", flush=True)

            for classe, nome in ORGAOS:
                sup_img, sup_mask, _ = suportes[classe]

                for pid in QUERIES:
                    imagens, labels = volume(pid)

                    fatias = mapa[nome][str(pid)]

                    for z in fatias:
                        z = int(z)
                        gt = labels[z] == classe
                        assert gt.any()

                        chave = (passo, nome, pid, z)
                        assert chave in referencia, chave

                        query = preparar(imagens[z])

                        def entrada():
                            return ALPNetInput(
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
                                coarse_model_input=entrada(),
                                degrees_rotate=0
                            )

                            pred_sam = (
                                pred_sam.detach()
                                .cpu().numpy().squeeze()
                            )

                        assert pred_sam.shape == (256, 256)
                        assert np.isin(pred_sam, [0, 1]).all()

                        m = metricas(pred_sam.astype(bool), gt)

                        r = {
                            "checkpoint": passo,
                            "orgao": nome,
                            "paciente": pid,
                            "fatia": z,
                            "dice_alp": float(
                                referencia[chave]["dice"]
                            ),
                            "dice_sam": m["dice"],
                            "iou_sam": m["iou"],
                            "precisao_sam": m["precisao"],
                            "recall_sam": m["recall"],
                            "tp_sam": m["tp"],
                            "fp_sam": m["fp"],
                            "fn_sam": m["fn"],
                            "pixels_gt": int(gt.sum()),
                            "pixels_sam": m["pixels_pred"]
                        }

                        escritor.writerow(r)
                        registros.append(r)
                        concluidos += 1

                        if concluidos % 25 == 0:
                            salvar_json(
                                SAIDA / "status.json",
                                {
                                    "estado": "executando",
                                    "concluidos": concluidos,
                                    "total": total,
                                    "checkpoint": passo,
                                    "orgao": nome,
                                    "paciente": pid,
                                    "segundos": time.perf_counter() - inicio
                                }
                            )
                            print(
                                f"{concluidos}/{total}",
                                flush=True
                            )

                        del query, pred_sam, scores

    assert concluidos == total

    # Agregação por paciente.
    grupos = defaultdict(list)

    for r in registros:
        chave = (
            r["checkpoint"],
            r["orgao"],
            r["paciente"]
        )
        grupos[chave].append(r)

    pacientes = []

    for (passo, orgao, pid), itens in sorted(grupos.items()):
        pacientes.append({
            "checkpoint": passo,
            "orgao": orgao,
            "paciente": pid,
            "n_fatias": len(itens),
            "dice_alp": float(np.mean([
                r["dice_alp"] for r in itens
            ])),
            "dice_sam": float(np.mean([
                r["dice_sam"] for r in itens
            ]))
        })

    with (SAIDA / "resumo_pacientes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        escritor = csv.DictWriter(
            f, fieldnames=list(pacientes[0])
        )
        escritor.writeheader()
        escritor.writerows(pacientes)

    # Média e desvio padrão entre os 6 pacientes.
    resumo = []

    print("\n=== RESULTADOS POR ÓRGÃO ===", flush=True)

    for passo in PASSOS:
        for _, orgao in ORGAOS:
            itens = [
                p for p in pacientes
                if p["checkpoint"] == passo
                and p["orgao"] == orgao
            ]
            assert len(itens) == 6

            alp = np.array([p["dice_alp"] for p in itens])
            sam = np.array([p["dice_sam"] for p in itens])

            linha = {
                "checkpoint": passo,
                "orgao": orgao,
                "n_pacientes": 6,
                "dice_alp_media": float(alp.mean()),
                "dice_alp_dp": float(alp.std(ddof=1)),
                "dice_sam_media": float(sam.mean()),
                "dice_sam_dp": float(sam.std(ddof=1)),
                "delta_sam_pp": float(
                    (sam.mean() - alp.mean()) * 100
                )
            }

            resumo.append(linha)

            print(
                f"{orgao} | passo {passo} | "
                f"ALP {alp.mean():.4f} ± {alp.std(ddof=1):.4f} | "
                f"SAM {sam.mean():.4f} ± {sam.std(ddof=1):.4f}",
                flush=True
            )

    with (SAIDA / "resumo_geral.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        escritor = csv.DictWriter(
            f, fieldnames=list(resumo[0])
        )
        escritor.writeheader()
        escritor.writerows(resumo)

    salvar_json(
        SAIDA / "status.json",
        {
            "estado": "concluido",
            "concluidos": concluidos,
            "total": total,
            "segundos": time.perf_counter() - inicio
        }
    )

    print("\nAVALIAÇÃO ALP + SAM CONCLUÍDA", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        erro = traceback.format_exc()
        print(erro, flush=True)

        SAIDA.mkdir(parents=True, exist_ok=True)
        salvar_json(
            SAIDA / "status.json",
            {
                "estado": "erro",
                "traceback": erro
            }
        )
        raise
