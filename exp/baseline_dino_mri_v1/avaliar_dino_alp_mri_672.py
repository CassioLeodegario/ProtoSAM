
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
import torch.nn.functional as F
import SimpleITK as sitk

ROOT = Path("/home/1542521/cassio_pereira/ProtoSAM-repro")
DADOS = ROOT / "data/CHAOST2/chaos_MR_T2_normalized"
SAIDA = ROOT / "reproducao/baseline_dino_mri_672_fold0_alp"

DINO_REPO = (
    ROOT / "pretrained_model/hub/facebookresearch_dinov2_main"
)
DINO_PESOS = (
    ROOT / "pretrained_model/hub/checkpoints"
    / "dinov2_vitl14_pretrain.pth"
)

REF_VMAMBA = (
    ROOT / "reproducao/eft_mri_chaost2_fold0_10000_v1"
    / "avaliacao_alp_rins_positivas_f0_7ckpts"
    / "metricas_fatias.csv"
)

CSV_FATIAS = SAIDA / "metricas_fatias.csv"
CSV_PACIENTES = SAIDA / "resumo_pacientes.csv"
CSV_GERAL = SAIDA / "resumo_geral.csv"
STATUS = SAIDA / "status.json"

ORGAOS = [(2, "RK"), (3, "LK")]
QUERIES = [0, 1, 2, 3]
SUPORTE = 4
TOTAL = 112

CAMPOS = [
    "encoder", "orgao", "paciente", "fatia",
    "dice", "iou", "precisao", "recall",
    "tp", "fp", "fn", "pixels_gt", "pixels_pred"
]

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import models.grid_proto_fewshot as modulo_alp

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def salvar_json(caminho, dados):
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(tmp, caminho)


class FewShotSegDinoLocal(modulo_alp.FewShotSeg):
    def get_encoder(self):
        assert self.config["which_model"] == "dinov2_l14"
        assert self.config["lora"] == 0

        self.encoder = torch.hub.load(
            str(DINO_REPO),
            "dinov2_vitl14",
            source="local",
            pretrained=False
        )

        pesos = torch.load(
            DINO_PESOS,
            map_location="cpu",
            weights_only=True
        )

        resultado = self.encoder.load_state_dict(
            pesos, strict=True
        )

        print("Pesos DINO:", resultado, flush=True)
        del pesos

        lado = max(
            self.image_size // 14,
            modulo_alp.DEFAULT_FEATURE_SIZE
        )

        self.config["feature_hw"] = [lado, lado]


CONFIG_ALP = {
    "which_model": "dinov2_l14",
    "cls_name": "grid_proto",
    "proto_grid_size": 8,
    "lora": 0,
    "align": False,
    "debug": False,
    "use_coco_init": False,
    "use_pos_enc": False,
    "use_slice_adapter": False,
    "adapter_layers": 3,
    "dinov2_loss": False
}


@lru_cache(maxsize=5)
def volume(pid):
    imagem = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"image_{pid}.nii.gz"))
    ).astype(np.float32)

    labels = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"label_{pid}.nii.gz"))
    )

    assert imagem.shape == labels.shape
    assert imagem.shape[1:] == (256, 256)
    assert np.isfinite(imagem).all()

    media = float(imagem.mean())
    desvio = float(imagem.std())

    assert desvio > 1e-6

    # Exatamente o protocolo de entrada VMamba-MRI.
    imagem = (imagem - media) / desvio

    return imagem, labels


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

    return F.interpolate(
        tensor,
        size=(672, 672),
        mode="bilinear",
        align_corners=False
    )


def chave(registro):
    return (
        registro["orgao"],
        int(registro["paciente"]),
        int(registro["fatia"])
    )


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
        "pixels_gt": int(gt.sum()),
        "pixels_pred": int(pred.sum())
    }


def resumir(registros):
    grupos = defaultdict(list)

    for registro in registros:
        grupos[
            (registro["orgao"], int(registro["paciente"]))
        ].append(registro)

    pacientes = []

    for (orgao, pid), valores in sorted(grupos.items()):
        dice = np.array([
            float(r["dice"]) for r in valores
        ])

        iou = np.array([
            float(r["iou"]) for r in valores
        ])

        pacientes.append({
            "orgao": orgao,
            "paciente": pid,
            "n_fatias": len(valores),
            "dice_medio": float(dice.mean()),
            "dice_dp_fatias": (
                float(dice.std(ddof=1))
                if len(dice) > 1 else 0.0
            ),
            "iou_medio": float(iou.mean())
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

    print("\n=== RESULTADO DINOv2-L + ALP ===", flush=True)

    for _, orgao in ORGAOS:
        grupo = [
            r for r in pacientes
            if r["orgao"] == orgao
        ]

        assert len(grupo) == 4

        dice = np.array([
            r["dice_medio"] for r in grupo
        ])

        iou = np.array([
            r["iou_medio"] for r in grupo
        ])

        registro = {
            "encoder": "DINOv2-L",
            "orgao": orgao,
            "n_pacientes": 4,
            "dice_medio": float(dice.mean()),
            "dice_dp_pacientes": float(dice.std(ddof=1)),
            "iou_medio": float(iou.mean())
        }

        geral.append(registro)

        print(
            f"{orgao}: Dice "
            f"{registro['dice_medio']:.4f} ± "
            f"{registro['dice_dp_pacientes']:.4f}",
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

    media = np.mean([r["dice_medio"] for r in geral])

    print(
        f"Média dos dois rins: {media:.4f}",
        flush=True
    )


def main():
    assert torch.cuda.is_available()
    assert DINO_PESOS.is_file()
    assert (DINO_REPO / "hubconf.py").is_file()

    SAIDA.mkdir(parents=True, exist_ok=True)

    mapa = json.loads(
        (DADOS / "classmap_1.json").read_text(
            encoding="utf-8"
        )
    )

    # Ordem idêntica à avaliação anterior.
    esperadas = [
        (nome, pid, int(z))
        for _, nome in ORGAOS
        for pid in QUERIES
        for z in mapa[nome][str(pid)]
    ]

    assert len(esperadas) == TOTAL
    assert len(set(esperadas)) == TOTAL

    # Verificar os mesmos casos do VMamba (checkpoint 0).
    referencia = {}

    with REF_VMAMBA.open(
        newline="", encoding="utf-8"
    ) as f:
        for r in csv.DictReader(f):
            if int(r["checkpoint"]) == 0:
                k = chave(r)
                assert k not in referencia
                referencia[k] = r

    assert len(referencia) == TOTAL
    assert set(referencia) == set(esperadas)

    # Permitir retomada sem repetir inferências concluídas.
    anteriores = []

    if CSV_FATIAS.exists():
        assert CSV_FATIAS.read_bytes().endswith(b"\n")

        with CSV_FATIAS.open(
            newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == CAMPOS
            anteriores = list(reader)

        assert len(anteriores) <= TOTAL

        for indice, registro in enumerate(anteriores):
            assert chave(registro) == esperadas[indice], (
                f"Histórico inconsistente na linha {indice+1}"
            )

    concluidos = len(anteriores)

    print(
        f"Retomada: {concluidos}/{TOTAL}",
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

    print("Carregando DINOv2-L...", flush=True)

    modelo = FewShotSegDinoLocal(
        image_size=672,
        pretrained_path=None,
        cfg=CONFIG_ALP.copy()
    ).to("cuda", dtype=torch.float32)

    modelo.eval().requires_grad_(False)

    assert not any(p.requires_grad for p in modelo.parameters())
    assert modelo.config["feature_hw"] == [48, 48]

    # Mesmo suporte one-shot do VMamba.
    imagens_s, labels_s = volume(SUPORTE)
    suportes = {}

    for classe, nome in ORGAOS:
        positivas = np.flatnonzero(
            np.any(labels_s == classe, axis=(1, 2))
        )

        assert len(positivas) > 0

        z = int(positivas[len(positivas) // 2])

        sup_img = preparar(imagens_s[z])
        sup_mask_256 = torch.from_numpy(
            (labels_s[z] == classe).astype(np.float32)
        )[None, None].to("cuda")

        sup_mask = F.interpolate(
            sup_mask_256,
            size=(672, 672),
            mode="nearest"
        )[:, 0]

        assert sup_mask.shape == (1, 672, 672)
        assert torch.all(
            (sup_mask == 0) | (sup_mask == 1)
        ).item()

        suportes[classe] = (sup_img, sup_mask)

        print(
            f"Suporte {nome}: paciente {SUPORTE}, fatia {z}",
            flush=True
        )

    if not CSV_FATIAS.exists():
        with CSV_FATIAS.open(
            "x", newline="", encoding="utf-8"
        ) as f:
            csv.DictWriter(
                f, fieldnames=CAMPOS
            ).writeheader()

    inicio = time.perf_counter()

    salvar_json(STATUS, {
        "estado": "executando",
        "concluidos": concluidos,
        "total": TOTAL
    })

    with CSV_FATIAS.open(
        "a", newline="", encoding="utf-8", buffering=1
    ) as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)

        for nome, pid, z in esperadas[concluidos:]:
            classe = 2 if nome == "RK" else 3
            sup_img, sup_mask = suportes[classe]

            imagens, labels = volume(pid)

            query = preparar(imagens[z])
            gt = labels[z] == classe

            assert gt.any()
            assert int(gt.sum()) == int(
                referencia[(nome, pid, z)]["pixels_gt"]
            )

            with torch.inference_mode():
                retorno = modelo(
                    [[sup_img]],
                    [[sup_mask]],
                    [[1 - sup_mask]],
                    [query],
                    isval=True,
                    val_wsize=2,
                    show_viz=False
                )

                logits = retorno[0]

                assert logits.shape == (1, 2, 672, 672)
                assert torch.isfinite(logits).all().item()

                logits_256 = F.interpolate(
                    logits,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False
                )

                pred = (
                    logits_256.argmax(dim=1)[0]
                    .cpu().numpy() == 1
                )

            registro = {
                "encoder": "DINOv2-L",
                "orgao": nome,
                "paciente": pid,
                "fatia": z,
                **metricas(pred, gt)
            }

            writer.writerow(registro)
            concluidos += 1

            if concluidos % 5 == 0 or concluidos == TOTAL:
                salvar_json(STATUS, {
                    "estado": "executando",
                    "concluidos": concluidos,
                    "total": TOTAL,
                    "orgao": nome,
                    "paciente": pid,
                    "tempo_decorrido_s": (
                        time.perf_counter() - inicio
                    )
                })

            if concluidos % 10 == 0:
                print(
                    f"{concluidos}/{TOTAL}",
                    flush=True
                )

            del query, retorno, logits

    with CSV_FATIAS.open(
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

    print("AVALIAÇÃO DINO + ALP CONCLUÍDA", flush=True)


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
