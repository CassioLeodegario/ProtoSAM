
import os
import sys
import json
import time
import random
import hashlib
import traceback
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
MODULOS = ROOT / "exp/eft_tiny_512_v1"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULOS))

from eft_modelo_672 import construir_modelo
from eft_alinhamento_seguro import instalar

DADOS = ROOT / "data/CHAOST2/chaos_MR_T2_normalized"
RUN = ROOT / "reproducao/eft_mri_chaost2_fold0_piloto200"
PLANO = RUN / "plano_200.json"

SEMENTE = 1234
PASSOS = 200
LR = 1e-5
WD = 0.01

MEDIA = [123.675, 116.28, 103.53]
DESVIO = [58.395, 57.12, 57.375]


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def salvar_json(caminho, dados):
    temp = caminho.with_suffix(caminho.suffix + ".tmp")
    temp.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False)
    )
    os.replace(temp, caminho)


@lru_cache(maxsize=4)
def carregar_volume(pid):
    imagem = sitk.GetArrayFromImage(
        sitk.ReadImage(str(DADOS / f"image_{pid}.nii.gz"))
    )
    regioes = sitk.GetArrayFromImage(
        sitk.ReadImage(str(
            DADOS / f"superpix-MIDDLE_{pid}.nii.gz"
        ))
    )

    assert imagem.shape == regioes.shape
    assert imagem.shape[1:] == (256, 256)

    # MRI: normalização pela média e desvio do volume.
    imagem = imagem.astype(np.float32)
    media_volume = float(imagem.mean())
    desvio_volume = float(imagem.std())

    assert np.isfinite(media_volume)
    assert np.isfinite(desvio_volume)
    assert desvio_volume > 1e-6

    imagem = (imagem - media_volume) / desvio_volume

    return imagem, regioes


def preparar_imagem(array):
    t = torch.from_numpy(
        np.ascontiguousarray(array)
    ).permute(2, 0, 1)[None].to(
        "cuda", dtype=torch.float32
    )

    media = t.new_tensor(MEDIA)[None, :, None, None]
    desvio = t.new_tensor(DESVIO)[None, :, None, None]

    return t  # Entrada MRI ja normalizada por volume


def episodio(registro):
    pid = int(registro["paciente"])
    z = int(registro["fatia"])
    regiao = int(registro["regiao"])

    imagem, regioes = carregar_volume(pid)

    fatia = imagem[z].astype(np.float32)
    mascara = regioes[z] == regiao

    assert mascara.any()
    assert int(mascara.sum()) == int(registro["area"])

    rgb = np.repeat(fatia[..., None], 3, axis=2)

    support = preparar_imagem(rgb)
    query = preparar_imagem(np.fliplr(rgb))

    fg = torch.from_numpy(
        mascara.copy()
    )[None].to("cuda", dtype=torch.float32)

    alvo = torch.from_numpy(
        np.fliplr(mascara).copy()
    )[None].to("cuda", dtype=torch.long)

    return support, fg, query, alvo


def salvar_checkpoint(passo, modelo, otimizador):
    estado = {
        "passo": passo,
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }

    destino = RUN / f"checkpoint_{passo:04d}.pt"
    temp = RUN / f"checkpoint_{passo:04d}.pt.tmp"

    torch.save(estado, temp)
    os.replace(temp, destino)

    print(f"Checkpoint salvo: {passo}", flush=True)


def treinar():
    assert torch.cuda.is_available()

    if (RUN / "checkpoint_0000.pt").exists():
        raise RuntimeError(
            "Já existe um treinamento iniciado. "
            "Não iniciar novamente."
        )

    plano = json.loads(PLANO.read_text())
    episodios = plano["episodios"]

    assert len(episodios) == PASSOS
    assert plano["fold"] == 0
    assert plano["classes_excluidas"] == [2, 3]
    assert plano["resolucao"] == 256

    classmap = json.loads(
        (DADOS / "classmap_1.json").read_text()
    )

    for ep in episodios:
        pid = int(ep["paciente"])
        z = int(ep["fatia"])
        assert pid in range(5, 20)
        assert z not in classmap["RK"][str(pid)]
        assert z not in classmap["LK"][str(pid)]

    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    pesos = ROOT / "pretrained_model/vmamba_tiny_v2.pth"
    assert pesos.is_file()

    config = {
        "dataset": "CHAOST2",
        "modalidade": "MRI T2-SPIR",
        "fold": 0,
        "encoder": "VMamba-Tiny",
        "inicializacao": "pretrained",
        "pesos": str(pesos),
        "pesos_sha256": sha256(pesos),
        "plano_sha256": sha256(PLANO),
        "semente": SEMENTE,
        "passos": PASSOS,
        "resolucao": 256,
        "classes_excluidas": [2, 3],
        "loss": "CE + alinhamento",
        "otimizador": "AdamW",
        "lr": LR,
        "weight_decay": WD,
        "checkpoint_passos": [0, 50, 100, 200],
    }

    salvar_json(RUN / "config_treino.json", config)

    print("Construindo VMamba-Tiny...", flush=True)

    modelo = construir_modelo("tiny", repo=ROOT)
    estado_align = instalar(modelo)
    modelo.to("cuda")
    modelo.train()

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WD
    )

    salvar_checkpoint(0, modelo, otimizador)

    inicio_total = time.time()
    log_path = RUN / "passos.jsonl"

    with log_path.open("x", buffering=1) as log:
        for passo, registro in enumerate(episodios, 1):
            inicio = time.time()

            support, fg, query, alvo = episodio(registro)

            otimizador.zero_grad(set_to_none=True)
            estado_align["ultimo"] = None

            retorno = modelo(
                [[support]],
                [[fg]],
                [[1 - fg]],
                [query],
                isval=False,
                val_wsize=2,
                show_viz=False
            )

            logits, alinhamento = retorno[:2]

            assert logits.shape == (1, 2, 256, 256)

            ce = F.cross_entropy(logits, alvo)
            loss = ce + alinhamento

            assert torch.isfinite(loss).item()

            loss.backward()

            grads = [
                p.grad for p in modelo.encoder.parameters()
                if p.grad is not None
            ]

            assert grads
            assert any(torch.count_nonzero(g).item() > 0 for g in grads)
            assert all(torch.isfinite(g).all().item() for g in grads)

            otimizador.step()

            torch.cuda.synchronize()

            dados_passo = {
                "passo": passo,
                "paciente": registro["paciente"],
                "fatia": registro["fatia"],
                "regiao": registro["regiao"],
                "ce": float(ce.detach()),
                "alinhamento": float(
                    alinhamento.detach()
                    if torch.is_tensor(alinhamento)
                    else alinhamento
                ),
                "loss": float(loss.detach()),
                "segundos": time.time() - inicio,
                "alinhamento_excecao": estado_align["ultimo"],
            }

            log.write(json.dumps(dados_passo) + "\n")

            if passo % 10 == 0 or passo == 1:
                print(
                    f"Passo {passo}/{PASSOS} | "
                    f"CE={dados_passo['ce']:.4f} | "
                    f"Align={dados_passo['alinhamento']:.4f} | "
                    f"Loss={dados_passo['loss']:.4f}",
                    flush=True
                )

            if passo in (50, 100, 200):
                salvar_checkpoint(passo, modelo, otimizador)

            if passo % 10 == 0 or passo == 1:
                salvar_json(RUN / "status.json", {
                    "estado": "executando",
                    "passo": passo,
                    "total": PASSOS,
                    "loss": dados_passo["loss"],
                    "tempo_decorrido_s": time.time() - inicio_total
                })

    salvar_json(RUN / "status.json", {
        "estado": "concluido",
        "passo": PASSOS,
        "total": PASSOS,
        "tempo_total_s": time.time() - inicio_total
    })

    print("TREINAMENTO CONCLUÍDO", flush=True)


if __name__ == "__main__":
    try:
        treinar()
    except Exception:
        erro = traceback.format_exc()
        print(erro, flush=True)

        salvar_json(RUN / "status.json", {
            "estado": "erro",
            "traceback": erro
        })
        raise
