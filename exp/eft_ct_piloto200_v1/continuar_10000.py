
import os
import sys
import json
import time
import random
import hashlib
import shutil
import traceback
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "exp/eft_ct_piloto200_v1/treinar_ct.py"
ORIGEM = ROOT / "reproducao/eft_ct_felzenszwalb_fold0_piloto200"
RUN = ROOT / "reproducao/eft_ct_felzenszwalb_fold0_10000_v1"
DADOS = ROOT / "data/SABS/sabs_CT_normalized"
AVALIACAO = ORIGEM / "avaliacao_alp_rins_positivas_f0"

os.chdir(ROOT)

spec = importlib.util.spec_from_file_location("eft_ct_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PASSO_INICIAL = 200
TOTAL = 10000
SEMENTE = 1234

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def json_atomico(path, valor):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(valor, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(temp, path)

def processo_avaliacao_ativo():
    arquivo = AVALIACAO / "lancamento.json"
    if not arquivo.exists():
        return False

    pid = json.loads(arquivo.read_text())["pid"]
    proc = Path(f"/proc/{pid}")

    if not proc.exists():
        return False

    try:
        cmd = (proc / "cmdline").read_bytes()
        estado = (proc / "stat").read_text().split(") ")[1][0]
        return b"avaliar_alp_ct.py" in cmd and estado not in ("Z", "X")
    except (OSError, IndexError):
        return False

def aguardar_gpu():
    if processo_avaliacao_ativo():
        print("Aguardando a avaliação ALP liberar a GPU...", flush=True)

    while processo_avaliacao_ativo():
        json_atomico(RUN / "status.json", {
            "estado": "aguardando_avaliacao",
            "passo": 200,
            "total": TOTAL
        })
        time.sleep(10)

    print("GPU liberada pela avaliação anterior.", flush=True)

def construir_plano():
    plano200 = json.loads(
        (ORIGEM / "plano_200.json").read_text()
    )["episodios"]

    assert len(plano200) == 200

    mapa = json.loads(
        (DADOS / "classmap_1.json").read_text()
    )

    utilizados = {
        (int(r["paciente"]), int(r["fatia"]), int(r["regiao"]))
        for r in plano200
    }

    candidatos = []

    for pid in range(7, 30):
        caminho = DADOS / f"superpix-MIDDLE_{pid}.nii.gz"

        volume = sitk.GetArrayFromImage(
            sitk.ReadImage(str(caminho))
        )

        excluidas = (
            set(mapa["KID_R"][str(pid)])
            | set(mapa["KID_l"][str(pid)])
        )

        for z in range(volume.shape[0]):
            if z in excluidas:
                continue

            ids, areas = np.unique(
                volume[z], return_counts=True
            )

            for regiao, area in zip(ids, areas):
                regiao = int(regiao)
                area = int(area)

                if regiao <= 0 or not (400 <= area <= 10000):
                    continue

                if (pid, z, regiao) in utilizados:
                    continue

                candidatos.append({
                    "paciente": pid,
                    "fatia": z,
                    "regiao": regiao,
                    "area": area
                })

    assert len(candidatos) >= 9800

    novos = random.Random(SEMENTE).sample(
        candidatos, 9800
    )

    plano = {
        "metodo": "EFT-CT Felzenszwalb",
        "fold": 0,
        "resolucao": 256,
        "semente": SEMENTE,
        "passos": TOTAL,
        "primeiros_200": str(ORIGEM / "plano_200.json"),
        "episodios": plano200 + novos
    }

    assert len(plano["episodios"]) == TOTAL

    json_atomico(RUN / "plano_10000.json", plano)

    print(
        f"Plano: 200 episódios anteriores + "
        f"{len(novos)} novos.",
        flush=True
    )

    return plano["episodios"]

def salvar(passo, modelo, otimizador):
    estado = {
        "passo": passo,
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all()
    }

    temp = RUN / "checkpoint_latest.pt.tmp"
    latest = RUN / "checkpoint_latest.pt"

    torch.save(estado, temp)
    os.replace(temp, latest)

    if passo % 500 == 0 or passo == TOTAL:
        destino = RUN / f"checkpoint_{passo:05d}.pt"
        temporario = destino.with_suffix(".pt.tmp")
        shutil.copy2(latest, temporario)
        os.replace(temporario, destino)
        print(f"Checkpoint preservado: {passo}", flush=True)

def executar():
    RUN.mkdir(parents=True, exist_ok=True)

    # Preservar a avaliação: não disputar a GPU com ela.
    aguardar_gpu()

    # Impedir reinício acidental sobre resultados existentes.
    assert not (RUN / "passos.jsonl").exists()
    assert not (RUN / "checkpoint_latest.pt").exists()

    episodios = construir_plano()

    origem_ckpt = ORIGEM / "checkpoint_0200.pt"

    config = {
        "dataset": "SABS CT",
        "fold": 0,
        "encoder": "VMamba-Tiny",
        "inicio": "continuação exata do checkpoint 200",
        "checkpoint_origem": str(origem_ckpt),
        "checkpoint_sha256": sha(origem_ckpt),
        "script_base_sha256": sha(BASE),
        "plano_sha256": sha(RUN / "plano_10000.json"),
        "passos_totais": TOTAL,
        "passos_adicionais": 9800,
        "resolucao": 256,
        "otimizador": "AdamW",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "loss": "cross entropy + alinhamento",
        "classes_excluidas": [2, 3],
        "checkpoints_preservados": "500, 1000, ..., 10000"
    }

    json_atomico(RUN / "config.json", config)
    shutil.copy2(BASE, RUN / "treinador_base.py")

    print("Construindo VMamba-Tiny...", flush=True)

    modelo = base.construir_modelo("tiny", repo=ROOT)
    estado_align = base.instalar(modelo)
    modelo.to("cuda")

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    checkpoint = torch.load(
        origem_ckpt,
        map_location="cpu",
        weights_only=False
    )

    assert checkpoint["passo"] == PASSO_INICIAL

    modelo.load_state_dict(
        checkpoint["modelo"],
        strict=True
    )

    otimizador.load_state_dict(
        checkpoint["otimizador"]
    )

    random.setstate(checkpoint["rng_python"])
    np.random.set_state(checkpoint["rng_numpy"])
    torch.set_rng_state(checkpoint["rng_torch"])
    torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])

    del checkpoint

    modelo.train()

    print(
        "Checkpoint 200 restaurado: "
        "pesos, otimizador e RNG.",
        flush=True
    )

    inicio = time.perf_counter()

    with (RUN / "passos.jsonl").open(
        "x", encoding="utf-8", buffering=1
    ) as log:

        for passo in range(201, TOTAL + 1):
            registro = episodios[passo - 1]
            tempo_passo = time.perf_counter()

            support, fg, query, alvo = base.episodio(registro)

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

            # Gradientes conferidos em intervalos regulares.
            if passo == 201 or passo % 100 == 0:
                grads = [
                    p.grad
                    for p in modelo.encoder.parameters()
                    if p.grad is not None
                ]
                assert grads
                assert all(
                    torch.isfinite(g).all().item()
                    for g in grads
                )
                assert any(
                    torch.count_nonzero(g).item() > 0
                    for g in grads
                )

            otimizador.step()

            if passo % 25 == 0:
                torch.cuda.synchronize()

            linha = {
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
                "segundos": time.perf_counter() - tempo_passo
            }

            log.write(json.dumps(linha) + "\n")

            if passo % 25 == 0 or passo == 201:
                json_atomico(RUN / "status.json", {
                    "estado": "executando",
                    "passo": passo,
                    "total": TOTAL,
                    "loss": linha["loss"],
                    "segundos_decorridos": (
                        time.perf_counter() - inicio
                    )
                })

            if passo % 100 == 0:
                salvar(passo, modelo, otimizador)
                print(
                    f"Passo {passo}/{TOTAL} | "
                    f"loss={linha['loss']:.4f}",
                    flush=True
                )

            del retorno, logits, alinhamento
            del ce, loss, support, fg, query, alvo

    json_atomico(RUN / "status.json", {
        "estado": "concluido",
        "passo": TOTAL,
        "total": TOTAL,
        "segundos": time.perf_counter() - inicio
    })

    print("TREINAMENTO 10000 CONCLUÍDO", flush=True)

if __name__ == "__main__":
    try:
        executar()
    except Exception:
        erro = traceback.format_exc()
        print(erro, flush=True)
        RUN.mkdir(parents=True, exist_ok=True)
        json_atomico(
            RUN / "status.json",
            {"estado": "erro", "traceback": erro}
        )
        raise
