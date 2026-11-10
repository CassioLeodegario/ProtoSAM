
import os
import sys
import time
import json
import random
import hashlib
import shutil
import traceback
import importlib.util
import fcntl

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/1542521/cassio_pereira/ProtoSAM-repro")
EXP = ROOT / "exp/eft_mri_chaos_v1"
PILOTO = ROOT / "reproducao/eft_mri_chaost2_fold0_piloto200"
RUN = ROOT / "reproducao/eft_mri_chaost2_fold0_10000_v1"

BASE = EXP / "treinar_mri_200.py"
PLANO = PILOTO / "plano_10000.json"
ORIGEM = PILOTO / "checkpoint_0200.pt"

LATEST = RUN / "checkpoint_latest.pt"
STATUS = RUN / "status.json"
HISTORICO = RUN / "passos.jsonl"
CONFIG = RUN / "config_continuacao.json"

TOTAL = 10000
INICIAL = 200
INTERVALO_RECUPERACAO = 50
INTERVALO_HISTORICO = 500

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sha256(caminho):
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(bloco)
    return h.hexdigest()


def json_atomico(caminho, dados):
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(tmp, caminho)


def preservar_checkpoint(passo):
    destino = RUN / f"checkpoint_{passo:05d}.pt"

    if destino.exists():
        return

    tmp = destino.with_name(destino.name + ".tmp")
    shutil.copyfile(LATEST, tmp)
    os.replace(tmp, destino)

    print(
        f"Checkpoint histórico preservado: {passo}",
        flush=True
    )


def salvar_checkpoint(passo, modelo, otimizador, procedencia):
    estado = {
        "passo": passo,
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "procedencia": procedencia
    }

    tmp = LATEST.with_name(LATEST.name + ".tmp")
    torch.save(estado, tmp)
    os.replace(tmp, LATEST)

    if passo % INTERVALO_HISTORICO == 0:
        preservar_checkpoint(passo)

    print(f"Checkpoint de recuperação: {passo}", flush=True)


def ajustar_historico(passo_restaurado):
    # Se houve interrupção entre dois checkpoints,
    # descartar registros posteriores ao estado
    # efetivamente restaurado.
    if not HISTORICO.exists():
        assert passo_restaurado == INICIAL
        return

    linhas_validas = []
    ultimo = INICIAL

    for linha in HISTORICO.read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            registro = json.loads(linha)
        except json.JSONDecodeError:
            continue

        passo = int(registro["passo"])

        if passo > passo_restaurado:
            break

        assert passo == ultimo + 1, (
            f"Histórico descontínuo: {ultimo} -> {passo}"
        )

        linhas_validas.append(
            json.dumps(registro, ensure_ascii=False)
        )
        ultimo = passo

    assert ultimo == passo_restaurado, (
        f"Histórico termina em {ultimo}, "
        f"checkpoint está no passo {passo_restaurado}"
    )

    tmp = HISTORICO.with_name(HISTORICO.name + ".tmp")
    tmp.write_text(
        "\n".join(linhas_validas)
        + ("\n" if linhas_validas else ""),
        encoding="utf-8"
    )
    os.replace(tmp, HISTORICO)


def executar():
    assert torch.cuda.is_available()
    assert ORIGEM.is_file()
    assert BASE.is_file()

    plano = json.loads(
        PLANO.read_text(encoding="utf-8")
    )

    episodios = plano["episodios"]

    assert len(episodios) == TOTAL
    assert plano["fold"] == 0
    assert plano["classes_excluidas"] == [2, 3]
    assert plano["resolucao"] == 256

    for i, ep in enumerate(episodios):
        assert int(ep["passo"]) == i + 1
        assert int(ep["paciente"]) in range(5, 20)
        assert int(ep["area"]) >= 400

    procedencia = {
        "checkpoint_origem": str(ORIGEM),
        "checkpoint_origem_sha256": sha256(ORIGEM),
        "plano_sha256": sha256(PLANO),
        "script_piloto_sha256": sha256(BASE)
    }

    config = {
        "dataset": "CHAOST2",
        "modalidade": "MRI T2-SPIR",
        "encoder": "VMamba-Tiny",
        "fold": 0,
        "passo_inicial": INICIAL,
        "passo_final": TOTAL,
        "normalizacao": "z-score por volume",
        "loss": "CE + alinhamento",
        "otimizador": "AdamW",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "checkpoint_recuperacao": INTERVALO_RECUPERACAO,
        "checkpoint_historico": INTERVALO_HISTORICO,
        "procedencia": procedencia
    }

    if CONFIG.exists():
        assert json.loads(
            CONFIG.read_text(encoding="utf-8")
        ) == config, "Configuração de retomada diferente."
    else:
        json_atomico(CONFIG, config)

    spec = importlib.util.spec_from_file_location(
        "eft_mri_base_continuacao", BASE
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

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

    # Retomar do checkpoint mais recente, quando existir.
    fonte = LATEST if LATEST.exists() else ORIGEM

    checkpoint = torch.load(
        fonte,
        map_location="cpu",
        weights_only=False
    )

    passo_inicial = int(checkpoint["passo"])

    assert INICIAL <= passo_inicial <= TOTAL

    if fonte == LATEST:
        assert checkpoint["procedencia"] == procedencia
    else:
        assert passo_inicial == INICIAL

    modelo.load_state_dict(
        checkpoint["modelo"], strict=True
    )

    otimizador.load_state_dict(
        checkpoint["otimizador"]
    )

    # Reconstituir os geradores aleatórios do
    # checkpoint, sem reiniciar a semente.
    random.setstate(checkpoint["rng_python"])
    np.random.set_state(checkpoint["rng_numpy"])
    torch.set_rng_state(checkpoint["rng_torch"])
    torch.cuda.set_rng_state_all(
        checkpoint["rng_cuda"]
    )

    del checkpoint

    modelo.train()

    ajustar_historico(passo_inicial)

    if (
        passo_inicial % INTERVALO_HISTORICO == 0
        and fonte == LATEST
    ):
        preservar_checkpoint(passo_inicial)

    print(
        f"Checkpoint restaurado: passo {passo_inicial}",
        flush=True
    )
    print(
        f"Próximo passo: {passo_inicial + 1}",
        flush=True
    )
    print("Meta: 10000 passos", flush=True)

    inicio_total = time.time()

    json_atomico(STATUS, {
        "estado": "executando",
        "passo": passo_inicial,
        "total": TOTAL,
        "checkpoint_origem": str(fonte)
    })

    with HISTORICO.open(
        "a", encoding="utf-8", buffering=1
    ) as log:

        for passo in range(passo_inicial + 1, TOTAL + 1):
            inicio = time.time()
            ep = episodios[passo - 1]

            support, fg, query, alvo = base.episodio(ep)

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
                p.grad
                for p in modelo.encoder.parameters()
                if p.grad is not None
            ]

            assert grads
            assert any(
                torch.count_nonzero(g).item() > 0
                for g in grads
            )
            assert all(
                torch.isfinite(g).all().item()
                for g in grads
            )

            otimizador.step()
            torch.cuda.synchronize()

            registro = {
                "passo": passo,
                "paciente": ep["paciente"],
                "fatia": ep["fatia"],
                "regiao": ep["regiao"],
                "ce": float(ce.detach()),
                "alinhamento": float(
                    alinhamento.detach()
                    if torch.is_tensor(alinhamento)
                    else alinhamento
                ),
                "loss": float(loss.detach()),
                "segundos": time.time() - inicio,
                "alinhamento_excecao": estado_align["ultimo"]
            }

            log.write(json.dumps(registro) + "\n")

            if passo % 10 == 0:
                print(
                    f"Passo {passo}/{TOTAL} | "
                    f"CE={registro['ce']:.4f} | "
                    f"Align={registro['alinhamento']:.4f} | "
                    f"Loss={registro['loss']:.4f}",
                    flush=True
                )

            if passo % INTERVALO_RECUPERACAO == 0:
                salvar_checkpoint(
                    passo, modelo, otimizador, procedencia
                )

            if passo % 10 == 0 or passo == TOTAL:
                json_atomico(STATUS, {
                    "estado": "executando",
                    "passo": passo,
                    "total": TOTAL,
                    "loss": registro["loss"],
                    "ce": registro["ce"],
                    "alinhamento": registro["alinhamento"],
                    "tempo_decorrido_s": time.time() - inicio_total
                })

    assert LATEST.exists()

    json_atomico(STATUS, {
        "estado": "concluido",
        "passo": TOTAL,
        "total": TOTAL,
        "tempo_decorrido_s": time.time() - inicio_total
    })

    print("EFT-MRI 10000 PASSOS CONCLUÍDO", flush=True)


if __name__ == "__main__":
    # Bloqueio para impedir dois processos
    # de treinamento na mesma execução.
    RUN.mkdir(parents=True, exist_ok=True)

    with (RUN / "execucao.lock").open("w") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Outra execução EFT-MRI já está ativa."
            )

        try:
            executar()
        except Exception:
            erro = traceback.format_exc()

            json_atomico(STATUS, {
                "estado": "erro",
                "traceback": erro
            })

            print(erro, flush=True)
            raise
