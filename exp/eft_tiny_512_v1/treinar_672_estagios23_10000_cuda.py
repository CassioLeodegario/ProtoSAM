from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

CODIGO = Path(__file__).resolve().parent
ROOT = CODIGO.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from eft_modelo_672 import construir_modelo
from eft_alinhamento_seguro import instalar

DADOS = ROOT / "reproducao/eft_sp_dados512_regioes256_20261016T212513467126Z"
H2000 = ROOT / "reproducao/eft_tiny_2000_v2_20261011T204725195685Z"
H5000 = ROOT / "reproducao/eft_tiny_5000_20261011T224215171455Z"
PRESERVAR = set([0, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()

def json_atomico(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)

def rng_atual():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }

def restaurar_rng(r):
    random.setstate(r["python"])
    n = r["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(r["torch"])
    torch.cuda.set_rng_state_all(r["cuda"])

def episodio(registro):
    from PIL import Image

    # Recupera exatamente a região256 que originou o cache512.
    with np.load(DADOS / registro["cache"], allow_pickle=False) as d:
        mask512 = (d["segmentos"] == registro["regiao"]) & d["util"]

    assert mask512.shape == (512, 512)
    mask256 = mask512[::2, ::2].copy()
    assert np.array_equal(
        mask512, mask256.repeat(2, axis=0).repeat(2, axis=1)
    ), "Cache512 não corresponde à ampliação exata da região256."

    # RGB original: não amplia o RGB512 do cache.
    imagem_original = Path(registro["imagem"])
    assert imagem_original.is_file()
    assert sha(imagem_original) == registro["imagem_sha256"]

    with Image.open(imagem_original) as im:
        rgb = np.array(
            im.convert("RGB").resize(
                (672, 672), resample=Image.Resampling.BILINEAR
            )
        )

    mask = np.array(
        Image.fromarray(mask256.astype(np.uint8)).resize(
            (672, 672), resample=Image.Resampling.NEAREST
        )
    ) > 0

    assert rgb.shape == (672, 672, 3)
    assert mask.shape == (672, 672) and mask.any()

    def imagem(a):
        t = torch.from_numpy(np.ascontiguousarray(a))
        t = t.permute(2, 0, 1)[None].to("cuda", dtype=torch.float32)
        media = t.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
        desvio = t.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
        return (t - media) / desvio

    return (
        imagem(rgb),
        torch.from_numpy(mask.copy())[None].to("cuda", dtype=torch.float32),
        imagem(np.fliplr(rgb)),
        torch.from_numpy(np.fliplr(mask).copy())[None].to(
            "cuda", dtype=torch.long
        ),
    )

def preparar_plano():
    assert (DADOS / "preparacao_concluida.json").is_file()
    novos = ler(DADOS / "planos.json")
    antigo2000 = ler(H2000 / "planos.json")
    antigo5000 = ler(H5000 / "planos.json")

    assert len(antigo5000["treino"]) == 5000
    assert antigo5000["treino"][:2000] == antigo2000["treino"], (
        "O plano histórico de 5000 não preserva o de 2000."
    )

    por_indice = {}
    for r in novos["treino"] + novos["validacao"]:
        por_indice.setdefault(r["indice"], r)

    treino_ids = {r["indice"] for r in novos["treino"]}
    val_ids = {r["indice"] for r in novos["validacao"]}
    assert len(treino_ids) == 1304 and len(val_ids) == 145
    assert not treino_ids & val_ids

    plano = []
    for r in antigo5000["treino"]:
        assert r["indice"] in treino_ids
        base = por_indice[r["indice"]]
        assert base["imagem_origem"] == r["imagem"]
        assert base["imagem_sha256"] == r["imagem_sha256"]
        assert base["cache256_sha256"] == r["cache_sha256"]
        assert base["split"] == r["split"] == "treino"
        assert r["regiao"] in base["candidatos"]
        plano.append(dict(base, regiao=r["regiao"]))

    assert plano[:2000] == novos["treino"]
    print("Plano conferido: 5000 episódios históricos, repetidos duas vezes na mesma ordem.", flush=True)

    print("Conferindo integridade dos 1449 caches de 512...", flush=True)
    for i, r in enumerate(por_indice.values(), 1):
        assert sha(DADOS / r["cache"]) == r["cache_sha256"], r["cache"]
        if i % 300 == 0:
            print(f"Caches: {i}/1449", flush=True)

    assert len(plano) == 5000
    plano10k = [dict(r) for r in plano] + [dict(r) for r in plano]
    assert plano10k[:5000] == plano10k[5000:]
    return {"treino": plano10k, "validacao": novos["validacao"]}

def versao(nome):
    try:
        return importlib.metadata.version(nome)
    except importlib.metadata.PackageNotFoundError:
        return None


def configurar_congelamento(modelo):
    encoder = modelo.encoder
    assert len(encoder.layers) == 4

    congelados = (
        encoder.patch_embed,
        encoder.layers[0],
        encoder.layers[1],
    )
    for modulo in congelados:
        modulo.requires_grad_(False)
        modulo.eval()

    for modulo in (encoder.layers[2], encoder.layers[3], encoder.outnorm3):
        assert all(p.requires_grad for p in modulo.parameters())

    parametros_fixos = {
        nome: p for nome, p in modelo.named_parameters()
        if not p.requires_grad
    }
    assert parametros_fixos

    print("Congelados: patch_embed, estagios 0 e 1; modo eval.", flush=True)
    print("Treinaveis: estagios 2 e 3, outnorm3.", flush=True)
    print(
        "Parametros treinaveis:",
        sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        flush=True,
    )
    print(
        "Parametros congelados:",
        sum(p.numel() for p in parametros_fixos.values()),
        flush=True,
    )
    return congelados, parametros_fixos


def main(pasta):
    import selective_scan_cuda
    from models import vmamba
    assert vmamba.WITH_SELECTIVESCAN_MAMBA, 'Backend CUDA indisponivel.'
    assert sha(selective_scan_cuda.__file__) == '48cd62932a7b3ccd5e57eb80c40fcced8e3f35ec7e5ff219fc90d20021961f66', 'Extensao CUDA diferente da validada.'
    assert torch.cuda.is_available()
    def recusar_fallback(*args, **kwargs):
        raise RuntimeError('Fallback Torch proibido neste treino.')
    vmamba.selective_scan_torch = recusar_fallback
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

    plano = preparar_plano()
    fontes = [
        Path(__file__), CODIGO / "eft_modelo_672.py",
        CODIGO / "eft_alinhamento_seguro.py",
        ROOT / "models/vmamba.py",
        ROOT / "models/grid_proto_fewshot.py",
        ROOT / "models/alpmodule.py",
    ]
    pesos = ROOT / "pretrained_model/vmamba_tiny_v2.pth"
    digest_pesos = sha(pesos)
    assert digest_pesos == "09739f6d95638e5caf0d33fcbca85b7cff62b8ca16ec2926d781109939c6b201"

    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    ).stdout.strip()

    config = {
        "formato": 1, "total_passos": 10000, "semente": 1234,
        "experimento": "treino672 estagios23; avaliacao penultima; 10k; WD0.01",
        "repeticao_plano": {
            "episodios_por_ciclo": 5000,
            "ciclos": 2,
            "ordem": "historica, identica nos dois ciclos",
        },
        "decay_aplicado_a": "todos os parametros do otimizador",
        "backend_scan": "selective_scan_cuda",
        "mamba_ssm": versao("mamba-ssm"),
        "extensao_scan": str(Path(selective_scan_cuda.__file__).resolve()),
        "extensao_scan_sha256": sha(selective_scan_cuda.__file__),
        "benchmark_previo": '/home/1542521/cassio_pereira/ProtoSAM-repro/reproducao/eft512_benchmark_cuda_20261017T101300393042Z',
        "benchmark_sha256": 'f7d36e812db75c1bc55f08f97202f4e781a376618ddb1d5795bb488d62bb7fe2',
        "inicio": "pesos pre-treinados; sem checkpoint EFT anterior",
        "checkpoint_origem": str(pesos),
        "checkpoint_origem_sha256": digest_pesos,
        "resolucao_treino": [672, 672], "grade_alp": [32, 32],
        "preparo_672": {
            "rgb": "original RGB -> PIL bilinear 672",
            "mascara": "regiao256 recuperada do cache512 -> nearest 672",
            "superpixels_recalculados": False,
            "aumento": "query por flip horizontal apos resize",
        },
        "batch_episodico": 1, "adaptacao": "estagios 2 e 3 treinaveis; features da ultima camada",
        "congelamento": {
            "modulos": ["patch_embed", "layers.0", "layers.1"],
            "modo": "eval durante treinamento",
            "inclui_downsample_dos_estagios_congelados": True,
            "outnorm3_treinavel": True,
        },
        "otimizador": "AdamW", "lr": 1e-5, "betas": [0.9, 0.999],
        "eps": 1e-8, "weight_decay": 0.01,
        "amp": False, "scheduler": None, "clipping": None,
        "loss": "CE + alinhamento; pesos 1 e 1",
        "alinhamento_seguro": "ativo desde o primeiro passo",
        "dados": str(DADOS),
        "dados_config_sha256": sha(DADOS / "config.json"),
        "plano512_origem_sha256": sha(DADOS / "planos.json"),
        "plano5000_historico_sha256": sha(H5000 / "planos.json"),
        "checkpoint_a_cada": 50, "checkpoints_preservados": sorted(PRESERVAR),
        "avaliacao_polipos": "separada: encoder672/SAM1024",
        "limitacao_retomada": "retomada salva pesos, otimizador e RNG; identidade bit a bit nao garantida; diagnostico anterior de variacao realizado no backend Torch",
        "fontes_sha256": {str(p): sha(p) for p in fontes},
        "ambiente": {
            "python": platform.python_version(),
            "pacotes": {n: versao(n) for n in (
                "torch", "torchvision", "numpy", "Pillow", "timm", "fvcore", "triton"
            )},
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_driver": gpu_info,
            "tf32_matmul": False, "tf32_cudnn": True,
            "cudnn_benchmark": False, "deterministico": False,
        },
    }

    pasta.mkdir(parents=True, exist_ok=True)
    cfg = pasta / "config.json"
    if cfg.exists():
        assert ler(cfg) == config, "Código/configuração/ambiente mudou. Retomada interrompida."
        assert ler(pasta / "planos.json") == plano, "Plano da execução mudou."
    else:
        assert not (pasta / "checkpoint_latest.pt").exists()
        json_atomico(cfg, config)
        json_atomico(pasta / "planos.json", plano)
        (pasta / "fontes").mkdir(exist_ok=True)
        for i, p in enumerate(fontes):
            shutil.copy2(p, pasta / "fontes" / f"{i:02d}_{p.name}")
        shutil.copy2(DADOS / "config.json", pasta / "config_dados.json")

    print("EXECUÇÃO:", pasta, flush=True)
    modelo = construir_modelo("tiny", repo=ROOT)
    modulos_fixos, parametros_fixos = configurar_congelamento(modelo)
    estado_align = instalar(modelo)
    otimizador = torch.optim.AdamW(
        [p for p in modelo.parameters() if p.requires_grad],
        lr=1e-5, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01
    )
    latest = pasta / "checkpoint_latest.pt"

    def preservar(n):
        destino = pasta / f"checkpoint_{n:04d}.pt"
        if n in PRESERVAR and not destino.exists():
            try:
                os.link(latest, destino)
            except OSError:
                tmp = destino.with_suffix(".tmp")
                shutil.copy2(latest, tmp)
                os.replace(tmp, destino)

    def salvar(n):
        tmp = latest.with_suffix(".tmp")
        torch.save({
            "formato": 1, "passo": n,
            "modelo": modelo.state_dict(),
            "otimizador": otimizador.state_dict(),
            "scheduler": None, "rng": rng_atual(), "config": config,
        }, tmp)
        os.replace(tmp, latest)
        preservar(n)

    passo = 0
    if latest.exists():
        salvo = torch.load(latest, map_location="cpu", weights_only=True)
        assert salvo["formato"] == 1 and salvo["config"] == config
        assert salvo["scheduler"] is None
        modelo.load_state_dict(salvo["modelo"], strict=True)
        otimizador.load_state_dict(salvo["otimizador"])
        passo = int(salvo["passo"])
        assert 0 <= passo <= 10000
        restaurar_rng(salvo["rng"])
        del salvo
        preservar(passo)
        print(f"Retomada após passo {passo}.", flush=True)
    else:
        salvar(0)
        print("Início novo: checkpoint 0 salvo, pesos pré-treinados.", flush=True)

    tentativa = time.time_ns()
    log = pasta / f"passos_{tentativa}.jsonl"
    modelo.train()
    for modulo in modulos_fixos:
        modulo.eval()
    assert all(
        not sub.training
        for modulo in modulos_fixos for sub in modulo.modules()
    )
    referencia_fixa = {
        nome: p.detach().cpu().clone()
        for nome, p in parametros_fixos.items()
    }
    torch.cuda.reset_peak_memory_stats()
    inicio_execucao = time.perf_counter()
    passo_inicial = passo

    with log.open("x", encoding="utf-8", buffering=1) as f:
        for n in range(passo + 1, 10001):
            inicio = time.perf_counter()
            r = plano["treino"][n - 1]
            s, fg, q, alvo = episodio(r)
            otimizador.zero_grad(set_to_none=True)
            estado_align["ultimo"] = None

            retorno = modelo(
                [[s]], [[fg]], [[1 - fg]], [q],
                isval=False, val_wsize=2, show_viz=False
            )
            logits, alinhamento = retorno[:2]
            assert logits.shape == (1, 2, 672, 672)
            assert torch.isfinite(logits).all().item(), "Logits não finitos."

            ce = F.cross_entropy(logits, alvo)
            loss = ce + alinhamento
            assert torch.isfinite(loss).item(), "Loss não finita."
            loss.backward()
            assert all(p.grad is None for p in parametros_fixos.values()), (
                'Parametro congelado recebeu gradiente.'
            )
            for indice_estagio in (2, 3):
                assert any(
                    p.grad is not None and p.grad.abs().max().item() > 0
                    for p in modelo.encoder.layers[indice_estagio].parameters()
                ), f'Estagio {indice_estagio} sem gradiente nao zero.'

            grads = [p.grad for p in modelo.encoder.parameters() if p.grad is not None]
            assert grads
            assert torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
            assert torch.stack([g.abs().max() for g in grads]).max().item() > 0

            otimizador.step()
            assert torch.stack([
                torch.isfinite(p).all() for p in modelo.parameters()
            ]).all().item(), "Pesos não finitos."

            linha = {
                "passo": n, "ciclo_plano": (n - 1) // 5000 + 1,
                "posicao_no_ciclo": (n - 1) % 5000 + 1,
                "indice": r["indice"], "imagem": r["imagem"],
                "regiao": r["regiao"], "ce": ce.item(),
                "alinhamento": float(alinhamento.detach())
                    if torch.is_tensor(alinhamento) else float(alinhamento),
                "total": loss.item(),
                "alinhamento_excecao": estado_align["ultimo"],
                "segundos": time.perf_counter() - inicio,
            }
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            del retorno, logits, alinhamento, ce, loss, grads, s, fg, q, alvo
            otimizador.zero_grad(set_to_none=True)

            if n % 50 == 0:
                assert all(
                    torch.equal(p.detach().cpu(), referencia_fixa[nome])
                    for nome, p in parametros_fixos.items()
                ), 'Pesos congelados foram alterados.'
                salvar(n)
                print(f"Checkpoint salvo: {n}", flush=True)

            if n == passo_inicial + 1 or n % 10 == 0:
                media = (time.perf_counter() - inicio_execucao) / (n - passo_inicial)
                restante_h = media * (10000 - n) / 3600
                print(
                    f"{n}/10000 | loss={linha['total']:.6f} | "
                    f"CE={linha['ce']:.6f} | align={linha['alinhamento']:.6f} | "
                    f"{media:.2f}s/passo | restante estimado={restante_h:.2f}h",
                    flush=True
                )

    json_atomico(pasta / "concluido.json", {
        "passos": 10000,
        "checkpoint": str(latest),
        "pico_memoria_alocada_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "avaliacao_polipos_pendente": True,
        "pesos_congelados_inalterados_na_tentativa": True,
    })
    print("TREINAMENTO CONCLUÍDO: 672, estagios 2-3 treinaveis, 10000 passos, WD=0.01.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    pasta = Path(args.run).resolve()
    import fcntl
    pasta.mkdir(parents=True, exist_ok=True)
    trava = (pasta / 'execucao.lock').open('w')
    fcntl.flock(trava, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main(pasta)
    except BaseException:
        if pasta.is_dir():
            (pasta / f"erro_{time.time_ns()}.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
