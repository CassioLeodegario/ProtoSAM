"""EFT-SP VMamba: diagnóstico compartilhado, versão 1.
Adaptação experimental. Não é uma receita de treinamento definitivo.
Forward e alignLoss são herdados do ALP instalado no repositório.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import platform
import random
import shutil
import subprocess
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from models.vmamba import Backbone_VSSM
from models.grid_proto_fewshot import FewShotSeg, MultiProtoAsConv

VARIANTES = {
    "tiny": {"depths": [2, 2, 5, 2], "dims": 96, "drop_path_rate": 0.2},
    "small": {"depths": [2, 2, 15, 2], "dims": 96, "drop_path_rate": 0.3},
    "base": {"depths": [2, 2, 15, 2], "dims": 128, "drop_path_rate": 0.6},
}
DIR_EPISODIOS = Path(
    "/workspace/ProtoSAM-repro/reproducao/"
    "eft_sp_episodios_inspecao_20261011T144630000218Z"
)
VMAMBA_SHA256 = "572207a2b6553ff85d81b602eebda2041623073f01fc0c6d1280c5a2add040f6"

def sha256(caminho):
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()

def construir_modelo(variante, repo=Path("/workspace/ProtoSAM-repro")):
    repo = Path(repo).resolve()
    if variante not in VARIANTES:
        raise ValueError(f"Variante desconhecida: {variante}")
    assert torch.cuda.is_available(), "CUDA indisponível."
    assert Path(inspect.getfile(FewShotSeg)).resolve() == repo / "models/grid_proto_fewshot.py"
    assert Path(inspect.getfile(Backbone_VSSM)).resolve() == repo / "models/vmamba.py"
    assert sha256(repo / "models/vmamba.py") == VMAMBA_SHA256, "VMamba difere da versão registrada."
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    variante_cfg = VARIANTES[variante]
    canais = variante_cfg["dims"] * 8
    CHECKPOINT = repo / "pretrained_model" / f"vmamba_{variante}_v2.pth"
    # Encoder: configurações registradas no handoff e no notebook validado.
    encoder = Backbone_VSSM(
        out_indices=(3,),
        pretrained=None,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=variante_cfg["depths"],
        dims=variante_cfg["dims"],
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        drop_path_rate=variante_cfg["drop_path_rate"],
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    pesos = dict(checkpoint["model"])

    # Recupera a normalização aprendida; descarta apenas a cabeça.
    for sufixo in ("weight", "bias"):
        pesos[f"outnorm3.{sufixo}"] = pesos.pop(
            f"classifier.norm.{sufixo}"
        )
        pesos.pop(f"classifier.head.{sufixo}")

    print(encoder.load_state_dict(pesos, strict=True))
    del checkpoint, pesos


    # 2. Adaptação da extração; forward e alignLoss oficiais preservados.
    class FewShotSegVMambaDiagnostico(FewShotSeg):
        def get_encoder(self):
            self.encoder = encoder
            self.config["feature_hw"] = [32, 32]

        def get_cls(self):
            self.cls_unit = MultiProtoAsConv(
                proto_grid=[8, 8],
                feature_hw=[32, 32],
                embed_dim=canais,
            )

        def get_features(self, imgs_concat):
            imagens = F.interpolate(
                imgs_concat,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            features = self.encoder(imagens)[-1]
            assert features.shape[1:] == (canais, 8, 8)

            return F.interpolate(
                features,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            )


    modelo = FewShotSegVMambaDiagnostico(
        image_size=256,
        pretrained_path=None,
        cfg={
            "align": True,
            "debug": False,
            "cls_name": "grid_proto",
            "proto_grid_size": 8,
            "feature_hw": [32, 32],
            "which_model": "vmamba_" + variante,
            "lora": 0,
        },
    ).to("cuda", dtype=torch.float32)

    assert type(modelo).forward is FewShotSeg.forward
    assert type(modelo).alignLoss is FewShotSeg.alignLoss

    modelo.train().requires_grad_(True)
    modelo.zero_grad(set_to_none=True)



    modelo.variante_eft = variante
    modelo.repo_eft = str(repo)
    modelo.checkpoint_eft = str(CHECKPOINT)
    return modelo


def testar_otimizacao(modelo, pasta_episodios=DIR_EPISODIOS):
    pasta_episodios = Path(pasta_episodios)
    arquivos = sorted(pasta_episodios.glob("episodio_*.npz"))
    assert len(arquivos) == 6, f"Esperados 6 episódios; encontrados {len(arquivos)}."

    pasta_saida = (Path(modelo.repo_eft) / "reproducao") / (
        f"eft_{modelo.variante_eft}_diagnostico_v1_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    pasta_saida.mkdir(parents=True, exist_ok=False)

    config = {
        "finalidade": "Diagnóstico de 12 atualizações; não é treinamento definitivo",
        "encoder": "vmamba_" + modelo.variante_eft,
        "semente": 1234,
        "episodios": [str(p) for p in arquivos],
        "atualizacoes": 12,
        "episodios_por_atualizacao": 1,
        "otimizador": "AdamW",
        "lr": 1e-5,
        "weight_decay": 0.0,
        "loss": "CE sem pesos + alinhamento oficial, coeficientes 1 e 1",
        "adaptacao": "Todos os parâmetros habilitados, apenas neste diagnóstico",
        "precisao": "float32, sem autocast",
        "restaurar_pesos_ao_final": True,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    config.update({
        "backbone": VARIANTES[modelo.variante_eft],
        "estagio": 3,
        "entrada_encoder": [256, 256],
        "grade_nativa": [8, 8],
        "grade_alp": [32, 32],
        "kernel_alp": [4, 4],
        "normalizacao": "classifier.norm transferida para outnorm3",
        "media_rgb": [123.675, 116.28, 103.53],
        "desvio_rgb": [58.395, 57.12, 57.375],
        "checkpoint": modelo.checkpoint_eft,
        "checkpoint_sha256": sha256(modelo.checkpoint_eft),
        "episodios_sha256": {p.name: sha256(p) for p in arquivos},
        "python": platform.python_version(),
        "cuda_pytorch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sam_no_treino": False,
        "mascaras_manuais_utilizadas": False,
    })
    fontes = [Path(__file__), Path(inspect.getfile(FewShotSeg)),
              Path(inspect.getfile(Backbone_VSSM)),
              Path(inspect.getfile(MultiProtoAsConv))]
    fontes = list(dict.fromkeys(p.resolve() for p in fontes))
    (pasta_saida / "fontes").mkdir()
    config["fontes_sha256"] = {str(p): sha256(p) for p in fontes}
    for i, p in enumerate(fontes):
        shutil.copy2(p, pasta_saida / "fontes" / f"{i:02d}_{p.name}")
    shutil.copy2(pasta_episodios / "config.json", pasta_saida / "config_episodios.json")
    for nome, comando in [
        ("gpu.txt", ["nvidia-smi"]),
        ("git_head.txt", ["git", "-C", modelo.repo_eft, "rev-parse", "HEAD"]),
        ("git_status.txt", ["git", "-C", modelo.repo_eft, "status", "--short"]),
    ]:
        try:
            resultado = subprocess.run(comando, capture_output=True, text=True, check=False)
            conteudo = resultado.stdout + resultado.stderr
        except OSError as erro:
            conteudo = repr(erro)
        (pasta_saida / nome).write_text(conteudo, encoding="utf-8")

    (pasta_saida / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = next(modelo.parameters()).device
    assert device.type == "cuda"
    assert all(p.requires_grad for p in modelo.parameters())

    # Cópia independente em RAM: permite desfazer as atualizações do diagnóstico.
    estado_inicial = {
        nome: tensor.detach().cpu().clone()
        for nome, tensor in modelo.state_dict().items()
    }
    modo_anterior = modelo.training

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    media = torch.tensor(
        [123.675, 116.28, 103.53], device=device
    ).view(1, 3, 1, 1)
    desvio = torch.tensor(
        [58.395, 57.12, 57.375], device=device
    ).view(1, 3, 1, 1)

    def carregar_episodio_teste(caminho):
        with np.load(caminho, allow_pickle=False) as dados:
            def imagem(chave):
                x = torch.from_numpy(dados[chave].copy())
                x = x.permute(2, 0, 1).unsqueeze(0)
                return (x.to(device=device, dtype=torch.float32) - media) / desvio

            support = imagem("support_rgb")
            query = imagem("query_rgb")
            fg = torch.from_numpy(dados["support_mask"].copy())
            fg = fg.unsqueeze(0).to(device=device, dtype=torch.float32)
            alvo = torch.from_numpy(dados["query_mask"].copy())
            alvo = alvo.unsqueeze(0).to(device=device, dtype=torch.long)

        assert set(torch.unique(fg).tolist()).issubset({0.0, 1.0})
        assert set(torch.unique(alvo).tolist()).issubset({0, 1})
        return support, fg, query, alvo

    def executar_teste_curto():
        otimizador = torch.optim.AdamW(
            modelo.parameters(), lr=config["lr"], weight_decay=0.0
        )
        historico = []
        modelo.train()

        try:
            for passo in range(1, config["atualizacoes"] + 1):
                caminho = arquivos[(passo - 1) % len(arquivos)]
                support, fg, query, alvo = carregar_episodio_teste(caminho)
                otimizador.zero_grad(set_to_none=True)

                retorno = modelo(
                    [[support]], [[fg]], [[1.0 - fg]], [query],
                    isval=False, val_wsize=2, show_viz=False,
                )
                logits, alinhamento = retorno[0], retorno[1]
                ce = F.cross_entropy(logits, alvo)
                total = ce + alinhamento

                assert logits.shape == (1, 2, 256, 256), f"Shape inesperado: {logits.shape}"
                assert torch.isfinite(logits).all().item(), "Logits não finitos."
                assert torch.isfinite(total).all().item(), "Perda não finita."
                total.backward()

                com_gradiente = [
                    (nome, p) for nome, p in modelo.encoder.named_parameters()
                    if p.grad is not None
                ]
                assert com_gradiente, "Nenhum parâmetro recebeu gradiente."
                assert all(
                    torch.isfinite(p.grad).all().item()
                    for _, p in com_gradiente
                ), "Gradiente não finito."

                # Acompanhar um parâmetro com gradiente não nulo.
                escolhido = next(
                    (
                        (nome, p) for nome, p in com_gradiente
                        if torch.count_nonzero(p.grad).item() > 0
                    ),
                    None,
                )
                assert escolhido is not None, "Todos os gradientes são zero."
                nome_monitorado, parametro = escolhido
                antes = parametro.detach().clone()

                otimizador.step()

                assert all(
                    torch.isfinite(p).all().item() for p in modelo.parameters()
                ), "Parâmetro não finito após atualização."

                delta = (parametro.detach() - antes).abs().max().item()
                assert delta > 0, "O parâmetro monitorado não mudou."

                linha = {
                    "passo": passo,
                    "episodio": caminho.name,
                    "ce": ce.detach().item(),
                    "alinhamento": (
                        alinhamento.detach().item()
                        if torch.is_tensor(alinhamento)
                        else float(alinhamento)
                    ),
                    "total": total.detach().item(),
                    "parametro_monitorado": nome_monitorado,
                    "mudanca_maxima": delta,
                    "tensores_encoder_com_gradiente": len(com_gradiente),
                }
                historico.append(linha)

                with (pasta_saida / "passos.jsonl").open(
                    "a", encoding="utf-8"
                ) as arquivo:
                    arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")

                print(
                    f"{passo:02d}/12 | {caminho.name} | "
                    f"CE={linha['ce']:.6f} | "
                    f"alinhamento={linha['alinhamento']:.6f} | "
                    f"total={linha['total']:.6f} | Δpeso={delta:.3e}",
                    flush=True,
                )

                del retorno, logits, alinhamento, ce, total, antes

            return historico

        except Exception:
            (pasta_saida / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            modelo.load_state_dict(estado_inicial, strict=True)
            modelo.zero_grad(set_to_none=True)
            modelo.train(modo_anterior)
            resumo = {
                "atualizacoes_registradas": len(historico),
                "teste_completo": len(historico) == config["atualizacoes"],
                "pesos_iniciais_restaurados": True,
                "treinamento_definitivo": False,
            }
            (pasta_saida / "resumo.json").write_text(
                json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("\nPesos iniciais restaurados.")
            print("Registros:", pasta_saida)

    try:
        historico_teste_eft = executar_teste_curto()
    finally:
        del estado_inicial
    return historico_teste_eft, pasta_saida
