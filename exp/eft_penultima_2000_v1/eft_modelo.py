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
    canais = variante_cfg["dims"] * 4
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
            features = self.encoder.patch_embed(imagens)
            for i in range(3):
                features = self.encoder.layers[i].blocks(features)
                if i < 2:
                    features = self.encoder.layers[i].downsample(features)

            assert self.encoder.channel_first
            assert features.shape[1:] == (canais, 16, 16)

            features = F.layer_norm(
                features.permute(0,2,3,1), (canais,),
                eps=self.encoder.outnorm3.eps,
            ).permute(0,3,1,2).contiguous()

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
    # Módulos posteriores à saída escolhida ficam inativos.
    modelo.encoder.layers[2].downsample.requires_grad_(False)
    modelo.encoder.layers[3].requires_grad_(False)
    modelo.encoder.outnorm3.requires_grad_(False)
    modelo.zero_grad(set_to_none=True)



    modelo.variante_eft = variante
    modelo.repo_eft = str(repo)
    modelo.checkpoint_eft = str(CHECKPOINT)
    return modelo
