
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from eft_modelo_512 import construir_modelo as construir_original
from models.grid_proto_fewshot import MultiProtoAsConv


def construir_modelo(variante, repo):
    assert variante == "tiny"
    modelo = construir_original(variante, repo=Path(repo))
    encoder = modelo.encoder

    assert encoder.channel_first
    assert len(encoder.layers) == 4
    assert int(encoder.dims[2]) == 384
    assert len(modelo.cls_unit.state_dict()) == 0

    # Preserva o checkpoint completo, mas congela módulos não utilizados.
    encoder.layers[2].downsample.requires_grad_(False)
    encoder.layers[3].requires_grad_(False)
    encoder.outnorm3.requires_grad_(False)

    modelo.cls_unit = MultiProtoAsConv(
        proto_grid=[8, 8],
        feature_hw=[32, 32],
        embed_dim=384,
    ).to("cuda", dtype=torch.float32)
    assert len(modelo.cls_unit.state_dict()) == 0

    def get_features_penultima(self, imgs_concat):
        x = F.interpolate(
            imgs_concat, size=(512, 512),
            mode="bilinear", align_corners=False,
        )
        x = self.encoder.patch_embed(x)

        for i in range(3):
            camada = self.encoder.layers[i]
            x = camada.blocks(x)
            if i < 2:
                x = camada.downsample(x)

        assert tuple(x.shape[1:]) == (384, 32, 32), tuple(x.shape)
        return x.contiguous()

    modelo.get_features = MethodType(get_features_penultima, modelo)
    modelo.train()
    modelo.zero_grad(set_to_none=True)

    print(
        "EFT direto: penultima, 384x32x32, sem outnorm; "
        "patch_embed e estagios 0-2 treinaveis.",
        flush=True,
    )
    print(
        "Parametros treinaveis:",
        sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        flush=True,
    )
    return modelo
