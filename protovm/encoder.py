"""Construção do ALP (FewShotSeg oficial) com VMamba ou DINOv2 como encoder.

Uma única função substitui as sete cópias de eft_modelo.py e os construtores
de validação do ProtoSAM-repro. forward e alignLoss continuam os oficiais;
só get_encoder, get_cls e get_features são adaptados.

Resolução, estágio, normalização e grade são parâmetros:
    resolucao  lado da imagem que entra no encoder (256, 512, 672, ...)
    estagio    2 (penúltimo) ou 3 (último) do VMamba
    norm       'checkpoint'  -> outnorm3 com a norm do classificador pré-treinado (só estágio 3)
               'sem_norm'    -> saída crua dos blocos
               'ln_unitaria' -> LayerNorm sem parâmetros (escala 1, viés 0)
    grade      lado do mapa de features entregue ao ALP (sempre interpolado)
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from protovm import RAIZ
from protovm.util import semear, sha256

VARIANTES = {
    'tiny': {'depths': [2, 2, 5, 2], 'dims': 96, 'drop_path_rate': 0.2},
    'small': {'depths': [2, 2, 15, 2], 'dims': 96, 'drop_path_rate': 0.3},
    'base': {'depths': [2, 2, 15, 2], 'dims': 128, 'drop_path_rate': 0.6},
}
NORMS = ('checkpoint', 'sem_norm', 'ln_unitaria')
PESOS = RAIZ / 'pretrained_model'


def _vmamba_pretreinado(variante):
    from models.vmamba import Backbone_VSSM
    cfg = VARIANTES[variante]
    encoder = Backbone_VSSM(
        out_indices=(3,), pretrained=None, patch_size=4, in_chans=3, num_classes=1000,
        depths=cfg['depths'], dims=cfg['dims'],
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer='silu',
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
        forward_type='v05_noz', mlp_ratio=4.0, mlp_act_layer='gelu', mlp_drop_rate=0.0,
        drop_path_rate=cfg['drop_path_rate'], patch_norm=True, norm_layer='ln2d',
        downsample_version='v3', patchembed_version='v2', use_checkpoint=False, posembed=False,
    )
    checkpoint = torch.load(PESOS / f'vmamba_{variante}_v2.pth', map_location='cpu', weights_only=True)
    pesos = dict(checkpoint['model'])
    # Recupera a normalização aprendida do classificador; descarta só a cabeça.
    for sufixo in ('weight', 'bias'):
        pesos[f'outnorm3.{sufixo}'] = pesos.pop(f'classifier.norm.{sufixo}')
        pesos.pop(f'classifier.head.{sufixo}')
    encoder.load_state_dict(pesos, strict=True)
    return encoder


def _features_vmamba(encoder, imagens, estagio, norm):
    """Mesmo caminho de Backbone_VSSM.forward, parando no estágio pedido."""
    assert encoder.channel_first
    x = encoder.patch_embed(imagens)
    for i in range(estagio + 1):
        saida = encoder.layers[i].blocks(x)
        if i < estagio:
            x = encoder.layers[i].downsample(saida)
    if norm == 'checkpoint':
        return encoder.outnorm3(saida)
    if norm == 'ln_unitaria':
        c = saida.shape[1]
        return F.layer_norm(saida.permute(0, 2, 3, 1), (c,), eps=encoder.outnorm3.eps).permute(0, 3, 1, 2).contiguous()
    return saida


def construir_alp(modelo='vmamba_tiny', estagio=3, norm='checkpoint', resolucao=672,
                  grade=32, proto_grid=8, treino=False, pesos=None, semente=1234):
    """Retorna o FewShotSeg na GPU.

    pesos: caminho de um checkpoint EFT (chave 'modelo'); None usa o pré-treinado.
    treino: True liga alinhamento e modo train; False congela tudo em eval.
    """
    from models.grid_proto_fewshot import FewShotSeg, MultiProtoAsConv
    assert torch.cuda.is_available(), 'CUDA indisponível.'
    semear(semente)

    if modelo.startswith('dinov2'):
        assert modelo in DINOV2, f'Modelo desconhecido: {modelo}'
        return _construir_dinov2(modelo, resolucao, treino=treino, pesos=pesos)

    variante = modelo.removeprefix('vmamba_')
    assert variante in VARIANTES, f'Modelo desconhecido: {modelo}'
    assert estagio in (2, 3) and norm in NORMS
    assert not (norm == 'checkpoint' and estagio != 3), "norm 'checkpoint' só existe no estágio 3"
    lado = resolucao // (4 * 2 ** estagio)
    assert lado * 4 * 2 ** estagio == resolucao, f'Resolução {resolucao} não divisível no estágio {estagio}'
    canais = VARIANTES[variante]['dims'] * 2 ** estagio
    encoder = _vmamba_pretreinado(variante)

    class ALPVMamba(FewShotSeg):
        def get_encoder(self):
            self.encoder = encoder
            self.config['feature_hw'] = [grade, grade]

        def get_cls(self):
            self.cls_unit = MultiProtoAsConv(proto_grid=[proto_grid, proto_grid],
                                             feature_hw=[grade, grade], embed_dim=canais)

        def get_features(self, imgs_concat):
            imagens = F.interpolate(imgs_concat, size=(resolucao, resolucao),
                                    mode='bilinear', align_corners=False)
            fts = _features_vmamba(self.encoder, imagens, estagio, norm)
            assert tuple(fts.shape[1:]) == (canais, lado, lado), fts.shape
            return F.interpolate(fts, size=(grade, grade), mode='bilinear', align_corners=False)

    cfg = {'which_model': modelo, 'cls_name': 'grid_proto', 'proto_grid_size': proto_grid,
           'feature_hw': [grade, grade], 'lora': 0, 'align': treino, 'debug': False}
    alp = ALPVMamba(image_size=resolucao, pretrained_path=None, cfg=cfg)
    assert type(alp).forward is FewShotSeg.forward and type(alp).alignLoss is FewShotSeg.alignLoss
    alp.descricao = {'modelo': modelo, 'estagio': estagio, 'norm': norm, 'resolucao': resolucao,
                     'grade': grade, 'lado_features': lado, 'canais': canais, 'pesos_sha256': None}

    if pesos is not None:
        salvo = torch.load(pesos, map_location='cpu', weights_only=True)
        alp.load_state_dict(salvo['modelo'], strict=True)
        alp.descricao['passo_eft'] = int(salvo['passo'])
        # Identifica o arquivo avaliado: o nome da pasta sozinho não distingue dois checkpoints.
        alp.descricao['pesos_arquivo'] = str(Path(pesos).resolve())
        alp.descricao['pesos_sha256'] = sha256(pesos)
        del salvo
    alp = alp.to('cuda', dtype=torch.float32)

    if treino:
        alp.train().requires_grad_(True)
        if estagio < 3:
            # Módulos depois da saída escolhida não participam do grafo.
            alp.encoder.layers[estagio].downsample.requires_grad_(False)
            for i in range(estagio + 1, 4):
                alp.encoder.layers[i].requires_grad_(False)
            alp.encoder.outnorm3.requires_grad_(False)
    else:
        alp.eval().requires_grad_(False)
    return alp


# modelo: (entrada no hub do DINOv2, arquivo de pesos em pretrained_model/hub/checkpoints, canais)
DINOV2 = {
    'dinov2_s14': ('dinov2_vits14', 'dinov2_vits14_pretrain.pth', 384),
    'dinov2_b14': ('dinov2_vitb14', 'dinov2_vitb14_pretrain.pth', 768),
    'dinov2_l14': ('dinov2_vitl14', 'dinov2_vitl14_pretrain.pth', 1024),
}


def _construir_dinov2(modelo, resolucao, treino=False, pesos=None):
    import models.grid_proto_fewshot as oficial
    entrada, arquivo, canais = DINOV2[modelo]
    repo_hub = PESOS / 'hub' / 'facebookresearch_dinov2_main'
    pesos = PESOS / 'hub' / 'checkpoints' / arquivo

    class ALPDinoLocal(oficial.FewShotSeg):
        """Só troca o download pelo carregamento local; o resto é o oficial."""
        def get_encoder(self):
            self.encoder = torch.hub.load(str(repo_hub), entrada, source='local', pretrained=False)
            self.encoder.load_state_dict(torch.load(pesos, map_location='cpu', weights_only=True), strict=True)
            lado = max(self.image_size // 14, oficial.DEFAULT_FEATURE_SIZE)
            self.config['feature_hw'] = [lado, lado]

        def get_cls(self):
            if modelo == 'dinov2_s14':
                # O oficial só conhece a dimensão do B (768) e do L (1024); para o S cairia em 256.
                self.cls_unit = oficial.MultiProtoAsConv(proto_grid=[8, 8], feature_hw=self.config['feature_hw'],
                                                         embed_dim=canais)
            else:
                super().get_cls()

    cfg = {'which_model': modelo, 'cls_name': 'grid_proto', 'proto_grid_size': 8, 'lora': 0,
           'align': treino, 'debug': False, 'use_coco_init': False, 'use_pos_enc': False,
           'use_slice_adapter': False, 'adapter_layers': 3, 'dinov2_loss': False}
    alp = ALPDinoLocal(image_size=resolucao, pretrained_path=None, cfg=cfg)
    assert getattr(alp.encoder, 'embed_dim', canais) == canais, (modelo, alp.encoder.embed_dim)
    alp.descricao = {'modelo': modelo, 'resolucao': resolucao,
                     'grade': alp.config['feature_hw'][0], 'estagio': None, 'norm': None, 'pesos_sha256': None}
    if pesos is not None:  # checkpoint de um ajuste auto-supervisionado (scripts/treinar_regioes.py)
        salvo = torch.load(pesos, map_location='cpu', weights_only=True)
        alp.load_state_dict(salvo['modelo'], strict=True)
        alp.descricao['passo_eft'] = int(salvo['passo'])
        alp.descricao['pesos_arquivo'] = str(Path(pesos).resolve())
        alp.descricao['pesos_sha256'] = sha256(pesos)
        del salvo
    alp = alp.to('cuda', dtype=torch.float32)
    return alp.train().requires_grad_(True) if treino else alp.eval().requires_grad_(False)
