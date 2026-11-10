from pathlib import Path
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import selective_scan_cuda

ROOT = Path("/home/1542521/cassio_pereira/ProtoSAM-repro")
TREINO = Path('/home/1542521/cassio_pereira/ProtoSAM-repro/reproducao/eft_SAM_vs_SP256_1231_20261021T194500705194Z/treino_SAM')
REFERENCIA = ROOT / "reproducao/eft_tiny_passo_2000_validacao_20261011T221527472147Z"
LEGADO = ROOT / "exp/eft_validacao_polipos_v1"
PASSOS = [0, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

os.chdir(ROOT)
sys.path.insert(0, str(LEGADO))
sys.path.insert(0, str(ROOT))

# Reutiliza somente as funções de construção e métricas já inspecionadas.
# Não chama executar() nem modifica os arquivos antigos.
import validar_marcos_5000 as legado
legado.REPO = ROOT

from dataloaders.PolypDataset import PolypDataset
from dataloaders.PolypTransforms import get_polyp_transform
from models.segment_anything.utils.transforms import ResizeLongestSide
from models.ProtoSAM import ProtoSAM, ALPNetWrapper, ALPNetInput
from models import vmamba

def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()

def salvar(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)

def csv_atomico(p, linhas):
    p = Path(p)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    os.replace(tmp, p)

def atual(p):
    p = Path(p)
    if p.is_relative_to("/workspace"):
        p = ROOT.parent / p.relative_to("/workspace")
    return p.resolve()

def hash_tensor(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def instalar_penultima(modelo):
    from types import MethodType
    import torch.nn.functional as F
    from models.grid_proto_fewshot import MultiProtoAsConv

    encoder = modelo.encoder
    assert encoder.channel_first
    assert len(encoder.layers) == 4
    assert int(encoder.dims[2]) == 384

    # Não descartar silenciosamente parâmetros aprendidos do ALP.
    assert len(modelo.cls_unit.state_dict()) == 0, (
        "ALP contém estado; é necessário conferir antes da adaptação."
    )
    cls_nova = MultiProtoAsConv(
        proto_grid=[8, 8],
        feature_hw=[32, 32],
        embed_dim=384,
    ).to("cuda", dtype=torch.float32)
    assert len(cls_nova.state_dict()) == 0
    modelo.cls_unit = cls_nova

    def get_features_penultima(self, imgs_concat):
        x = F.interpolate(
            imgs_concat, size=(672, 672),
            mode="bilinear", align_corners=False,
        )
        x = self.encoder.patch_embed(x)

        # Mesmo percurso do Backbone_VSSM até a saída do estágio 2.
        for i in range(3):
            camada = self.encoder.layers[i]
            x = camada.blocks(x)
            if i < 2:
                x = camada.downsample(x)

        assert tuple(x.shape[1:]) == (384, 42, 42), tuple(x.shape)
        assert torch.isfinite(x).all().item()

        # Nenhuma outnorm adicional. Norms internas preservadas.
        return F.interpolate(
            x.contiguous(), size=(32, 32),
            mode="bilinear", align_corners=False,
        )

    modelo.get_features = MethodType(get_features_penultima, modelo)
    print(
        "Extração: estágio 2 antes do downsampling | "
        "384x42x42 -> 384x32x32 | sem outnorm",
        flush=True,
    )


def main(out):
    assert torch.cuda.is_available() and vmamba.WITH_SELECTIVESCAN_MAMBA
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    ref = ler(REFERENCIA / "config.json")
    cfg_treino = ler(TREINO / "config.json")
    planos = ler(TREINO / "planos.json")
    # Recupera somente os registros de validacao da divisao original.
    # O plano de treino e os checkpoints permanecem intactos.
    origem_divisao = Path('/home/1542521/cassio_pereira/ProtoSAM-repro/reproducao/eft_ultima672_10000_wd001_20261018T141350233303Z/planos.json')
    assert sha(origem_divisao) == '880b04375c4a25fa0efd326afc666011c47fe460c5180ad91653fe0cb4be024b'
    plano_divisao = ler(origem_divisao)
    assert planos["validacao"] == [], "Plano diferente do esperado."
    assert len(plano_divisao["validacao"]) == 145
    assert len({r["indice"] for r in plano_divisao["validacao"]}) == 145
    assert not (
        {r["indice"] for r in planos["treino"]}
        & {r["indice"] for r in plano_divisao["validacao"]}
    )
    planos = dict(planos, validacao=plano_divisao["validacao"])

    assert ler(TREINO / "concluido.json")["passos"] == 10000
    assert sha(selective_scan_cuda.__file__) == cfg_treino["extensao_scan_sha256"]

    oficiais = [
        "models/grid_proto_fewshot.py", "models/alpmodule.py",
        "models/ProtoSAM.py", "dataloaders/PolypDataset.py",
        "dataloaders/PolypTransforms.py",
        "models/segment_anything/utils/transforms.py", "models/vmamba.py",
    ]
    for relativo in oficiais:
        hashes = [
            v for k, v in ref["fontes_sha256"].items()
            if k.endswith("/" + relativo)
        ]
        assert len(hashes) == 1 and sha(ROOT / relativo) == hashes[0], relativo

    sup_meta = dict(ref["suporte"])
    for chave in ("imagem", "mascara"):
        sup_meta[chave] = str(atual(sup_meta[chave]))
        assert sha(sup_meta[chave]) == sup_meta[chave + "_sha256"]

    reservadas = {f'{r["indice"]:04d}': r for r in planos["validacao"]}
    assert len(reservadas) == 145
    assert not (
        {r["indice"] for r in planos["validacao"]}
        & {r["indice"] for r in planos["treino"]}
    )
    assert {r["id"] for r in ref["episodios"]} == set(reservadas)

    # Mesma construção do ds_train da factory; somente caminhos atualizados.
    _, transform_test = get_polyp_transform()
    image_root = str(Path(sup_meta["imagem"]).parent)
    gt_root = str(Path(sup_meta["mascara"]).parent)
    ds = PolypDataset(
        root=image_root, image_root=image_root, gt_root=gt_root,
        augmentations=transform_test, sam_trans=ResizeLongestSide(1024),
        train=True, image_size=(1024, 1024)
    )
    mapa = {str(Path(p).resolve()): i for i, p in enumerate(ds.images)}
    assert len(mapa) == len(ds.images)

    itens = []
    for antigo in ref["episodios"]:
        item = dict(antigo)
        r = reservadas[item["id"]]
        for chave in ("imagem", "mascara"):
            item[chave] = str(atual(item[chave]))
            assert sha(item[chave]) == item[chave + "_sha256"]
        assert item["imagem"] == str(atual(r["imagem"]))
        assert item["imagem_sha256"] == r["imagem_sha256"]
        i = mapa[item["imagem"]]
        assert Path(ds.gts[i]).resolve() == Path(item["mascara"])
        item["indice_dataset"] = i
        itens.append(item)

    assert sup_meta["imagem"] not in {i["imagem"] for i in itens}
    indice_sup = mapa[sup_meta["imagem"]]
    assert Path(ds.gts[indice_sup]).resolve() == Path(sup_meta["mascara"])
    suporte = ds[indice_sup]
    legado.conferir_amostra(suporte)
    assert int((suporte["label"] > .5).sum()) == 51940
    assert hash_tensor(suporte["image"]) == ref["suporte_tensor_sha256"], (
        "Tensor do suporte difere da referência."
    )

    cfg_sam = dict(ref["sam"])
    cfg_sam["sam_pretrained_path"] = str(ROOT / "pretrained_model/sam_vit_h.pth")
    assert sha(cfg_sam["sam_pretrained_path"]) == ref["checkpoint_sam_sha256"]

    fontes = [Path(__file__), LEGADO / "validar_marcos_5000.py",
              LEGADO / "eft_modelo_validacao.py"] + [ROOT / r for r in oficiais]
    checkpoints = {
        str(n): sha(TREINO / f"checkpoint_{n:04d}.pt") for n in PASSOS
    }
    config = {
        "finalidade": "penultima sem outnorm; treino672 ultima 10k WD0.01; S00",
        "extracao": {
            "estagio_indice": 2,
            "ponto": "apos blocks, antes de downsample",
            "canais": 384,
            "grade_nativa": [42, 42],
            "normalizacao_saida": "nenhuma",
            "normalizacoes_internas": "preservadas",
            "novo_treinamento": False,
        },
        "treino": str(TREINO), "treino_config_sha256": sha(TREINO / "config.json"),
        "passos": PASSOS, "checkpoints_sha256": checkpoints,
        "episodios": itens, "suporte": sup_meta,
        "suporte_tensor_sha256": hash_tensor(suporte["image"]),
        "encoder": 672, "grade_alp": [32, 32], "canvas": 1024,
        "sam": cfg_sam, "sam_sha256": ref["checkpoint_sam_sha256"],
        "selecao_sam": ref["selecao_sam"],
        "alp_inferencia": ref["alp_inferencia"],
        "referencia_config_sha256": sha(REFERENCIA / "config.json"),
        "fontes_sha256": {str(p): sha(p) for p in fontes},
        "backend": "selective_scan_cuda",
        "extensao_sha256": sha(selective_scan_cuda.__file__),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "pacotes": {p: importlib.metadata.version(p) for p in [
            "numpy", "Pillow", "torchvision", "timm", "scipy",
            "opencv-python", "mamba-ssm"
        ] if importlib.util.find_spec(
            {"Pillow": "PIL", "opencv-python": "cv2", "mamba-ssm": "mamba_ssm"}.get(p, p)
        ) is not None},
        "gpu_driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
        ], text=True).strip(),
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        assert ler(out / "config.json") == config, "Configuração mudou; retomada recusada."
    else:
        salvar(out / "config.json", config)
        (out / "fontes").mkdir(exist_ok=True)
        for i, p in enumerate(fontes):
            shutil.copy2(p, out / "fontes" / f"{i:02d}_{p.name}")

    print("AVALIAÇÃO:", out, flush=True)
    print("Suporte, dados e código conferidos. 145 queries por checkpoint.", flush=True)

    original_fwd = selective_scan_cuda.fwd
    original_fallback = vmamba.selective_scan_torch
    chamadas = {"cuda": 0}
    def contar(*args, **kwargs):
        chamadas["cuda"] += 1
        return original_fwd(*args, **kwargs)
    def recusar(*args, **kwargs):
        raise RuntimeError("Fallback Torch não permitido nesta avaliação.")
    selective_scan_cuda.fwd = contar
    vmamba.selective_scan_torch = recusar

    tabela = []
    try:
        for passo in PASSOS:
            pasta = out / f"passo_{passo:04d}"
            for sub in ("registros", "alp", "sam", "gt"):
                (pasta / sub).mkdir(parents=True, exist_ok=True)

            modelo = legado.construir_672("tiny")
            ckpt = torch.load(
                TREINO / f"checkpoint_{passo:04d}.pt",
                map_location="cpu", weights_only=True
            )
            assert ckpt["passo"] == passo and ckpt["config"] == cfg_treino
            if passo == 0:
                inicial = modelo.state_dict()
                assert inicial.keys() == ckpt["modelo"].keys()
                assert all(
                    torch.equal(inicial[k].cpu(), ckpt["modelo"][k])
                    for k in inicial
                ), "Checkpoint 0 difere da inicialização pré-treinada."
                del inicial
            modelo.load_state_dict(ckpt["modelo"], strict=True)
            del ckpt
            instalar_penultima(modelo)

            wrapper = ALPNetWrapper(modelo)
            protosam = ProtoSAM(
                coarse_segmentation_model=wrapper, **cfg_sam
            ).to("cuda", dtype=torch.float32)
            protosam.eval().requires_grad_(False)
            modelo.eval().requires_grad_(False)
            sup_img = suporte["image"].unsqueeze(0).cuda().float()
            sup_mask = suporte["label"].unsqueeze(0).cuda().float()
            linhas = []

            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                for pos, item in enumerate(itens, 1):
                    registro = pasta / "registros" / f'{item["id"]}.json'
                    if registro.exists():
                        linha = ler(registro)
                        assert linha["item"] == item and linha["passo"] == passo
                        for relativo, digest in linha["arquivos_sha256"].items():
                            assert sha(pasta / relativo) == digest
                    else:
                        inicio = time.perf_counter()
                        amostra = ds[item["indice_dataset"]]
                        legado.conferir_amostra(amostra)
                        gt = amostra["label"].cpu().numpy() > .5
                        input_sha = hash_tensor(amostra["image"])

                        # Confere o pré-processamento anterior quando o registro está disponível.
                        antigo = REFERENCIA / "antes/registros" / f'{item["id"]}.json'
                        if antigo.is_file():
                            assert input_sha == ler(antigo)["query_tensor_sha256"], (
                                f'Pré-processamento mudou: {item["id"]}'
                            )
                        if passo != 0:
                            base = ler(out / "passo_0000/registros" / f'{item["id"]}.json')
                            assert input_sha == base["query_tensor_sha256"]
                            with Image.open(out / "passo_0000/gt" / f'{item["id"]}.png') as im:
                                assert np.array_equal(gt, np.asarray(im) > 0)

                        qry = amostra["image"].unsqueeze(0).cuda().float()
                        def entrada():
                            return ALPNetInput(
                                support_images=[sup_img], support_labels=[sup_mask],
                                query_images=qry, isval=True, val_wsize=2,
                                show_viz=False, supp_fts=None
                            )
                        chamadas["cuda"] = 0
                        logits = wrapper(entrada())
                        assert tuple(logits.shape) == (1, 2, 1024, 1024)
                        assert torch.isfinite(logits).all().item()
                        alp = (logits.argmax(1)[0] == 1).cpu().numpy()
                        del logits

                        saida, scores = protosam(
                            query_image=qry, coarse_model_input=entrada(), degrees_rotate=0
                        )
                        sam = saida.detach().cpu().numpy().squeeze()
                        assert sam.shape == (1024, 1024) and np.isin(sam, [0, 1]).all()
                        assert chamadas["cuda"] > 0
                        sam = sam.astype(bool)

                        hashes = {}
                        for tipo, mask in (("alp", alp), ("sam", sam), ("gt", gt)):
                            arquivo = pasta / tipo / f'{item["id"]}.png'
                            Image.fromarray(mask.astype(np.uint8) * 255).save(arquivo)
                            hashes[str(arquivo.relative_to(pasta))] = sha(arquivo)

                        linha = {
                            "passo": passo, "item": item,
                            "metricas": {"ALP": legado.medir(alp, gt), "SAM": legado.medir(sam, gt)},
                            "query_tensor_sha256": input_sha,
                            "arquivos_sha256": hashes,
                            "chamadas_cuda": chamadas["cuda"],
                            "scores_sam": len(scores),
                            "segundos": time.perf_counter() - inicio,
                        }
                        salvar(registro, linha)
                        del saida, scores, qry

                    linhas.append(linha)
                    print(
                        f"Passo {passo} | {pos}/145 | "
                        f"ALP={linha['metricas']['ALP']['Dice']:.4f} | "
                        f"SAM={linha['metricas']['SAM']['Dice']:.4f}", flush=True
                    )

            resumo = {"passo": passo, "n": len(linhas)}
            por_imagem = []
            for linha in linhas:
                row = {"id": linha["item"]["id"], "imagem": linha["item"]["imagem"]}
                for etapa in ("ALP", "SAM"):
                    for metrica in ("Dice", "IoU", "precisao", "recall"):
                        row[f"{etapa}_{metrica}"] = linha["metricas"][etapa][metrica]
                por_imagem.append(row)
            for etapa in ("ALP", "SAM"):
                for metrica in ("Dice", "IoU", "precisao", "recall"):
                    chave = f"{etapa}_{metrica}"
                    resumo[chave] = float(np.mean([r[chave] for r in por_imagem]))

            csv_atomico(pasta / "por_imagem.csv", por_imagem)
            salvar(pasta / "resumo.json", resumo)
            tabela.append(resumo)
            csv_atomico(out / "por_checkpoint.csv", tabela)
            print("RESUMO:", json.dumps(resumo), flush=True)

            del protosam, wrapper, modelo, sup_img, sup_mask
            gc.collect()
            torch.cuda.empty_cache()

        melhor = max(tabela, key=lambda r: (r["SAM_Dice"], -r["passo"]))
        salvar(out / "concluido.json", {
            "checkpoints": len(tabela), "queries_por_checkpoint": 145,
            "melhor_incluindo_passo0": melhor,
            "criterio": "maior Dice medio ALP+SAM; empate favorece menor passo",
            "limite": "validacao S00; camada e norm de saida diferem da referencia; nao e resultado de teste",
        })
        print("AVALIAÇÃO CONCLUÍDA.", flush=True)
    finally:
        selective_scan_cuda.fwd = original_fwd
        vmamba.selective_scan_torch = original_fallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    try:
        main(out)
    except BaseException:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"erro_{time.time_ns()}.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
