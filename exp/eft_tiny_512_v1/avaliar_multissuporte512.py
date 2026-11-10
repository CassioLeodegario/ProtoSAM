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
TREINO = ROOT / "reproducao/eft_tiny512_cuda_5000_20261017T103006205469Z"
REFERENCIA = ROOT / "reproducao/eft_tiny_passo_2000_validacao_20261011T221527472147Z"
LEGADO = ROOT / "exp/eft_validacao_polipos_v1"
PASSOS = [0, 200]
PLANO_SUPORTES = Path('/home/1542521/cassio_pereira/ProtoSAM-repro/reproducao/multissuporte_tiny_eft2000_kvasir_v1/plano.json')

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

def main(out, suporte_id):
    assert torch.cuda.is_available() and vmamba.WITH_SELECTIVESCAN_MAMBA
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)

    ref = ler(REFERENCIA / "config.json")
    plano_suportes = ler(PLANO_SUPORTES)
    assert plano_suportes["n_suportes"] == 20
    assert plano_suportes["n_queries"] == 145
    assert plano_suportes["resolucao_encoder"] == 672
    assert plano_suportes["canvas_sam"] == 1024

    selecionados = [
        s for s in plano_suportes["suportes"] if s["id"] == suporte_id
    ]
    assert len(selecionados) == 1
    escolhido = selecionados[0]
    historico_suporte = PLANO_SUPORTES.parent / suporte_id / "config.json"
    ref_sup = ler(historico_suporte)

    # Identidade do suporte fixada pelo plano; nenhum novo sorteio.
    meta = dict(escolhido["meta"])
    for chave in ("imagem", "mascara"):
        assert atual(meta[chave]) == atual(ref_sup["suporte"][chave])
        assert escolhido[chave + "_sha256"] == ref_sup["suporte"][chave + "_sha256"]
        meta[chave + "_sha256"] = escolhido[chave + "_sha256"]

    assert ref_sup["checkpoint_original_sha256"] == ref["checkpoint_original_sha256"]
    assert ref_sup["checkpoint_sam_sha256"] == ref["checkpoint_sam_sha256"]
    assert {r["id"] for r in ref_sup["episodios"]} == {
        r["id"] for r in ref["episodios"]
    }

    ref["suporte"] = meta
    ref["suporte_tensor_sha256"] = ref_sup["suporte_tensor_sha256"]

    cfg_treino = ler(TREINO / "config.json")
    planos = ler(TREINO / "planos.json")
    assert ler(TREINO / "concluido.json")["passos"] == 5000
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
    area = int((suporte["label"] > .5).sum())
    assert 0 < area < 1024 * 1024
    if suporte_id == "S00":
        assert area == 51940
    else:
        imagens_treino = {str(atual(r["imagem"])) for r in planos["treino"]}
        assert sup_meta["imagem"] in imagens_treino

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
        "finalidade": "multissuporte fixo; inicial versus EFT512 passo200",
        "suporte_id": suporte_id,
        "plano_suportes_sha256": sha(PLANO_SUPORTES),
        "historico_suporte_config_sha256": sha(historico_suporte),
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
            "limite": "selecao na validacao S00; nao e resultado de teste",
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
        import fcntl
        out.mkdir(parents=True, exist_ok=True)
        trava = (out / "execucao.lock").open("a")
        fcntl.flock(trava, fcntl.LOCK_EX | fcntl.LOCK_NB)

        plano_fixo = ler(PLANO_SUPORTES)
        suportes = plano_fixo["suportes"]
        assert len(suportes) == 20
        assert [s["id"] for s in suportes] == [f"S{i:02d}" for i in range(20)]
        assert len({str(atual(s["meta"]["imagem"])) for s in suportes}) == 20

        config_lote = {
            "plano_original": str(PLANO_SUPORTES),
            "plano_sha256": sha(PLANO_SUPORTES),
            "avaliador_sha256": sha(__file__),
            "treino": str(TREINO),
            "passos": PASSOS,
            "suportes": [s["id"] for s in suportes],
            "observacao": "S00 usado na selecao; resumo principal S01-S19",
        }
        if (out / "config_lote.json").exists():
            assert ler(out / "config_lote.json") == config_lote
        else:
            salvar(out / "config_lote.json", config_lote)
            shutil.copy2(PLANO_SUPORTES, out / "plano_original.json")
            shutil.copy2(__file__, out / "avaliador_usado.py")

        tabela_suportes = []
        for s in suportes:
            sid = s["id"]
            print(f"\\n=== SUPORTE {sid} ===", flush=True)
            main(out / sid, sid)

            inicial = ler(out / sid / "passo_0000/resumo.json")
            eft = ler(out / sid / "passo_0200/resumo.json")
            row = {"suporte": sid, "nome": s["meta"]["nome"]}
            for etapa in ("ALP", "SAM"):
                for metrica in ("Dice", "IoU", "precisao", "recall"):
                    chave = f"{etapa}_{metrica}"
                    row[f"inicial_{chave}"] = inicial[chave]
                    row[f"eft512_200_{chave}"] = eft[chave]
                row[f"delta_{etapa}_Dice"] = eft[f"{etapa}_Dice"] - inicial[f"{etapa}_Dice"]
            tabela_suportes.append(row)
            csv_atomico(out / "por_suporte.csv", tabela_suportes)
            print("SUPORTE CONCLUIDO:", json.dumps(row), flush=True)

        sorteados = [r for r in tabela_suportes if r["suporte"] != "S00"]
        resumo_multi = {
            "n_suportes_sorteados": len(sorteados),
            "n_queries_por_suporte": 145,
            "S00": tabela_suportes[0],
            "S01_S19": {},
            "limite": "validacao exploratoria; nao mede variacao entre treinamentos",
        }
        for etapa in ("ALP", "SAM"):
            dados = {}
            for condicao in ("inicial", "eft512_200"):
                valores = np.array([r[f"{condicao}_{etapa}_Dice"] for r in sorteados])
                dados[condicao] = {
                    "media": float(valores.mean()),
                    "desvio_padrao_entre_suportes_ddof1": float(valores.std(ddof=1)),
                    "minimo": float(valores.min()),
                    "maximo": float(valores.max()),
                }
            delta = np.array([r[f"delta_{etapa}_Dice"] for r in sorteados])
            dados["delta_medio_pp"] = float(delta.mean() * 100)
            dados["suportes_com_melhora"] = int((delta > 1e-8).sum())
            dados["suportes_com_piora"] = int((delta < -1e-8).sum())
            resumo_multi["S01_S19"][etapa] = dados

        salvar(out / "resumo_multissuporte.json", resumo_multi)
        print(json.dumps(resumo_multi, indent=2, ensure_ascii=False), flush=True)
        print("MULTISSUPORTE CONCLUIDO: 20 suportes.", flush=True)

    except BaseException:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"erro_{time.time_ns()}.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
