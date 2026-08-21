"""
E1 — custo do encoder isolado, no ponto exato que o ALPNet consome (get_features).

Corrige os dois problemas do benchmark_encoder_v2.py:
  D1  cronometragem: torch.cuda.Event em vez de perf_counter; mediana + IQR
  D2  VRAM: pesos e ativação medidos separadamente, não somados

E acrescenta o que faltava: FLOPs, tokens reais e mapa efetivo entregue ao ALP (D3).

Uso:
    python -m exp.e1_encoder_cost                       # grade completa
    python -m exp.e1_encoder_cost --models vmamba_tiny --sizes 512 1024
    python -m exp.e1_encoder_cost --no-wandb --no-flops
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.grid_proto_fewshot import FewShotSeg
from exp.envinfo import collect, REPO_ROOT

WARMUP = 30
REPEATS = 100
SIZES = [256, 384, 512, 672, 768, 1024]
MODELS = ["dinov2_l14", "dinov2_b14", "dinov2_s14", "vmamba_tiny"]

PATCH = {"dinov2_l14": 14, "dinov2_b14": 14, "dinov2_s14": 14, "vmamba_tiny": 32}


def build_model(modelname, image_size):
    cfg = {
        "align": False, "use_coco_init": False, "which_model": modelname,
        "cls_name": "grid_proto", "proto_grid_size": 8,
        "feature_hw": [image_size // 8, image_size // 8],
        "reload_model_path": None, "lora": 0, "use_slice_adapter": False,
        "adapter_layers": 3, "debug": False, "use_pos_enc": False,
    }
    return FewShotSeg(image_size=image_size, pretrained_path=None, cfg=cfg)


class FeatWrap(nn.Module):
    """get_features como um Module, para o fvcore conseguir percorrer."""

    def __init__(self, fss):
        super().__init__()
        self.fss = fss

    def forward(self, x):
        return self.fss.get_features(x)


def capture_tokens_real(model, x):
    """
    Tokens reais que o encoder produz ANTES do upsample para 32x32.
    Medido por hook, não digitado a partir de uma fórmula.
    """
    grabbed = {}

    def hook(_mod, _inp, out):
        if isinstance(out, dict) and "x_norm_patchtokens" in out:      # DINOv2
            hw = out["x_norm_patchtokens"].shape[1]
            grabbed["tokens_real"] = int(round(hw ** 0.5))
        elif isinstance(out, (list, tuple)) and torch.is_tensor(out[-1]):  # VMamba
            grabbed["tokens_real"] = int(out[-1].shape[-1])
        elif torch.is_tensor(out):
            grabbed["tokens_real"] = int(out.shape[-1])

    h = model.encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.get_features(x)
    finally:
        h.remove()
    return grabbed.get("tokens_real")


def measure_flops(model, x):
    """GFLOPs de get_features. Devolve (gflops, ops_nao_suportadas)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, {"fvcore": "ausente"}
    try:
        fca = FlopCountAnalysis(FeatWrap(model).eval(), x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
        return total / 1e9, dict(fca.unsupported_ops())
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


@torch.no_grad()
def bench_one(modelname, size, device, do_flops=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    model = build_model(modelname, size).to(device).eval()
    torch.cuda.synchronize(device)
    vram_weights = (torch.cuda.memory_allocated(device) - base) / 1024 ** 2

    x = torch.randn(1, 3, size, size, device=device)

    for _ in range(WARMUP):
        model.get_features(x)
    torch.cuda.synchronize(device)

    # --- VRAM de ativação: pico do forward menos o que já estava residente ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    out = model.get_features(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    vram_activation = (peak - resident) / 1024 ** 2
    vram_total = peak / 1024 ** 2
    feat_hw_alp = int(out.shape[-1])
    feat_c = int(out.shape[1])
    del out

    # --- latência: cuda.Event, mediana + IQR ---
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        starts[i].record()
        model.get_features(x)
        ends[i].record()
    torch.cuda.synchronize(device)
    times = np.array([starts[i].elapsed_time(ends[i]) for i in range(REPEATS)])

    tokens_real = capture_tokens_real(model, x)
    gflops, unsupported = measure_flops(model, x) if do_flops else (None, None)
    n_params = sum(p.numel() for p in model.encoder.parameters())

    rec = {
        "encoder": modelname,
        "input_size": size,
        "tokens_real": tokens_real,
        "feature_hw_alp": feat_hw_alp,
        "feature_channels": feat_c,
        "upsampled_to_32": bool(tokens_real is not None and tokens_real < 32),
        "gflops": None if gflops is None else round(gflops, 3),
        "latency_ms_median": round(float(np.median(times)), 4),
        "latency_ms_p25": round(float(np.percentile(times, 25)), 4),
        "latency_ms_p75": round(float(np.percentile(times, 75)), 4),
        "latency_ms_min": round(float(times.min()), 4),
        "throughput_fps": round(1000.0 / float(np.median(times)), 3),
        "vram_weights_mb": round(vram_weights, 1),
        "vram_activation_mb": round(vram_activation, 1),
        "vram_total_mb": round(vram_total, 1),
        "encoder_params_M": round(n_params / 1e6, 2),
        "flops_unsupported_ops": json.dumps(unsupported) if unsupported else None,
    }

    del model, x
    torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def bench_sam_encoder(device, size=1024, do_flops=True):
    """Encoder de imagem do SAM-H como ponto de referência fixo."""
    from models.segment_anything import sam_model_registry
    ckpt = os.path.join(REPO_ROOT, "pretrained_model/sam_vit_h.pth")
    if not os.path.isfile(ckpt):
        print(f"  [skip] SAM-H: checkpoint ausente em {ckpt}")
        return None

    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    enc = sam_model_registry["vit_h"](checkpoint=ckpt).image_encoder.to(device).eval()
    torch.cuda.synchronize(device)
    vram_weights = (torch.cuda.memory_allocated(device) - base) / 1024 ** 2

    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(WARMUP):
        enc(x)
    torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    out = enc(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    feat_hw, feat_c = int(out.shape[-1]), int(out.shape[1])
    del out

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        starts[i].record()
        enc(x)
        ends[i].record()
    torch.cuda.synchronize(device)
    times = np.array([starts[i].elapsed_time(ends[i]) for i in range(REPEATS)])

    gflops, unsupported = (None, None)
    if do_flops:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(enc, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            gflops = fca.total() / 1e9
            unsupported = dict(fca.unsupported_ops())
        except Exception as e:
            unsupported = {"error": f"{type(e).__name__}: {e}"}

    rec = {
        "encoder": "sam_h_image_encoder", "input_size": size,
        "tokens_real": feat_hw, "feature_hw_alp": feat_hw, "feature_channels": feat_c,
        "upsampled_to_32": False,
        "gflops": None if gflops is None else round(gflops, 3),
        "latency_ms_median": round(float(np.median(times)), 4),
        "latency_ms_p25": round(float(np.percentile(times, 25)), 4),
        "latency_ms_p75": round(float(np.percentile(times, 75)), 4),
        "latency_ms_min": round(float(times.min()), 4),
        "throughput_fps": round(1000.0 / float(np.median(times)), 3),
        "vram_weights_mb": round(vram_weights, 1),
        "vram_activation_mb": round((peak - resident) / 1024 ** 2, 1),
        "vram_total_mb": round(peak / 1024 ** 2, 1),
        "encoder_params_M": round(sum(p.numel() for p in enc.parameters()) / 1e6, 2),
        "flops_unsupported_ops": json.dumps(unsupported) if unsupported else None,
    }
    del enc, x
    torch.cuda.empty_cache()
    return rec


def check_criteria(records):
    """Critérios de aceitação do guia. Devolve lista de falhas."""
    fails = []
    by = {(r["encoder"], r["input_size"]): r for r in records}

    a, b = by.get(("vmamba_tiny", 512)), by.get(("vmamba_tiny", 1024))
    if a and b and a["gflops"] and b["gflops"]:
        ratio = b["gflops"] / a["gflops"]
        if not (3.2 <= ratio <= 4.8):
            fails.append(f"FLOPs VMamba 512->1024 cresceram {ratio:.2f}x (esperado ~4x)")

    ws = [r["vram_weights_mb"] for r in records if r["encoder"] == "dinov2_l14"]
    if ws:
        if max(ws) - min(ws) > 20:
            fails.append(f"VRAM de pesos do DINOv2-L varia com a resolucao: {min(ws)}-{max(ws)} MB")
        if not (1000 <= np.mean(ws) <= 1400):
            fails.append(f"VRAM de pesos do DINOv2-L = {np.mean(ws):.0f} MB (esperado ~1160 MB)")

    r = by.get(("dinov2_l14", 1024))
    if r and not (90 <= r["latency_ms_median"] <= 220):
        fails.append(f"Latencia DINOv2-L @1024 = {r['latency_ms_median']:.1f} ms (esperado ~140 ms)")

    return fails


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-flops", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--project", default="qualification")
    p.add_argument("--entity", default="leodegario")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    torch.manual_seed(args.seed)

    env = collect()
    print("=== ambiente ===")
    for k in ["git_commit", "git_dirty", "gpu_name", "torch", "tf32_matmul",
              "cudnn_benchmark", "selective_scan_kernel", "xformers"]:
        print(f"  {k:24s} {env.get(k)}")
    print()

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb

    records = []
    jobs = [(m, s) for m in args.models for s in args.sizes]
    for modelname, size in jobs:
        print(f"[{modelname} @ {size}] ...", flush=True)
        try:
            rec = bench_one(modelname, size, device, do_flops=not args.no_flops)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM — pulando")
            torch.cuda.empty_cache()
            continue
        records.append(rec)
        print(f"  tokens={rec['tokens_real']}^2 -> ALP {rec['feature_hw_alp']}^2 | "
              f"{rec['gflops']} GFLOPs | {rec['latency_ms_median']:.2f} ms "
              f"[{rec['latency_ms_p25']:.2f}-{rec['latency_ms_p75']:.2f}] | "
              f"pesos {rec['vram_weights_mb']:.0f} MB + ativacao {rec['vram_activation_mb']:.0f} MB")
        if rec["flops_unsupported_ops"] and rec["flops_unsupported_ops"] != "{}":
            print(f"  [ATENCAO] ops nao contabilizadas pelo fvcore: {rec['flops_unsupported_ops']}")

        if wandb:
            run = wandb.init(
                entity=args.entity, project=args.project, group="E1-cost",
                name=f"e1__{modelname}__{size}__fp32__s{args.seed}",
                tags=[modelname, "E1", "encoder-only", f"size{size}"],
                config={**env, **rec, "dtype": "fp32", "batch_size": 1,
                        "warmup": WARMUP, "repeats": REPEATS, "seed": args.seed},
                reinit=True,
            )
            run.log({f"cost/{k}": v for k, v in rec.items() if isinstance(v, (int, float))})
            run.finish()

    if not args.no_sam:
        print("[sam_h_image_encoder @ 1024] ...", flush=True)
        try:
            rec = bench_sam_encoder(device, 1024, do_flops=not args.no_flops)
            if rec:
                records.append(rec)
                print(f"  {rec['gflops']} GFLOPs | {rec['latency_ms_median']:.2f} ms | "
                      f"pesos {rec['vram_weights_mb']:.0f} MB + ativacao {rec['vram_activation_mb']:.0f} MB")
                if wandb:
                    run = wandb.init(
                        entity=args.entity, project=args.project, group="E1-cost",
                        name=f"e1__sam_h__1024__fp32__s{args.seed}",
                        tags=["sam_h", "E1", "encoder-only", "size1024"],
                        config={**env, **rec, "dtype": "fp32", "batch_size": 1,
                                "warmup": WARMUP, "repeats": REPEATS, "seed": args.seed},
                        reinit=True,
                    )
                    run.log({f"cost/{k}": v for k, v in rec.items() if isinstance(v, (int, float))})
                    run.finish()
        except torch.cuda.OutOfMemoryError:
            print("  OOM — pulando SAM-H")
            torch.cuda.empty_cache()

    out_dir = os.path.join(REPO_ROOT, "exp/out")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "e1_encoder_cost.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "e0_environment.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)
    print(f"\n-> {csv_path}")

    print("\n=== CRITERIOS DE ACEITACAO ===")
    fails = check_criteria(records)
    if fails:
        for msg in fails:
            print(f"  FALHOU: {msg}")
        print("\n  >>> PARAR E INVESTIGAR — resultado contradiz o criterio declarado no guia.")
    else:
        print("  todos passaram")

    if wandb:
        run = wandb.init(entity=args.entity, project=args.project, group="E1-cost",
                         name=f"e1__summary__s{args.seed}", tags=["E1", "summary"],
                         config=env, reinit=True)
        cols = list(records[0].keys())
        run.log({"E1_table": wandb.Table(columns=cols,
                                         data=[[r.get(c) for c in cols] for r in records])})
        run.summary["criteria_failures"] = fails
        run.finish()


if __name__ == "__main__":
    main()
