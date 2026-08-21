"""
E1b — o encoder está limitado por computação ou por lançamento de kernel?

O E1 já mostrou o sintoma: os FLOPs do VMamba-T crescem 16x de 256 para 1024 px
e a latência não se move. Aqui a causa é medida por dois caminhos independentes,
que precisam concordar:

  1. PROFILER   — tempo de GPU (self_device_time_total) contra tempo de parede.
                  Razão baixa = a GPU passou a maior parte do tempo ociosa,
                  esperando a CPU enfileirar trabalho.
  2. BATCH      — se o gargalo é lançamento, aumentar o batch quase não muda o
                  tempo total, e o tempo por imagem despenca.

Também conta os lançamentos de kernel por iteração, que é a grandeza física por
trás da hipótese.

Uso:
    python -m exp.e1b_launchbound
    python -m exp.e1b_launchbound --models vmamba_tiny --sizes 512 --batches 1 4
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.grid_proto_fewshot import FewShotSeg
from exp.envinfo import collect, REPO_ROOT

WARMUP = 20
REPEATS = 50
PROFILE_ITERS = 20


def build_model(modelname, image_size):
    cfg = {
        "align": False, "use_coco_init": False, "which_model": modelname,
        "cls_name": "grid_proto", "proto_grid_size": 8,
        "feature_hw": [image_size // 8, image_size // 8],
        "reload_model_path": None, "lora": 0, "use_slice_adapter": False,
        "adapter_layers": 3, "debug": False, "use_pos_enc": False,
    }
    return FewShotSeg(image_size=image_size, pretrained_path=None, cfg=cfg)


def _clear_cuda(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _device_time(evt):
    """torch 2.8 renomeou self_cuda_time_total -> self_device_time_total."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


@torch.no_grad()
def measure(modelname, size, batch, device):
    _clear_cuda(device)
    model = build_model(modelname, size).to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)

    for _ in range(WARMUP):
        model.get_features(x)
    torch.cuda.synchronize(device)

    # --- tempo de parede, SEM profiler (o profiler infla o relógio) ---
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        starts[i].record()
        model.get_features(x)
        ends[i].record()
    torch.cuda.synchronize(device)
    wall = np.array([starts[i].elapsed_time(ends[i]) for i in range(REPEATS)])
    wall_ms = float(np.median(wall))

    # --- tempo de GPU e nº de lançamentos, COM profiler ---
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_ITERS):
            model.get_features(x)
        torch.cuda.synchronize(device)

    events = prof.key_averages()
    cuda_us = sum(_device_time(e) for e in events)
    n_kernels = sum(e.count for e in events if _device_time(e) > 0)
    cuda_ms = cuda_us / 1000.0 / PROFILE_ITERS
    kernels_per_iter = n_kernels / PROFILE_ITERS

    rec = {
        "encoder": modelname,
        "input_size": size,
        "batch": batch,
        "wall_total_ms": round(wall_ms, 3),
        "wall_per_image_ms": round(wall_ms / batch, 3),
        "cuda_time_ms": round(cuda_ms, 3),
        "cuda_over_wall": round(cuda_ms / wall_ms, 4) if wall_ms else None,
        "kernels_per_iter": round(kernels_per_iter, 1),
        "kernels_per_image": round(kernels_per_iter / batch, 1),
        "wall_ms_p25": round(float(np.percentile(wall, 25)), 3),
        "wall_ms_p75": round(float(np.percentile(wall, 75)), 3),
    }
    del model, x
    _clear_cuda(device)
    return rec


def verdict(records):
    """Leitura do resultado, conforme o guia."""
    lines = []
    for enc in sorted({r["encoder"] for r in records}):
        b1 = [r for r in records if r["encoder"] == enc and r["batch"] == 1]
        if not b1:
            continue
        ratios = [r["cuda_over_wall"] for r in b1 if r["cuda_over_wall"]]
        if not ratios:
            continue
        m = float(np.mean(ratios))
        if m < 0.4:
            lines.append(f"{enc}: CUDA/Wall medio = {m:.3f} (<0.40) -> LIMITADO POR LANCAMENTO")
        elif m > 0.8:
            lines.append(f"{enc}: CUDA/Wall medio = {m:.3f} (>0.80) -> limitado por computacao")
        else:
            lines.append(f"{enc}: CUDA/Wall medio = {m:.3f} -> regime intermediario")

    for enc in sorted({r["encoder"] for r in records}):
        for size in sorted({r["input_size"] for r in records if r["encoder"] == enc}):
            sub = sorted([r for r in records if r["encoder"] == enc and r["input_size"] == size],
                         key=lambda r: r["batch"])
            if len(sub) < 2:
                continue
            first, last = sub[0], sub[-1]
            drop = 1 - last["wall_per_image_ms"] / first["wall_per_image_ms"]
            grow = last["wall_total_ms"] / first["wall_total_ms"]
            lines.append(
                f"{enc}@{size}: batch {first['batch']}->{last['batch']} | "
                f"tempo/imagem cai {drop*100:.0f}% | tempo total cresce {grow:.2f}x "
                f"(x{last['batch']//first['batch']} de trabalho)")
    return lines


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vmamba_tiny", "dinov2_l14"])
    p.add_argument("--sizes", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--project", default="qualification")
    p.add_argument("--entity", default="leodegario")
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    torch.manual_seed(args.seed)

    env = collect()
    print(f"=== E1b | {env.get('gpu_name')} | torch {env.get('torch')} | "
          f"commit {str(env.get('git_commit'))[:8]} | dirty={env.get('git_dirty')} ===\n")

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb

    records = []
    for modelname in args.models:
        for size in args.sizes:
            for batch in args.batches:
                print(f"[{modelname} @ {size} batch={batch}] ...", flush=True)
                try:
                    rec = measure(modelname, size, batch, device)
                except torch.cuda.OutOfMemoryError:
                    print("  OOM — pulando")
                    _clear_cuda(device)
                    continue
                records.append(rec)
                print(f"  wall {rec['wall_total_ms']:.2f} ms | /imagem {rec['wall_per_image_ms']:.2f} ms "
                      f"| cuda {rec['cuda_time_ms']:.2f} ms | CUDA/Wall {rec['cuda_over_wall']:.3f} "
                      f"| {rec['kernels_per_iter']:.0f} kernels/iter")

                if wandb:
                    run = wandb.init(
                        entity=args.entity, project=args.project, group="E1b-launchbound",
                        name=f"e1b__{modelname}__{size}__b{batch}__s{args.seed}",
                        tags=[modelname, "E1b", f"size{size}", f"batch{batch}"],
                        config={**env, **rec, "dtype": "fp32", "warmup": WARMUP,
                                "repeats": REPEATS, "profile_iters": PROFILE_ITERS,
                                "seed": args.seed},
                        reinit=True,
                    )
                    run.log({f"launch/{k}": v for k, v in rec.items() if isinstance(v, (int, float))})
                    run.finish()

    out_dir = os.path.join(REPO_ROOT, "exp/out")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "e1b_launchbound.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\n-> {csv_path}")

    print("\n=== LEITURA DO RESULTADO ===")
    for line in verdict(records):
        print(f"  {line}")

    if wandb:
        run = wandb.init(entity=args.entity, project=args.project, group="E1b-launchbound",
                         name=f"e1b__summary__s{args.seed}", tags=["E1b", "summary"],
                         config=env, reinit=True)
        cols = list(records[0].keys())
        run.log({"E1b_table": wandb.Table(columns=cols,
                                          data=[[r.get(c) for c in cols] for r in records])})
        run.summary["verdict"] = verdict(records)
        run.finish()


if __name__ == "__main__":
    main()
