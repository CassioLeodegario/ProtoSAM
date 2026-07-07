"""
Benchmark do encoder isolado via get_features() — o ponto exato que o ALPNet consome.
Mede latência (média ± desvio) e VRAM de inferência em função da resolução.

Uso:
    python benchmark_encoder_v2.py
    python benchmark_encoder_v2.py --modelname dinov2_l14 --modelname vmamba_tiny
"""
import argparse
import csv
import time

import numpy as np
import torch
import wandb

from models.grid_proto_fewshot import FewShotSeg

SIZES  = [256, 384, 512, 672, 768, 1024]
WARMUP  = 10
REPEATS = 40


def build_model(modelname: str, image_size: int) -> FewShotSeg:
    cfg = {
        "align": False,
        "use_coco_init": False,
        "which_model": modelname,
        "cls_name": "grid_proto",
        "proto_grid_size": 8,
        "feature_hw": [image_size // 8, image_size // 8],
        "reload_model_path": None,
        "lora": 0,
        "use_slice_adapter": False,
        "adapter_layers": 3,
        "debug": False,
        "use_pos_enc": False,
    }
    return FewShotSeg(image_size=image_size, pretrained_path=None, cfg=cfg)


@torch.no_grad()
def benchmark_one(modelname: str, device: torch.device, wandb_project: str) -> list:
    records = []
    for size in SIZES:
        model = build_model(modelname, size).to(device).eval()
        x = torch.randn(1, 3, size, size, device=device)

        # warmup
        for _ in range(WARMUP):
            model.get_features(x)
        torch.cuda.synchronize(device)

        # VRAM só de inferência (pesos já carregados)
        torch.cuda.reset_peak_memory_stats(device)
        model.get_features(x)
        torch.cuda.synchronize(device)
        vram_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        feat_shape = tuple(model.get_features(x).shape)

        # latência por iteração
        times = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model.get_features(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        lat_mean, lat_std = float(times.mean()), float(times.std())
        n_params = sum(p.numel() for p in model.encoder.parameters())

        print(f"[{modelname}] size={size:>4}  feat={feat_shape}  "
              f"lat={lat_mean:.1f}±{lat_std:.1f} ms  "
              f"vram={vram_mb:.1f} MB  params={n_params/1e6:.1f}M")

        records.append({
            "modelname":         modelname,
            "input_size":        size,
            "feat_shape":        str(feat_shape),
            "latency_ms_mean":   round(lat_mean, 3),
            "latency_ms_std":    round(lat_std, 3),
            "throughput_fps":    round(1000.0 / lat_mean, 3),
            "vram_infer_mb":     round(vram_mb, 1),
            "encoder_params_M":  round(n_params / 1e6, 2),
        })

        del model, x
        torch.cuda.empty_cache()

    # CSV
    fname = f"encoder_bench_{modelname}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"  -> {fname}")

    # W&B
    wandb.init(
        project=wandb_project,
        entity="leodegario",
        name=f"{modelname}_encoder_bench",
        config={"modelname": modelname, "warmup": WARMUP, "repeats": REPEATS},
        tags=[modelname, "encoder-only", "benchmark"],
        reinit=True,
    )
    for r in records:
        wandb.log({
            "input_size":     r["input_size"],
            "latency_ms":     r["latency_ms_mean"],
            "latency_ms_std": r["latency_ms_std"],
            "throughput_fps": r["throughput_fps"],
            "vram_infer_mb":  r["vram_infer_mb"],
        })
    wandb.finish()

    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modelname", action="append", dest="models", default=None)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--wandb-project", default="protosam-polyp-sizes")
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global SIZES
    if args.sizes:
        SIZES = args.sizes

    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)

    for modelname in (args.models or ["dinov2_l14", "vmamba_tiny"]):
        benchmark_one(modelname, device, args.wandb_project)


if __name__ == "__main__":
    main()
