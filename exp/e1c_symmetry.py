"""
E1c — simetria de otimização de baixo nível.

Mede o efeito dos kernels especializados dos DOIS lados, para que a comparação de
custo do E1 não fique dependendo de quem tem a melhor engenharia de kernel.

  DINOv2-L : atenção eficiente (mem-efficient SDPA) vs MATH (materializa n×n).
             xFormers NÃO está instalado; o caminho eficiente é o SDPA nativo do
             PyTorch. O par ON/OFF do guia vira EFFICIENT vs MATH, que é mais
             informativo: MATH é a atenção ingênua de verdade.
  VMamba-T : selective scan em kernel CUDA vs referência em PyTorch puro
             (selective_scan_backend="torch" + scan_force_torch=True), mantendo
             todo o resto de v05_noz igual.

OOM é resultado válido, não falha: "não coube em 32 GB" é a resposta.

Uso:
    python -m exp.e1c_symmetry
    python -m exp.e1c_symmetry --sizes 1024 --no-wandb
"""
import argparse
import csv
import os
import sys
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.e1_encoder_cost import build_model, _clear_cuda, WARMUP, REPEATS
from exp.envinfo import collect, REPO_ROOT

# fp32 não tem FlashAttention; o caminho eficiente real é o mem-efficient.
EFFICIENT_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
MATH_BACKENDS = [SDPBackend.MATH]


def set_scan_backend(model, backend):
    """
    Troca o caminho do selective scan mantendo o resto de v05_noz idêntico.
    Devolve quantos blocos foram alterados (0 = algo mudou no VMamba e o
    experimento não é válido).
    """
    n = 0
    for m in model.encoder.modules():
        if type(m).__name__ == "SS2D" and hasattr(m, "forward_corev2"):
            if backend == "torch":
                m.forward_core = partial(
                    m.forward_corev2, force_fp32=False, no_einsum=True,
                    selective_scan_backend="torch", scan_force_torch=True)
            else:
                m.forward_core = partial(
                    m.forward_corev2, force_fp32=False, no_einsum=True)
            n += 1
    return n


@torch.no_grad()
def bench(modelname, size, condition, device):
    base = _clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(modelname, size).to(device).eval()
    torch.cuda.synchronize(device)
    vram_weights = (torch.cuda.memory_allocated(device) - base) / 1024 ** 2

    n_patched = None
    if "vmamba" in modelname:
        n_patched = set_scan_backend(model, "torch" if condition == "scan_torch" else "cuda")
        if n_patched == 0:
            raise RuntimeError("nenhum bloco SS2D encontrado — patch do scan nao aplicou")
        ctx = nullcontext
    else:
        backends = MATH_BACKENDS if condition == "attn_math" else EFFICIENT_BACKENDS
        ctx = partial(sdpa_kernel, backends)

    x = torch.randn(1, 3, size, size, device=device)

    with ctx():
        for _ in range(WARMUP):
            model.get_features(x)
        torch.cuda.synchronize(device)

        _clear_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        resident = torch.cuda.memory_allocated(device)
        out = model.get_features(x)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        vram_activation = (peak - resident) / 1024 ** 2
        del out

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
        for i in range(REPEATS):
            starts[i].record()
            model.get_features(x)
            ends[i].record()
        torch.cuda.synchronize(device)
        times = np.array([starts[i].elapsed_time(ends[i]) for i in range(REPEATS)])

    rec = {
        "encoder": modelname, "input_size": size, "condition": condition,
        "status": "ok", "blocks_patched": n_patched,
        "latency_ms_median": round(float(np.median(times)), 3),
        "latency_ms_p25": round(float(np.percentile(times, 25)), 3),
        "latency_ms_p75": round(float(np.percentile(times, 75)), 3),
        "vram_weights_mb": round(vram_weights, 1),
        "vram_activation_mb": round(vram_activation, 1),
        "vram_total_mb": round(peak / 1024 ** 2, 1),
    }
    del model, x
    _clear_cuda(device)
    return rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[512, 1024])
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
    print(f"=== E1c | {env.get('gpu_name')} | torch {env.get('torch')} | "
          f"commit {str(env.get('git_commit'))[:8]} | xformers {env.get('xformers')} ===\n")

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb

    jobs = []
    for size in args.sizes:
        jobs.append(("dinov2_l14", size, "attn_efficient"))
        jobs.append(("dinov2_l14", size, "attn_math"))
        jobs.append(("vmamba_tiny", size, "scan_cuda"))
        jobs.append(("vmamba_tiny", size, "scan_torch"))

    records = []
    for modelname, size, condition in jobs:
        print(f"[{modelname} @ {size} | {condition}] ...", flush=True)
        try:
            rec = bench(modelname, size, condition, device)
            print(f"  {rec['latency_ms_median']:.2f} ms "
                  f"[{rec['latency_ms_p25']:.2f}-{rec['latency_ms_p75']:.2f}] | "
                  f"ativacao {rec['vram_activation_mb']:.0f} MB")
        except torch.cuda.OutOfMemoryError:
            # OOM é resultado, não falha: a atenção ingenua nao cabe.
            print("  OOM — nao coube na GPU (isto E o resultado)")
            rec = {"encoder": modelname, "input_size": size, "condition": condition,
                   "status": "OOM", "blocks_patched": None,
                   "latency_ms_median": None, "latency_ms_p25": None,
                   "latency_ms_p75": None, "vram_weights_mb": None,
                   "vram_activation_mb": None, "vram_total_mb": None}
            _clear_cuda(device)
        except Exception as e:
            print(f"  ERRO: {type(e).__name__}: {e}")
            _clear_cuda(device)
            continue
        records.append(rec)

        if wandb:
            run = wandb.init(
                entity=args.entity, project=args.project, group="E1c-symmetry",
                name=f"e1c__{modelname}__{size}__{condition}__s{args.seed}",
                tags=[modelname, "E1c", condition, f"size{size}"],
                config={**env, **rec, "dtype": "fp32", "batch_size": 1, "seed": args.seed},
                reinit=True,
            )
            run.log({f"sym/{k}": v for k, v in rec.items() if isinstance(v, (int, float))})
            run.finish()

    out_dir = os.path.join(REPO_ROOT, "exp/out")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "e1c_symmetry.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\n-> {csv_path}")

    print("\n=== T3 — SIMETRIA ===")
    pairs = [("dinov2_l14", "attn_efficient", "attn_math"),
             ("vmamba_tiny", "scan_cuda", "scan_torch")]
    for enc, on, off in pairs:
        for size in args.sizes:
            a = next((r for r in records if r["encoder"] == enc and r["input_size"] == size
                      and r["condition"] == on), None)
            b = next((r for r in records if r["encoder"] == enc and r["input_size"] == size
                      and r["condition"] == off), None)
            if not a or not b:
                continue
            if b["status"] == "OOM":
                print(f"  {enc}@{size}: {off} NAO COUBE em 32 GB "
                      f"({on}: {a['vram_activation_mb']:.0f} MB de ativacao)")
                continue
            if a["status"] == "OOM":
                continue
            dl = b["latency_ms_median"] / a["latency_ms_median"]
            dm = b["vram_activation_mb"] - a["vram_activation_mb"]
            print(f"  {enc}@{size}: {off} e {dl:.2f}x mais lento e usa "
                  f"{dm:+.0f} MB de ativacao ({a['vram_activation_mb']:.0f} -> "
                  f"{b['vram_activation_mb']:.0f})")

    if wandb:
        run = wandb.init(entity=args.entity, project=args.project, group="E1c-symmetry",
                         name=f"e1c__summary__s{args.seed}", tags=["E1c", "summary"],
                         config=env, reinit=True)
        cols = list(records[0].keys())
        run.log({"E1c_table": wandb.Table(columns=cols,
                                          data=[[r.get(c) for c in cols] for r in records])})
        run.finish()


if __name__ == "__main__":
    main()
