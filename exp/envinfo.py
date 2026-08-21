"""
Impressão digital do ambiente de execução (E0).

Não mede nada — declara. Todo experimento importa isto e despeja o resultado na
config da run do W&B, de modo que nenhuma medição fique órfã de contexto.

Uso:
    from exp.envinfo import collect, dump
    cfg = collect()                      # dict pronto para wandb.init(config=...)
    dump("exp/out/e0_environment.json")  # + arquivo para subir como artifact
"""
import json
import os
import subprocess
import sys
import warnings

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd):
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_info():
    commit = _run(["git", "rev-parse", "HEAD"])
    tracked = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    untracked = _run(["git", "status", "--porcelain", "--untracked-files=all"])
    n_untracked = 0
    if untracked is not None and tracked is not None:
        n_untracked = len([l for l in untracked.splitlines() if l.startswith("??")])
    return {
        "git_commit": commit,
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        # dirty = arquivos RASTREADOS modificados. Arquivos novos não versionados
        # (caderno, saídas) são contados à parte para não poluir o sinal.
        "git_dirty": bool(tracked) if tracked is not None else None,
        "git_untracked_files": n_untracked,
    }


def gpu_info():
    if not torch.cuda.is_available():
        return {"gpu_name": None}
    p = torch.cuda.get_device_properties(0)
    smi = _run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,persistence_mode",
                "--format=csv,noheader,nounits"])
    driver = clk = persist = None
    if smi:
        parts = [x.strip() for x in smi.splitlines()[0].split(",")]
        if len(parts) == 3:
            driver, clk, persist = parts
    return {
        "gpu_name": p.name,
        "gpu_total_mem_mb": round(p.total_memory / 1024 ** 2, 1),
        "gpu_capability": f"{p.major}.{p.minor}",
        "driver": driver,
        "gpu_clock_max_sm_mhz": clk,
        "persistence_mode": persist,
    }


def lib_versions():
    import importlib
    out = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    for m in ["torchvision", "timm", "numpy", "wandb", "fvcore", "xformers", "monai"]:
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "installed")
        except Exception:
            out[m] = None
    return out


def backend_flags():
    return {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "num_threads": torch.get_num_threads(),
        "XFORMERS_DISABLED_env": os.environ.get("XFORMERS_DISABLED"),
    }


def selective_scan_info():
    """Qual caminho do selective scan do VMamba está disponível."""
    try:
        import models.vmamba as V
    except Exception as e:
        return {"selective_scan_error": f"{type(e).__name__}: {e}"}
    names = ["selective_scan_fn", "selective_scan_cuda", "selective_scan_cuda_core",
             "selective_scan_cuda_oflex", "selective_scan_cuda_ndstate", "selective_scan_cuda_nrow"]
    avail = {n: (getattr(V, n, None) is not None) for n in names}
    avail["selective_scan_kernel"] = "cuda" if avail.get("selective_scan_cuda") else "reference/pytorch"
    return avail


def attention_info(dinov2_model=None):
    """
    Classe de atenção efetivamente usada pelo DINOv2.
    Passe um modelo já construído para evitar baixar os pesos só para isto.
    """
    info = {"attn_class": None, "xformers_available": None}
    if dinov2_model is None:
        return info
    try:
        attn = dinov2_model.blocks[0].attn
        info["attn_class"] = type(attn).__name__
        mod = sys.modules[type(attn).__module__]
        info["xformers_available"] = getattr(mod, "XFORMERS_AVAILABLE", None)
        src = getattr(type(attn), "forward").__code__.co_names
        info["attn_uses_sdpa"] = "scaled_dot_product_attention" in src or any(
            "scaled_dot_product_attention" in n for n in src)
    except Exception as e:
        info["attn_error"] = f"{type(e).__name__}: {e}"
    return info


def collect(dinov2_model=None, extra=None):
    cfg = {}
    cfg.update(git_info())
    cfg.update(gpu_info())
    cfg.update(lib_versions())
    cfg.update(backend_flags())
    cfg.update(selective_scan_info())
    cfg.update(attention_info(dinov2_model))
    if extra:
        cfg.update(extra)
    return cfg


def dump(path="exp/out/e0_environment.json", dinov2_model=None, extra=None):
    cfg = collect(dinov2_model=dinov2_model, extra=extra)
    path = os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return cfg


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = dump()
        cfg["warnings_on_import"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    for k, v in cfg.items():
        print(f"{k:28s} {v}")
