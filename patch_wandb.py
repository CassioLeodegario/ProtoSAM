"""
Patch validation_protosam.py to integrate Weights & Biases logging
"""
filepath = "validation_protosam.py"
with open(filepath, "r") as f:
    code = f.read()

# === PATCH 1: Add wandb import ===
old_import = "import tqdm\nfrom tqdm.auto import tqdm"
new_import = "import wandb\nimport tqdm\nfrom tqdm.auto import tqdm"
assert old_import in code, "PATCH 1 FAILED"
code = code.replace(old_import, new_import)
print("PATCH 1 OK: wandb import added")

# === PATCH 2: Add wandb.init at start of main() ===
old_main = '''    print(f"config do_cca: {_config['do_cca']}, use_bbox: {_config['use_bbox']}")'''
new_main = '''    print(f"config do_cca: {_config['do_cca']}, use_bbox: {_config['use_bbox']}")
    
    # === W&B Experiment Tracking ===
    wandb.init(
        project="protosam-polyp",
        entity="leodegario",
        config={
            "backbone": _config.get("modelname", "unknown"),
            "sam_version": _config.get("protosam_sam_ver", "unknown"),
            "dataset": _config.get("dataset", "unknown"),
            "input_size": _config.get("input_size", "unknown"),
            "proto_grid": _config.get("proto_grid_size", "unknown"),
            "eval_fold": _config.get("eval_fold", 0),
            "support_idx": _config.get("support_idx", "unknown"),
            "seed": _config.get("seed", 42),
            "do_cca": _config.get("do_cca", False),
            "use_align": _config.get("usealign", False),
            "coarse_pred_only": _config.get("coarse_pred_only", False),
            "lora": _config.get("lora", 0),
        },
        tags=[_config.get("modelname", ""), _config.get("dataset", ""), _config.get("protosam_sam_ver", "")],
        reinit=True,
    )
    _start_time = time.time()'''

assert old_main in code, "PATCH 2 FAILED"
code = code.replace(old_main, new_main)
print("PATCH 2 OK: wandb.init added")

# === PATCH 3: Add wandb.log after metrics are computed ===
old_metrics = """    _log.info(f'mar_val batches meanDice: {m_meanDice}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')
    _log.info(f'mar_val batches meanIOU: {m_meanIOU}')"""

new_metrics = """    _log.info(f'mar_val batches meanDice: {m_meanDice}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')
    _log.info(f'mar_val batches meanIOU: {m_meanIOU}')
    
    # === W&B: Log final metrics ===
    _total_time = time.time() - _start_time
    _gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    wandb.log({
        "mean_dice": m_meanDice,
        "mean_iou": m_meanIOU,
        "mean_precision": m_meanPrec,
        "mean_recall": m_meanRec,
        "total_time_seconds": _total_time,
        "gpu_peak_memory_mb": _gpu_mem_mb,
        "num_test_images": len(mean_dice),
        "throughput_img_per_sec": len(mean_dice) / _total_time,
    })
    wandb.finish()"""

assert old_metrics in code, "PATCH 3 FAILED"
code = code.replace(old_metrics, new_metrics)
print("PATCH 3 OK: wandb.log and wandb.finish added")

with open(filepath, "w") as f:
    f.write(code)

print("\nAll patches applied successfully!")
