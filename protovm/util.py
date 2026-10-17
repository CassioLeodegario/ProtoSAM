"""Utilitários de registro: hashes, JSON atômico, ambiente e commit."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess

from protovm import RAIZ


def sha256(caminho):
    h = hashlib.sha256()
    with Path(caminho).open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def assinatura(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ler_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def gravar_json(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def agora():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def semear(semente):
    import numpy as np
    import torch
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def fixar_numerica():
    """Mesmas opções numéricas de todos os experimentos de out/2026."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def git_commit():
    """Commit atual e se há alterações não commitadas no código."""
    def git(*args):
        r = subprocess.run(['git', '-C', str(RAIZ), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = git('status', '--porcelain', '--', 'protovm', 'models', 'dataloaders', 'util', 'scripts')
    return {'commit': git('rev-parse', 'HEAD'), 'codigo_alterado': bool(status)}


def ambiente():
    import torch
    versoes = {}
    for nome in ['numpy', 'Pillow', 'scipy', 'scikit-image', 'torchvision', 'segment-anything']:
        try:
            versoes[nome] = importlib.metadata.version(nome)
        except importlib.metadata.PackageNotFoundError:
            versoes[nome] = None
    gpu = driver = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                           capture_output=True, text=True)
        driver = r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': gpu, 'driver': driver,
            'hostname': platform.node(), **versoes}
