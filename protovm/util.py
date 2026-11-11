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


# Pastas que definem o que um número significa. Alteração sem commit em qualquer uma marca a avaliação.
CODIGO = ('protovm', 'models', 'dataloaders', 'util', 'scripts', 'protocolo', 'configs')


def git_commit():
    """Commit atual e se há alterações não commitadas em código, protocolo ou configs.

    Se o git não responder, o código conta como alterado: não dá para provar que está commitado.
    """
    def git(*args):
        try:
            r = subprocess.run(['git', '-C', str(RAIZ), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain', '--', *CODIGO)
    return {'commit': commit, 'codigo_alterado': not commit or status is None or bool(status)}


def estado_do_codigo(antes, depois):
    """Estado de uma linha do registro: 'novo' se o código ficou commitado e igual do início ao
    fim da avaliação (antes e depois vêm de git_commit()); 'rascunho' em qualquer outro caso."""
    limpo = (bool(antes['commit']) and antes['commit'] == depois['commit']
             and not antes['codigo_alterado'] and not depois['codigo_alterado'])
    return 'novo' if limpo else 'rascunho'


def nome_do_treino(checkpoint):
    """Nome do treino de um checkpoint avulso: a pasta do arquivo, contada a partir de
    reproducao/ ou runs/ quando houver (ex.: 'eft_SAM_vs_SP256_x/treino_SAM')."""
    partes = Path(checkpoint).parent.parts
    for i in range(len(partes) - 1, -1, -1):
        if partes[i] in ('reproducao', 'runs') and i + 1 < len(partes):
            return '/'.join(partes[i + 1:])
    return partes[-1]


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
