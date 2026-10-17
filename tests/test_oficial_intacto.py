"""O código oficial do ProtoSAM não pode mudar: substitui as ~40 células de hash dos notebooks.

    python -m pytest tests/  (ou: python tests/test_oficial_intacto.py)
"""
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BASE_OFICIAL = '64868c909668aac9ad517f7192a2712e88268e8f'
VMAMBA_SHA256 = '572207a2b6553ff85d81b602eebda2041623073f01fc0c6d1280c5a2add040f6'
OFICIAIS = ['models', 'dataloaders', 'util', 'validation_protosam.py', 'config_ssl_upload.py', 'run_protosam.sh']


def test_codigo_oficial_igual_ao_commit_base():
    r = subprocess.run(['git', '-C', str(RAIZ), 'diff', '--stat', BASE_OFICIAL, '--', *OFICIAIS,
                        ':(exclude)models/vmamba.py'], capture_output=True, text=True, check=True)
    assert r.stdout == '', f'Código oficial alterado em relação a {BASE_OFICIAL[:7]}:\n{r.stdout}'


def test_vmamba_e_a_versao_registrada():
    import hashlib
    digest = hashlib.sha256((RAIZ / 'models' / 'vmamba.py').read_bytes()).hexdigest()
    assert digest == VMAMBA_SHA256, 'models/vmamba.py difere da versão usada em out/2026.'


if __name__ == '__main__':
    test_codigo_oficial_igual_ao_commit_base()
    test_vmamba_e_a_versao_registrada()
    print('OK: código oficial intacto e vmamba.py na versão registrada.')
