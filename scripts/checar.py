"""Checagem antes de rodar experimentos: código oficial intacto, travas e números de referência.

    python scripts/checar.py            # tudo; a última parte usa cerca de 2 minutos de GPU
    python scripts/checar.py --rapido   # sem a parte de GPU
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TESTES = ['test_oficial_intacto.py', 'test_travas.py', 'test_tabelas.py', 'test_referencias.py']

for teste in TESTES[:3] if '--rapido' in sys.argv else TESTES:
    print(f'== {teste}', flush=True)
    if subprocess.run([sys.executable, str(RAIZ / 'tests' / teste)]).returncode:
        sys.exit(f'CHECAGEM FALHOU em {teste}')
print('CHECAGEM OK')
