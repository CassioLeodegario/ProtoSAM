
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import os
import time

PASTA = Path(
    "/home/1542521/cassio_pereira/ProtoSAM-repro/"
    "data/CHAOST2/downloads"
)

URL = (
    "https://zenodo.org/records/3431873/files/"
    "CHAOS_Train_Sets.zip?download=1"
)

ZIP = PASTA / "CHAOS_Train_Sets.zip"
PARCIAL = PASTA / "CHAOS_Train_Sets.zip.part"
STATUS = PASTA / "status_download.json"

MD5_ESPERADO = "df21053002a1cc86df918a87da3b2c19"
TAMANHO_ESPERADO = 890771694

def registrar(estado, **dados):
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "estado": estado,
        **dados
    }, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)

try:
    if not ZIP.exists():
        for tentativa in range(1, 6):
            atual = PARCIAL.stat().st_size if PARCIAL.exists() else 0

            if atual == TAMANHO_ESPERADO:
                break

            try:
                headers = {"User-Agent": "ProtoSAM-Research/1.0"}

                if atual > 0:
                    headers["Range"] = f"bytes={atual}-"

                req = Request(URL, headers=headers)

                with urlopen(req, timeout=90) as resposta:
                    codigo = resposta.status

                    if codigo == 206:
                        intervalo = resposta.headers.get(
                            "Content-Range", ""
                        )
                        assert intervalo.startswith(f"bytes {atual}-"), (
                            f"Resposta Range inesperada: {intervalo}"
                        )
                        modo = "ab"

                    elif codigo == 200:
                        # Se o servidor ignorar Range,
                        # reiniciar a transferência corretamente.
                        modo = "wb"
                        atual = 0

                    else:
                        raise RuntimeError(f"HTTP inesperado: {codigo}")

                    with PARCIAL.open(modo) as arquivo:
                        ultimo_registro = atual

                        while True:
                            bloco = resposta.read(4 * 1024 * 1024)

                            if not bloco:
                                break

                            arquivo.write(bloco)
                            atual += len(bloco)

                            if atual - ultimo_registro >= 64 * 1024 * 1024:
                                registrar(
                                    "baixando",
                                    bytes_baixados=atual,
                                    total_bytes=TAMANHO_ESPERADO,
                                    percentual=round(
                                        100 * atual / TAMANHO_ESPERADO, 2
                                    ),
                                    tentativa=tentativa
                                )
                                print(
                                    f"Download: "
                                    f"{100 * atual / TAMANHO_ESPERADO:.1f}%",
                                    flush=True
                                )
                                ultimo_registro = atual

                if PARCIAL.stat().st_size == TAMANHO_ESPERADO:
                    break

                raise RuntimeError("Transferência incompleta")

            except Exception as e:
                print(
                    f"Tentativa {tentativa}/5 falhou: {e}",
                    flush=True
                )

                if tentativa == 5:
                    raise

                time.sleep(5)

        arquivo = PARCIAL
    else:
        arquivo = ZIP

    registrar("verificando_md5")

    assert arquivo.stat().st_size == TAMANHO_ESPERADO, (
        "Tamanho do arquivo não corresponde ao original."
    )

    h = hashlib.md5()

    with arquivo.open("rb") as f:
        for bloco in iter(
            lambda: f.read(4 * 1024 * 1024), b""
        ):
            h.update(bloco)

    digest = h.hexdigest()

    assert digest == MD5_ESPERADO, (
        f"MD5 incorreto: {digest}"
    )

    if arquivo == PARCIAL:
        os.replace(PARCIAL, ZIP)

    registrar(
        "concluido",
        arquivo=str(ZIP),
        tamanho_bytes=ZIP.stat().st_size,
        md5=digest
    )

    print("DOWNLOAD CONCLUÍDO E MD5 VERIFICADO", flush=True)

except Exception as e:
    registrar("erro", mensagem=str(e))
    raise
