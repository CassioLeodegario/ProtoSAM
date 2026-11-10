
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import felzenszwalb
from skimage.measure import label
import SimpleITK as sitk
import numpy as np
import json
import os
import traceback

REPO = Path("/home/1542521/cassio_pereira/ProtoSAM-repro")
EXP = REPO / "exp/eft_mri_chaos_v1"
DADOS = REPO / "data/CHAOST2/chaos_MR_T2_normalized"

STATUS = EXP / "status_superpixels.json"
MANIFESTO = EXP / "manifesto_superpixels.json"

LIMIAR = 50.0001
MIN_SIZE = 400
SIGMA = 1
MODE = "MIDDLE"


def salvar_json(caminho, dados):
    tmp = caminho.with_name(caminho.name + ".tmp")
    tmp.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(tmp, caminho)


def mascara_foreground(imagem):
    # Reprodução do fg_mask2d original.
    mascara = (imagem > LIMIAR).astype(np.float32)

    if mascara.max() < 0.999:
        return np.zeros(imagem.shape, dtype=np.uint8)

    componentes = label(mascara)

    contagens = np.bincount(componentes.ravel())[1:]
    maior = np.argmax(contagens) + 1

    mascara = componentes == maior
    mascara = binary_fill_holes(mascara)

    return mascara.astype(np.uint8)


def aplicar_mascara_original(superpixels, foreground):
    # Preserva o remapeamento do superpix_masking
    # presente no notebook original.
    regioes = np.int32(superpixels).copy()
    valores = np.unique(regioes)
    maior_valor = int(valores.max())

    regioes[regioes == 0] = maior_valor + 1

    valores = list(valores)
    valores.append(maior_valor)

    regioes = regioes * foreground.astype(np.int32)

    saida = np.zeros(regioes.shape, dtype=np.int32)
    proximo_rotulo = 1

    for valor in valores:
        if valor == 0:
            continue

        saida[regioes == valor] = proximo_rotulo
        proximo_rotulo += 1

    return saida


def salvar_nifti(array, referencia, destino):
    obj = sitk.GetImageFromArray(array)
    obj.CopyInformation(referencia)

    if destino.exists():
        anterior = sitk.ReadImage(str(destino))

        assert anterior.GetSize() == obj.GetSize()
        assert np.allclose(
            anterior.GetSpacing(), obj.GetSpacing()
        )
        assert np.allclose(
            anterior.GetOrigin(), obj.GetOrigin()
        )
        assert np.allclose(
            anterior.GetDirection(), obj.GetDirection()
        )
        assert np.array_equal(
            sitk.GetArrayFromImage(anterior), array
        )
        return "verificado"

    tmp = destino.with_name(
        destino.name.replace(".nii.gz", ".tmp.nii.gz")
    )

    sitk.WriteImage(obj, str(tmp), True)
    os.replace(tmp, destino)
    return "criado"


def executar():
    total_fatias = 0
    pacientes_concluidos = 0
    registros = []

    for pid in range(20):
        imagem_path = DADOS / f"image_{pid}.nii.gz"
        fg_path = DADOS / f"fgmask_{pid}.nii.gz"
        sp_path = DADOS / f"superpix-{MODE}_{pid}.nii.gz"

        imagem_obj = sitk.ReadImage(str(imagem_path))
        volume = sitk.GetArrayFromImage(imagem_obj)

        assert volume.ndim == 3
        assert volume.shape[1:] == (256, 256)
        assert np.isfinite(volume).all()

        z, h, w = volume.shape

        fg_volume = np.zeros((z, h, w), dtype=np.uint8)
        sp_volume = np.zeros((z, h, w), dtype=np.int32)

        regioes_total = 0
        fatias_com_foreground = 0

        for fatia in range(z):
            img2d = volume[fatia]

            # Parâmetros originais: scale padrão,
            # min_size=400, sigma=1.
            segmentos = felzenszwalb(
                img2d,
                min_size=MIN_SIZE,
                sigma=SIGMA
            )

            fg = mascara_foreground(img2d)
            sp = aplicar_mascara_original(segmentos, fg)

            assert np.all(sp[fg == 0] == 0)

            fg_volume[fatia] = fg
            sp_volume[fatia] = sp

            regioes_total += int(np.unique(sp[sp > 0]).size)

            if fg.any():
                fatias_com_foreground += 1

        assert sp_volume.shape == volume.shape
        assert fg_volume.shape == volume.shape

        estado_fg = salvar_nifti(
            fg_volume, imagem_obj, fg_path
        )
        estado_sp = salvar_nifti(
            sp_volume, imagem_obj, sp_path
        )

        total_fatias += z
        pacientes_concluidos += 1

        registro = {
            "paciente": pid,
            "fatias": z,
            "fatias_com_foreground": fatias_com_foreground,
            "regioes_total": regioes_total,
            "foreground_pixels": int(fg_volume.sum()),
            "arquivo_fgmask": fg_path.name,
            "arquivo_superpixels": sp_path.name,
            "estado_fg": estado_fg,
            "estado_superpixels": estado_sp
        }

        registros.append(registro)

        salvar_json(STATUS, {
            "estado": "executando",
            "pacientes_concluidos": pacientes_concluidos,
            "pacientes_total": 20,
            "fatias_processadas": total_fatias,
            "fatias_previstas": 713,
            "ultimo_paciente": pid,
            "ultimo_registro": registro
        })

        print(
            f"[{pacientes_concluidos:02d}/20] "
            f"Paciente {pid:02d} | "
            f"fatias={z} | "
            f"foreground={fatias_com_foreground} | "
            f"regioes={regioes_total}",
            flush=True
        )

    assert total_fatias == 713

    salvar_json(MANIFESTO, {
        "dataset": "CHAOS T2-SPIR",
        "metodo": "Felzenszwalb",
        "mode": MODE,
        "min_size": MIN_SIZE,
        "sigma": SIGMA,
        "scale": 1,
        "fg_thresh": LIMIAR,
        "mascara": "Maior componente conexo + preenchimento",
        "remapeamento": "superpix_masking original",
        "pacientes": registros
    })

    salvar_json(STATUS, {
        "estado": "concluido",
        "pacientes_concluidos": 20,
        "pacientes_total": 20,
        "fatias_processadas": total_fatias,
        "fatias_previstas": 713,
        "manifesto": str(MANIFESTO)
    })

    print("SUPERPIXELS MRI CONCLUÍDOS", flush=True)


if __name__ == "__main__":
    try:
        executar()
    except Exception:
        erro = traceback.format_exc()

        salvar_json(STATUS, {
            "estado": "erro",
            "mensagem": erro
        })

        print(erro, flush=True)
        raise
