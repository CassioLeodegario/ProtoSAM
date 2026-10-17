"""Importa para resultados/registro.csv as avaliações do ProtoSAM-repro (out/2026).

Os valores são lidos dos resumo.json/por_suporte.csv originais; a lista abaixo
só diz o que cada pasta significa. Inspeções de uma única query não entram.

Uso: python migracao/importar_registro_antigo.py <pasta ProtoSAM-repro>
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protovm import registro  # noqa: E402

REPRO = Path(sys.argv[1] if len(sys.argv) > 1 else '../ProtoSAM-repro').resolve() / 'reproducao'


def ler(p):
    return json.loads(p.read_text(encoding='utf-8'))


def valor(obj, caminho):
    for chave in caminho.split('.'):
        obj = obj[chave]
    return obj


def m(*partes, **extra):
    """Mescla dicionários de metadados (dict() só aceita um posicional)."""
    saida = {}
    for parte in partes:
        saida.update(parte)
    saida.update(extra)
    return saida


def gpu(pasta):
    """GPU em que a avaliação rodou. Usa a chave gpu_driver/gpu_driver_memoria;
    outras chaves (ex.: gpu_driver_referencia) citam GPUs de runs anteriores."""
    for nome in ('config.json', 'ambiente.json'):
        p = pasta / nome
        if not p.exists():
            continue
        cfg = ler(p)
        for chave in ('gpu_driver', 'gpu_driver_memoria', 'gpu'):
            if isinstance(cfg.get(chave), str):
                return cfg[chave].split(',')[0].strip()
        for v in cfg.values():
            if isinstance(v, dict) and isinstance(v.get('gpu'), str):
                return v['gpu']
    return ''


PRE_ST3 = dict(modelo='vmamba_tiny', pesos='pre_treinado', aval_estagio=3, aval_resolucao=672, grade=32)
EFT256 = dict(treino_estagio=3, treino_norm='checkpoint', treino_resolucao=256)
EFT256_PEN = dict(treino_estagio=2, treino_norm='ln_unitaria', treino_resolucao=256)
V = dict(conjunto='V', suporte='S00', n=145)
T = dict(conjunto='T', suporte='S00', n=100)

# (pasta, arquivo, chave ALP, chave SAM, metadados)
ENTRADAS = [
    # ---- T: DINOv2-L, baseline oficial ----
    ('01_avaliacao_K1_20261004T140725337013Z/comparacao_completa_alp_sam_20261006T083800723726Z', 'resumo.json',
     'medias_por_imagem.ALP.Dice', 'medias_por_imagem.SAM.Dice',
     m(T, modelo='dinov2_l14', pesos='pre_treinado', aval_resolucao=672, grade=48, obs='baseline oficial')),
    ('dinov2_l14_k1_selecao_sam_exploratoria_20261010T183138214333Z', 'resumo.json', None, 'Dice_maior_score',
     m(T, modelo='dinov2_l14', pesos='pre_treinado', aval_resolucao=672, grade=48,
          obs='exploratório: máscara SAM de maior score')),
    # ---- T: VMamba congelado ----
    *[(p, 'resumo.json', 'medias_por_imagem.ALP.Dice', 'medias_por_imagem.SAM.Dice',
       m(T, PRE_ST3, modelo=f'vmamba_{v}', aval_norm='ln_unitaria', obs='outnorm3 inicializada'))
      for v, p in [('tiny', 'vmamba_tiny_k1_avaliar_20261008T095813008819Z'),
                   ('small', 'vmamba_small_k1_avaliar_20261008T110134736074Z'),
                   ('base', 'vmamba_base_k1_avaliar_20261008T134621894550Z')]],
    *[(p, 'resumo.json', 'ALP.Dice_treinada', 'SAM.Dice_treinada',
       m(T, PRE_ST3, modelo=f'vmamba_{v}', aval_norm='checkpoint'))
      for v, p in [('tiny', 'vmamba_tiny_k1_norm_pareada_20261010T202022577876Z'),
                   ('small', 'vmamba_small_k1_norm_pareada_20261010T210704936556Z')]],
    ('vmamba_base_k1_norm_treinada_20261010T195514199066Z', 'resumo.json',
     'ALP_Dice_norm_treinada', 'SAM_Dice_norm_treinada', m(T, PRE_ST3, modelo='vmamba_base', aval_norm='checkpoint')),
    *[(p, 'resumo.json', 'medias.ALP.Dice', 'medias.SAM.Dice',
       m(T, PRE_ST3, modelo=f'vmamba_{v}', aval_estagio=2, aval_norm='ln_unitaria', grade=42))
      for v, p in [('tiny', 'vmamba_tiny_k1_estagio2_grade42_20261011T102701902569Z'),
                   ('small', 'vmamba_small_k1_estagio2_grade42_20261011T092524146795Z'),
                   ('base', 'vmamba_base_k1_estagio2_grade42_20261011T111220763092Z')]],
    *[('vmamba_base_k1_fusoes_20261011T123856652462Z', 'resumo.json', f'{f}.medias.ALP.Dice', f'{f}.medias.SAM.Dice',
       m(T, PRE_ST3, modelo='vmamba_base', aval_estagio=e, aval_norm='ln_unitaria+L2', grade=42, obs='fusão de estágios'))
      for f, e in [('estagios_1_2', '1+2'), ('estagios_2_3', '2+3')]],
    ('vmamba_base_k1_selecao_sam_exploratoria_20261010T180702018291Z', 'resumo.json', None, 'Dice_maior_score',
     m(T, PRE_ST3, modelo='vmamba_base', aval_norm='ln_unitaria', obs='exploratório: máscara SAM de maior score')),
    # ---- V: pontos de partida (pré-treinado) ----
    ('eft_tiny_validacao_polipos_20261011T172038284160Z', 'resumo.json', 'ALP.Dice.antes', 'SAM.Dice.antes',
     m(V, PRE_ST3, aval_norm='checkpoint')),
    ('eft_small_validacao_polipos_20261011T175805568497Z', 'resumo.json', 'ALP.Dice.antes', 'SAM.Dice.antes',
     m(V, PRE_ST3, modelo='vmamba_small', aval_norm='checkpoint')),
    ('eft_base_validacao_polipos_20261011T183647311706Z', 'resumo.json', 'ALP.Dice.antes', 'SAM.Dice.antes',
     m(V, PRE_ST3, modelo='vmamba_base', aval_norm='checkpoint')),
    *[(p, 'resumo.json', 'ALP.Dice.antes', 'SAM.Dice.antes', m(V, PRE_ST3, aval_estagio=e, aval_norm=n))
      for p, e, n in [('eft_tiny_camada3_sem_norm_20261012T172311538570Z', 3, 'sem_norm'),
                      ('eft_tiny_camada3_ln_unitaria_20261012T175643423487Z', 3, 'ln_unitaria'),
                      ('eft_tiny_camada2_sem_norm_20261012T184423197125Z', 2, 'sem_norm'),
                      ('eft_tiny_camada2_ln_unitaria_20261012T193231751412Z', 2, 'ln_unitaria')]],
    # ---- V: pilotos EFT de 200 passos ----
    *[(p, 'resumo.json', 'ALP.Dice.depois', 'SAM.Dice.depois',
       m(V, PRE_ST3, EFT256, modelo=f'vmamba_{v}', pesos=f'eft:{t}', passo=200, aval_norm='checkpoint'))
      for v, p, t in [('tiny', 'eft_tiny_validacao_polipos_20261011T172038284160Z', 'eft_tiny_piloto_20261011T152746615659Z'),
                      ('small', 'eft_small_validacao_polipos_20261011T175805568497Z', 'eft_small_piloto_20261011T154009078232Z'),
                      ('base', 'eft_base_validacao_polipos_20261011T183647311706Z', 'eft_base_piloto_20261011T155018632056Z')]],
    # ---- V: Tiny EFT no último estágio, por passo ----
    *[(p, 'resumo.json', 'ALP.Dice.depois', 'SAM.Dice.depois',
       m(V, PRE_ST3, EFT256, pesos=f'eft:{t}', passo=n, aval_norm='checkpoint', obs=o))
      for p, t, n, o in [
          ('eft_tiny_passo_1000_validacao_20261011T213201571961Z', 'eft_tiny_2000_v2_20261011T204725195685Z', 1000, ''),
          ('eft_tiny_passo_1500_validacao_20261011T215554386693Z', 'eft_tiny_2000_v2_20261011T204725195685Z', 1500, ''),
          ('eft_tiny_passo_2000_validacao_20261011T221527472147Z', 'eft_tiny_2000_v2_20261011T204725195685Z', 2000, 'melhor VMamba'),
          ('eft_tiny_referencia_driver_atual_20261012T215801165279Z', 'eft_tiny_2000_v2_20261011T204725195685Z', 2000,
           'repetição do passo 2000 com outro driver'),
          ('eft_tiny_passo_3000_validacao_20261011T233900809737Z', 'eft_tiny_5000_20261011T224215171455Z', 3000, ''),
          ('eft_tiny_passo_4000_validacao_20261011T235323843203Z', 'eft_tiny_5000_20261011T224215171455Z', 4000, ''),
          ('eft_tiny_passo_5000_validacao_20261012T000645443920Z', 'eft_tiny_5000_20261011T224215171455Z', 5000, '')]],
    # ---- V: pesos EFT 2000 (último estágio) lidos em outras camadas/normalizações ----
    *[(p, 'resumo.json', 'ALP.Dice.depois', 'SAM.Dice.depois',
       m(V, PRE_ST3, EFT256, pesos='eft:eft_tiny_2000_v2_20261011T204725195685Z', passo=2000, aval_estagio=e, aval_norm=n))
      for p, e, n in [('eft_tiny_camada3_sem_norm_20261012T172311538570Z', 3, 'sem_norm'),
                      ('eft_tiny_camada3_ln_unitaria_20261012T175643423487Z', 3, 'ln_unitaria'),
                      ('eft_tiny_camada2_sem_norm_20261012T184423197125Z', 2, 'sem_norm'),
                      ('eft_tiny_camada2_ln_unitaria_20261012T193231751412Z', 2, 'ln_unitaria')]],
    # ---- V: Tiny EFT treinado no penúltimo estágio ----
    *[(p, 'resumo.json', 'ALP.Dice.depois', 'SAM.Dice.depois',
       m(V, PRE_ST3, EFT256_PEN, pesos='eft:eft_tiny_penultima_ln_5000_20261012T210006906429Z', passo=n,
            aval_estagio=2, aval_norm='ln_unitaria'))
      for p, n in [('eft_penultima_passo_1000_validacao_20261012T235022253086Z', 1000),
                   ('eft_penultima_passo_1500_validacao_20261013T001309011868Z', 1500),
                   ('eft_penultima_passo_2000_validacao_20261013T003337707573Z', 2000),
                   ('eft_penultima_passo_3000_gpu4090_20261014T005254257635Z', 3000),
                   ('eft_penultima_passo_4000_gpu4090_20261014T011829759663Z', 4000),
                   ('eft_penultima_passo_5000_gpu4090_20261014T014449270029Z', 5000)]],
]


def importar():
    assert not registro.ARQUIVO.exists(), f'{registro.ARQUIVO} já existe; apague para reimportar.'
    total = 0
    for pasta, arquivo, chave_alp, chave_sam, meta in ENTRADAS:
        p = REPRO / pasta
        r = ler(p / arquivo)
        registro.adicionar(m(meta, estado='importado', origem=f'ProtoSAM-repro/reproducao/{pasta}',
                                gpu=gpu(p if (p / 'config.json').exists() else p.parent),
                                dice_alp=valor(r, chave_alp) if chave_alp else '',
                                dice_sam=valor(r, chave_sam)))
        total += 1

    # Exp. 15 e 16: 20 suportes em V.
    multi = REPRO / 'multissuporte_tiny_eft2000_kvasir_v1'
    with (multi / 'por_suporte.csv').open(encoding='utf-8') as f:
        for linha in csv.DictReader(f):
            s = linha['suporte']
            base = m(V, PRE_ST3, suporte=s, aval_norm='checkpoint', gpu=gpu(multi / s),
                        estado='importado', origem=f'ProtoSAM-repro/reproducao/{multi.name}/{s}', obs='exp. 15')
            registro.adicionar(m(base, dice_alp=linha['ALP_inicial'], dice_sam=linha['SAM_inicial']))
            registro.adicionar(m(base, EFT256, pesos='eft:eft_tiny_2000_v2_20261011T204725195685Z', passo=2000,
                                    dice_alp=linha['ALP_EFT'], dice_sam=linha['SAM_EFT']))
            total += 2
    dino = REPRO / 'multissuporte_dinov2_l14_kvasir_v1'
    with (dino / 'por_suporte.csv').open(encoding='utf-8') as f:
        for linha in csv.DictReader(f):
            registro.adicionar(m(V, suporte=linha['suporte'], modelo='dinov2_l14', pesos='pre_treinado',
                                    aval_resolucao=672, grade=48, estado='importado',
                                    origem=f'ProtoSAM-repro/reproducao/{dino.name}/{linha["suporte"]}',
                                    obs='exp. 16 (incompleto)', dice_alp=linha['ALP_DINO'], dice_sam=linha['SAM_DINO']))
            total += 1
    print(f'{total} linhas importadas em {registro.ARQUIVO}')


if __name__ == '__main__':
    importar()
