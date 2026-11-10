"""Validação anotada, pareada, do piloto EFT; somente as 145 imagens reservadas."""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import csv
import gc
import hashlib
import inspect
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import traceback
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from eft_modelo_validacao import construir_modelo, VARIANTES
from models.grid_proto_fewshot import FewShotSeg
from models.alpmodule import MultiProtoAsConv
from models.ProtoSAM import ProtoSAM, ALPNetWrapper, ALPNetInput
from models import vmamba
from dataloaders.PolypDataset import get_polyp_dataset
from models.segment_anything.utils.transforms import ResizeLongestSide

REPO = Path('/workspace/ProtoSAM-repro')
BASE = '64868c909668aac9ad517f7192a2712e88268e8f'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for bloco in iter(lambda: f.read(1024*1024), b''):
            h.update(bloco)
    return h.hexdigest()

def ler(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def salvar(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def medir(pred, gt):
    assert pred.shape == gt.shape == (1024,1024)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
            'IoU': tp/(tp+fp+fn) if tp+fp+fn else 1.0,
            'precisao': tp/(tp+fp) if tp+fp else 0.0,
            'recall': tp/(tp+fn) if tp+fn else 0.0}

def construir_672(variante):
    origem = construir_modelo(variante, REPO)
    encoder = origem.encoder
    canais = VARIANTES[variante]['dims'] * 8
    class FewShotSegValidacao(FewShotSeg):
        def get_encoder(self):
            self.encoder = encoder
            self.config['feature_hw'] = [32,32]
        def get_cls(self):
            self.cls_unit = MultiProtoAsConv(
                proto_grid=[8,8], feature_hw=[32,32], embed_dim=canais)
        def get_features(self, imgs_concat):
            imgs_concat = F.interpolate(imgs_concat, size=(672,672),
                                        mode='bilinear', align_corners=False)
            fts = self.encoder(imgs_concat)[-1]
            assert tuple(fts.shape[1:]) == (canais,21,21)
            return F.interpolate(fts, size=(32,32), mode='bilinear', align_corners=False)
    cfg = {'which_model': 'vmamba_' + variante, 'cls_name':'grid_proto',
           'proto_grid_size':8, 'lora':0, 'align':False, 'debug':False,
           'use_coco_init':False, 'use_pos_enc':False, 'use_slice_adapter':False,
           'adapter_layers':3, 'dinov2_loss':False}
    modelo = FewShotSegValidacao(image_size=672, pretrained_path=None, cfg=deepcopy(cfg)).cuda()
    # Transfere todo o estado inicial do construtor usado no piloto.
    modelo.load_state_dict(origem.state_dict(), strict=True)
    del origem
    assert type(modelo).forward is FewShotSeg.forward
    assert type(modelo).alignLoss is FewShotSeg.alignLoss
    return modelo.eval().requires_grad_(False)

def conferir_amostra(amostra):
    assert tuple(amostra['image'].shape) == (3,1024,1024)
    assert tuple(amostra['label'].shape) == (1024,1024)
    assert torch.isfinite(amostra['image']).all().item()
    assert ((amostra['label']==0) | (amostra['label']==1)).all().item()

def executar(piloto, passo, baseline, retomar=None):
    piloto = Path(piloto).resolve()
    resumo_piloto = ler(piloto / 'resumo.json')
    cfg_piloto = ler(piloto / 'config.json')
    variante = cfg_piloto['variante']
    assert variante == 'tiny' and resumo_piloto['atualizacoes'] == 2000
    assert passo in (1000, 1500, 2000)
    dados = Path(cfg_piloto['dados'])
    assert sha(dados / 'planos.json') == cfg_piloto['planos_sha256']
    planos = ler(dados / 'planos.json')
    manifesto_path = Path(planos['config']['manifesto'])
    assert sha(manifesto_path) == planos['config']['manifesto_sha256']
    manifesto = ler(manifesto_path)
    reservadas = planos['validacao']
    assert len(reservadas) == 145
    assert not {r['indice'] for r in reservadas} & {r['indice'] for r in planos['treino']}
    assert len(set(r['imagem'] for r in reservadas)) == 145
    suporte_meta = manifesto['suporte_k1']
    assert Path(suporte_meta['imagem']).name == 'cju3xtufwiv9c0818djsc4cqd.png'
    assert suporte_meta['imagem'] not in {r['imagem'] for r in reservadas}
    teste = {str(Path(r['imagem']).resolve()) for r in manifesto['teste']}
    assert not teste & {str(Path(r['imagem']).resolve()) for r in reservadas}

    # Verifica fonte oficial local; não modifica nenhum arquivo do repositório.
    arquivos_oficiais = ['models/grid_proto_fewshot.py','models/alpmodule.py',
        'models/ProtoSAM.py','dataloaders/PolypDataset.py','dataloaders/PolypTransforms.py',
        'models/segment_anything/utils/transforms.py']
    for relativo in arquivos_oficiais:
        original = subprocess.check_output(['git','-C',str(REPO),'show',f'{BASE}:{relativo}'])
        assert (REPO / relativo).read_bytes() == original, f'Código oficial alterado: {relativo}'
    assert Path(inspect.getfile(FewShotSeg)).resolve() == REPO / 'models/grid_proto_fewshot.py'
    assert Path(inspect.getfile(ProtoSAM)).resolve() == REPO / 'models/ProtoSAM.py'
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)

    # Reutiliza o mesmo dataset/preprocessamento do notebook congelado.
    # As queries de validação estão no pool original de treino, chamado ds_suportes.
    # O dataset de teste retornado pela factory não é indexado nem avaliado.
    ds_pool, ds_teste_nao_usado = get_polyp_dataset(
        image_size=(1024,1024), sam_trans=ResizeLongestSide(1024))
    del ds_teste_nao_usado
    mapa = {str(Path(p).resolve()): i for i,p in enumerate(ds_pool.images)}
    assert len(mapa) == len(ds_pool.images), 'Caminhos duplicados no dataset oficial.'
    pool_manifesto = {str(Path(r['imagem']).resolve()):r for r in manifesto['pool_adaptacao']}
    itens = []
    for r in reservadas:
        chave = str(Path(r['imagem']).resolve())
        assert chave in mapa and chave in pool_manifesto, chave
        original = pool_manifesto[chave]
        i = mapa[chave]
        assert Path(ds_pool.gts[i]).resolve() == Path(original['mascara']).resolve()
        assert sha(original['imagem']) == r['imagem_sha256'], 'Imagem mudou após preparação.'
        itens.append({'id': f'{r["indice"]:04d}', 'indice_dataset':i,
                      'imagem':original['imagem'], 'mascara':original['mascara'],
                      'imagem_sha256':sha(original['imagem']),
                      'mascara_sha256':sha(original['mascara'])})
    indice_sup = mapa[str(Path(suporte_meta['imagem']).resolve())]
    assert Path(ds_pool.gts[indice_sup]).resolve() == Path(suporte_meta['mascara']).resolve()
    suporte = ds_pool[indice_sup]
    conferir_amostra(suporte)
    assert int((suporte['label'] > .5).sum()) == 51940, 'Suporte difere da referência.'

    ckpt_sam = REPO / 'pretrained_model/sam_vit_h.pth'
    ckpt_eft = piloto / f'checkpoint_{passo:04d}.pt'
    assert ckpt_eft.is_file(), ckpt_eft
    ckpt_original = Path(cfg_piloto['checkpoint_origem'])
    assert sha(ckpt_original) == cfg_piloto['checkpoint_origem_sha256']
    cfg_sam = {'image_size':[1024,1024], 'sam_pretrained_path':str(ckpt_sam),
        'num_points_for_sam':1, 'use_points':True, 'use_bbox':True, 'use_mask':False,
        'debug':False, 'use_cca':False, 'point_mode':'both', 'use_sam_trans':True,
        'coarse_pred_only':False, 'alpnet_image_size':672, 'use_neg_points':False}
    fontes = [Path(__file__),Path(inspect.getfile(construir_modelo))] + [REPO/r for r in arquivos_oficiais] + [REPO/'models/vmamba.py']
    config = {'finalidade':'Validação anotada de checkpoints EFT; não é teste final',
        'passo':passo, 'checkpoint_eft':str(ckpt_eft),
        'variante':variante, 'piloto':str(piloto), 'episodios':itens,
        'suporte':dict(suporte_meta, imagem_sha256=sha(suporte_meta['imagem']), mascara_sha256=sha(suporte_meta['mascara'])),
        'resolucao_encoder':672, 'canvas':1024, 'estagio':3,
        'norm_antes':'classifier.norm pré-treinada transferida para outnorm3',
        'norm_depois':'outnorm3 do checkpoint EFT',
        'sam':cfg_sam, 'selecao_sam':'alternativa 0 por componente; união oficial',
        'alp_inferencia':{'isval':True,'val_wsize':2,'show_viz':False,'supp_fts':None},
        'grade_features':[32,32], 'kernel_configurado':[4,4],
        'checkpoint_eft_sha256':sha(ckpt_eft), 'checkpoint_original_sha256':sha(ckpt_original),
        'checkpoint_sam_sha256':sha(ckpt_sam),
        'fontes_sha256':{str(p):sha(p) for p in fontes},
        'gpu_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        'torch':str(torch.__version__), 'cuda':torch.version.cuda,
        'versoes':{p:importlib.metadata.version(p) for p in ['numpy','Pillow','torchvision','timm','scipy']},
        'suporte_tensor_sha256':hashlib.sha256(suporte['image'].cpu().numpy().tobytes()).hexdigest(),
        'cudnn':torch.backends.cudnn.version(), 'tf32_matmul':False, 'tf32_cudnn':True,
        'metricas':'Dice, IoU, precisão e recall; média por imagem no canvas1024',
        'sem_atualizacao_pesos':True, 'mascaras_query':'usadas somente no cálculo das métricas'}

    baseline = Path(baseline).resolve()
    ref = ler(baseline / 'config.json')

    for chave in (
        'variante', 'episodios', 'suporte', 'resolucao_encoder', 'canvas',
        'estagio', 'norm_antes', 'sam', 'selecao_sam', 'alp_inferencia',
        'grade_features', 'kernel_configurado', 'checkpoint_original_sha256',
        'checkpoint_sam_sha256', 'torch', 'cuda', 'versoes',
        'suporte_tensor_sha256', 'cudnn', 'tf32_matmul', 'tf32_cudnn',
        'gpu_driver',
    ):
        assert ref[chave] == config[chave], f'Baseline incompatível: {chave}'

    for fonte, digest in config['fontes_sha256'].items():
        if fonte in ref['fontes_sha256']:
            assert ref['fontes_sha256'][fonte] == digest, fonte

    assert ler(baseline / 'resumo.json')['n'] == 145
    config['baseline_reutilizada'] = str(baseline)
    config['baseline_config_sha256'] = sha(baseline / 'config.json')

    pasta = (
        Path(retomar).resolve()
        if retomar
        else REPO/'reproducao'/f'eft_tiny_passo_{passo}_validacao_{stamp()}'
    )

    if retomar:
        assert ler(pasta/'config.json') == config, 'Configuração/dados/pesos/ambiente mudaram.'
    else:
        pasta.mkdir(parents=True,exist_ok=False)
        salvar(pasta/'config.json',config)
        (pasta/'fontes').mkdir()
        for i,p in enumerate(fontes): shutil.copy2(p,pasta/'fontes'/f'{i:02d}_{p.name}')
    print('AVALIAÇÃO:',pasta,flush=True)
    if (pasta/'resumo.json').exists():
        print(json.dumps(ler(pasta/'resumo.json'),indent=2)); return pasta


    if not (pasta / 'antes').exists():
        temporario = pasta / f'antes_tmp_{stamp()}'
        shutil.copytree(baseline / 'antes', temporario)
        temporario.rename(pasta / 'antes')

    print(
        'Baseline reutilizada; suas máscaras serão conferidas por hash.',
        flush=True,
    )
    modelo = construir_672(variante)

    wrapper = ALPNetWrapper(modelo)
    protosam = ProtoSAM(coarse_segmentation_model=wrapper, **cfg_sam).to('cuda',dtype=torch.float32)
    protosam.eval().requires_grad_(False)
    modelo.eval().requires_grad_(False)
    sup_img = suporte['image'].unsqueeze(0).cuda().float()
    sup_mask = suporte['label'].unsqueeze(0).cuda().float()
    ext = vmamba.selective_scan_cuda
    fwd_original, fallback_original = ext.fwd, vmamba.selective_scan_torch
    chamadas = {'cuda':0}
    def contar_cuda(*args,**kwargs):
        chamadas['cuda'] += 1
        return fwd_original(*args,**kwargs)
    def recusar_fallback(*args,**kwargs):
        raise RuntimeError('Fallback selective_scan_torch não permitido.')
    registros = {}
    try:
        assert vmamba.WITH_SELECTIVESCAN_MAMBA
        ext.fwd, vmamba.selective_scan_torch = contar_cuda, recusar_fallback
        for condicao in ['antes','depois']:
            if condicao == 'depois':
                salvo = torch.load(ckpt_eft,map_location='cpu',weights_only=True)
                assert salvo['passo'] == passo and salvo['config'] == cfg_piloto
                modelo.load_state_dict(salvo['modelo'],strict=True)
                del salvo
            modelo.eval().requires_grad_(False)
            for sub in ['registros','alp','sam','gt']:
                (pasta/condicao/sub).mkdir(parents=True,exist_ok=True)
            linhas = []
            with torch.no_grad(), torch.autocast('cuda',enabled=False):
                for pos,item in enumerate(itens):
                    destino = pasta/condicao/'registros'/f'{item["id"]}.json'
                    if destino.exists():
                        linha = ler(destino)
                        assert linha['item'] == item
                        for relativo,digest in linha['arquivos_sha256'].items():
                            assert sha(pasta/relativo) == digest, f'Máscara alterada: {relativo}'
                        linhas.append(linha)
                        continue
                    amostra = ds_pool[item['indice_dataset']]
                    conferir_amostra(amostra)
                    gt = amostra['label'].cpu().numpy() > .5
                    input_sha = hashlib.sha256(amostra['image'].cpu().numpy().tobytes()).hexdigest()
                    if condicao=='depois':
                        antes_reg = ler(pasta/'antes'/'registros'/f'{item["id"]}.json')
                        assert input_sha==antes_reg['query_tensor_sha256'], 'Pré-processamento da query mudou entre condições.'
                    qry = amostra['image'].unsqueeze(0).cuda().float()
                    def entrada():
                        return ALPNetInput(support_images=[sup_img],support_labels=[sup_mask],
                            query_images=qry,isval=True,val_wsize=2,show_viz=False,supp_fts=None)
                    chamadas['cuda'] = 0
                    logits = wrapper(entrada())
                    assert tuple(logits.shape)==(1,2,1024,1024) and torch.isfinite(logits).all().item()
                    alp = (logits.argmax(1)[0]==1).cpu().numpy()
                    del logits
                    saida,scores = protosam(query_image=qry,coarse_model_input=entrada(),degrees_rotate=0)
                    sam = saida.detach().cpu().numpy().squeeze()
                    assert sam.shape==(1024,1024) and np.isin(sam,[0,1]).all()
                    assert chamadas['cuda']>0
                    sam = sam.astype(bool)
                    if condicao=='depois':
                        with Image.open(pasta/'antes'/'gt'/f'{item["id"]}.png') as im:
                            assert np.array_equal(gt,np.asarray(im)>0), 'GT mudou entre condições.'
                    hashes = {}
                    for tipo,mascara in [('alp',alp),('sam',sam),('gt',gt)]:
                        arquivo=pasta/condicao/tipo/f'{item["id"]}.png'
                        Image.fromarray(mascara.astype(np.uint8)*255).save(arquivo)
                        hashes[str(arquivo.relative_to(pasta))]=sha(arquivo)
                    linha={'item':item,'metricas':{'ALP':medir(alp,gt),'SAM':medir(sam,gt)},
                           'query_tensor_sha256':input_sha,
                           'scores_sam':len(scores),'chamadas_cuda':chamadas['cuda'],
                           'arquivos_sha256':hashes}
                    salvar(destino,linha)
                    linhas.append(linha)
                    print(f'{condicao} {pos+1}/145 | {Path(item["imagem"]).name} | ALP={linha["metricas"]["ALP"]["Dice"]:.4f} | SAM={linha["metricas"]["SAM"]["Dice"]:.4f}',flush=True)
                    del saida,scores,qry
            registros[condicao]=linhas
        resumo={'variante':variante,'passo':passo,'n':145,'tipo':'validação anotada; métricas em canvas1024','ALP':{},'SAM':{}}
        tabela=[]
        for a,d in zip(registros['antes'],registros['depois']):
            assert a['item']==d['item']
            linha={'id':a['item']['id'],'imagem':a['item']['imagem']}
            for etapa in ['ALP','SAM']:
                for m in ['Dice','IoU','precisao','recall']:
                    linha[f'antes_{etapa}_{m}']=a['metricas'][etapa][m]
                    linha[f'depois_{etapa}_{m}']=d['metricas'][etapa][m]
                linha[f'delta_{etapa}_Dice']=d['metricas'][etapa]['Dice']-a['metricas'][etapa]['Dice']
            tabela.append(linha)
        for etapa in ['ALP','SAM']:
            for m in ['Dice','IoU','precisao','recall']:
                resumo[etapa][m]={cond:float(np.mean([r[f'{cond}_{etapa}_{m}'] for r in tabela])) for cond in ['antes','depois']}
            delta=np.array([r[f'delta_{etapa}_Dice'] for r in tabela])
            resumo[etapa].update(delta_Dice_medio=float(delta.mean()),delta_Dice_mediano=float(np.median(delta)),
                melhoraram=int((delta>1e-8).sum()),pioraram=int((delta < -1e-8).sum()),empataram=int((np.abs(delta)<=1e-8).sum()))
        with (pasta/'por_imagem.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(tabela[0]));writer.writeheader();writer.writerows(tabela)
        salvar(pasta/'resumo.json',resumo)
        print(json.dumps(resumo,indent=2,ensure_ascii=False))
        return pasta
    except BaseException:
        (pasta/f'erro_{stamp()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('Interrompido. Retome com RETOMAR apontando para:',pasta,flush=True)
        raise
    finally:
        ext.fwd,vmamba.selective_scan_torch=fwd_original,fallback_original
        del protosam,wrapper,modelo
        gc.collect()
        torch.cuda.empty_cache()
