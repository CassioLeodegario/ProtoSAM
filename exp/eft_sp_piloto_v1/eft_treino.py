"""Piloto de 200 atualizações, com retomada a partir da última gravação atômica."""
from pathlib import Path
import importlib.metadata
import inspect
import json
import os
import platform
import random
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import eft_dados
import eft_modelo
from eft_dados import agora, assinatura, carregar_episodio, gravar_json, sha256
from eft_modelo import construir_modelo, VARIANTES, FewShotSeg, MultiProtoAsConv, Backbone_VSSM

def estado_rng():
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}

def restaurar_rng(r):
    random.setstate(r['python'])
    n = r['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(r['torch'])
    torch.cuda.set_rng_state_all(r['cuda'])

def salvar_checkpoint(pasta, modelo, otimizador, passo, config, baseline):
    destino = Path(pasta) / 'checkpoint_latest.pt'
    tmp = destino.with_suffix('.tmp')
    torch.save({'formato': 1, 'modelo': modelo.state_dict(),
                'otimizador': otimizador.state_dict(), 'scheduler': None,
                'passo': passo, 'rng': estado_rng(), 'config': config,
                'baseline': baseline}, tmp)
    os.replace(tmp, destino)

def avaliar(modelo, dados, plano, arquivo):
    """Eval desativa alinhamento no forward oficial: medir CE e Dice de pseudo-FG."""
    modo = modelo.training
    modelo.eval()
    linhas = []
    try:
        with torch.no_grad():
            for i, registro in enumerate(plano):
                s, fg, q, alvo = carregar_episodio(dados, registro, 'cuda')
                retorno = modelo([[s]], [[fg]], [[1-fg]], [q],
                                  isval=False, val_wsize=2, show_viz=False)
                # isval=False preserva kernel ALP 4x4; eval() controla drop path/alinhamento.
                logits = retorno[0]
                assert logits.shape == (1, 2, 256, 256)
                assert torch.isfinite(logits).all().item(), 'Logits inválidos na validação.'
                ce = F.cross_entropy(logits, alvo)
                assert torch.isfinite(ce).item()
                pred = logits.argmax(dim=1) == 1
                real = alvo == 1
                inter = (pred & real).sum().item()
                denom = pred.sum().item() + real.sum().item()
                assert denom > 0
                linhas.append({'indice': registro['indice'], 'imagem': registro['imagem'],
                               'regiao': registro['regiao'], 'ce': ce.item(),
                               'dice_pseudo_fg': 2.0 * inter / denom,
                               'fracao_fg_predita': pred.float().mean().item(),
                               'fracao_fg_alvo': real.float().mean().item()})
                if (i+1) % 25 == 0 or i+1 == len(plano):
                    print(f'Validação fixa: {i+1}/{len(plano)}', flush=True)
                del retorno, logits, ce
        resumo = {'n': len(linhas),
                  'ce_media': float(np.mean([r['ce'] for r in linhas])),
                  'dice_pseudo_fg_medio': float(np.mean([r['dice_pseudo_fg'] for r in linhas])),
                  'modo': 'eval; isval=False; kernel ALP 4x4; alinhamento desativado',
                  'por_imagem': linhas}
        gravar_json(arquivo, resumo)
        return resumo
    finally:
        modelo.train(modo)

def versoes():
    nomes = ['numpy', 'Pillow', 'scipy', 'scikit-image', 'timm', 'torchvision']
    resultado = {}
    for nome in nomes:
        try:
            resultado[nome] = importlib.metadata.version(nome)
        except importlib.metadata.PackageNotFoundError:
            resultado[nome] = 'não identificado'
    resultado.update(python=platform.python_version(), torch=str(torch.__version__),
                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                     gpu=torch.cuda.get_device_name(),
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32,
                     cudnn_benchmark=torch.backends.cudnn.benchmark,
                     deterministico=torch.are_deterministic_algorithms_enabled())
    return resultado

def executar_piloto(dados, variante='tiny', retomar=None, repo=Path('/workspace/ProtoSAM-repro')):
    """retomar é a pasta da execução, não o caminho de um checkpoint avulso."""
    assert variante in VARIANTES
    dados, repo = Path(dados).resolve(), Path(repo).resolve()
    planos = json.loads((dados / 'planos.json').read_text())
    assert len(planos['treino']) == 200 and planos['validacao']
    assert planos['config']['codigo_dados_sha256'] == sha256(eft_dados.__file__), 'Gerador mudou após preparação.'
    ti = {r['indice'] for r in planos['treino']}
    vi = {r['indice'] for r in planos['validacao']}
    assert len(ti) == 200 and not ti & vi, 'Divisão inconsistente.'
    # Verificar integridade dos caches realmente consumidos nesta execução.
    for r in planos['treino'] + planos['validacao']:
        assert sha256(dados / r['cache']) == r['cache_sha256'], f'Cache alterado: {r["cache"]}'
    fontes = list(dict.fromkeys(Path(p).resolve() for p in [
        __file__, eft_dados.__file__, eft_modelo.__file__,
        inspect.getfile(FewShotSeg), inspect.getfile(MultiProtoAsConv),
        inspect.getfile(Backbone_VSSM),
    ]))
    checkpoint_origem = repo / 'pretrained_model' / f'vmamba_{variante}_v2.pth'
    config = {'versao': 1, 'variante': variante, 'backbone': VARIANTES[variante],
              'dados': str(dados), 'planos_sha256': sha256(dados / 'planos.json'),
              'fontes_sha256': {str(p): sha256(p) for p in fontes},
              'checkpoint_origem': str(checkpoint_origem),
              'checkpoint_origem_sha256': sha256(checkpoint_origem),
              'semente': 1234, 'atualizacoes': 200, 'batch_size': 1,
              'otimizador': 'AdamW', 'lr': 1e-5, 'betas': [0.9, 0.999],
              'eps': 1e-8, 'weight_decay': 0.0, 'scheduler': None,
              'acumulacao': 1, 'amp': False, 'gradient_clipping': None,
              'adaptacao': 'full fine-tuning', 'loss': 'CE + alinhamento, ambos peso 1',
              'estagio': 3, 'normalizacao': 'classifier.norm -> outnorm3',
              'resolucao': [256,256], 'grade_alp': [32,32], 'kernel_alp': [4,4],
              'validacao': 'antes e depois; eval; CE e Dice pseudo-FG; isval=False',
              'checkpoint_a_cada': 25, 'selecao_melhor_checkpoint': False,
              'sam_no_treino': False, 'mascaras_manuais_utilizadas': False,
              'ambiente': versoes()}
    pasta = Path(retomar).resolve() if retomar else repo / 'reproducao' / f'eft_{variante}_piloto_{agora()}'
    if retomar:
        assert (pasta / 'config.json').is_file(), 'Pasta de retomada sem config.json.'
        assert json.loads((pasta / 'config.json').read_text()) == config, 'Configuração/código/ambiente mudou; não retomar silenciosamente.'
    else:
        pasta.mkdir(parents=True, exist_ok=False)
        gravar_json(pasta / 'config.json', config)
        (pasta / 'fontes').mkdir()
        for i, p in enumerate(fontes):
            shutil.copy2(p, pasta / 'fontes' / f'{i:02d}_{p.name}')
        shutil.copy2(dados / 'planos.json', pasta / 'planos.json')
        for nome, comando in [('gpu.txt', ['nvidia-smi']),
                              ('git_head.txt', ['git','-C',str(repo),'rev-parse','HEAD']),
                              ('git_status.txt', ['git','-C',str(repo),'status','--short'])]:
            r = subprocess.run(comando, capture_output=True, text=True, check=False)
            (pasta / nome).write_text(r.stdout + r.stderr, encoding='utf-8')
    print('EXECUÇÃO:', pasta, flush=True)
    print('Guarde esse caminho: permite retomar a partir do último checkpoint.', flush=True)
    if (pasta / 'resumo.json').exists():
        print('Piloto já concluído; nenhum novo passo executado.')
        return pasta
    tentativa = agora()
    log = pasta / f'passos_{tentativa}.jsonl'
    modelo = None
    otimizador = None
    passo = 0
    try:
        modelo = construir_modelo(variante, repo)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-5,
                                       betas=(0.9,0.999), eps=1e-8, weight_decay=0.0)
        ckpt = pasta / 'checkpoint_latest.pt'
        if ckpt.exists():
            salvo = torch.load(ckpt, map_location='cpu', weights_only=True)
            assert salvo['config'] == config and salvo['formato'] == 1
            modelo.load_state_dict(salvo['modelo'], strict=True)
            otimizador.load_state_dict(salvo['otimizador'])
            passo = int(salvo['passo'])
            assert 0 <= passo <= 200
            baseline = salvo['baseline']
            restaurar_rng(salvo['rng'])
            del salvo
            print(f'Retomando após atualização {passo}.', flush=True)
        else:
            print('Validação inicial das pseudo-máscaras.', flush=True)
            baseline = avaliar(modelo, dados, planos['validacao'], pasta / 'validacao_inicial.json')
            salvar_checkpoint(pasta, modelo, otimizador, 0, config, baseline)
        gravar_json(pasta / f'tentativa_{tentativa}.json',
                    {'passo_inicial': passo, 'log': log.name, 'config_sha256': assinatura(config)})
        modelo.train()
        for numero in range(passo + 1, 201):
            r = planos['treino'][numero - 1]
            inicio = time.perf_counter()
            s, fg, q, alvo = carregar_episodio(dados, r, 'cuda')
            otimizador.zero_grad(set_to_none=True)
            retorno = modelo([[s]], [[fg]], [[1-fg]], [q],
                              isval=False, val_wsize=2, show_viz=False)
            logits, alinhamento = retorno[:2]
            assert logits.shape == (1,2,256,256)
            assert torch.isfinite(logits).all().item(), 'Logits não finitos.'
            ce = F.cross_entropy(logits, alvo)
            total = ce + alinhamento
            assert torch.isfinite(total).item(), 'Perda não finita.'
            total.backward()
            gradientes = [p.grad for p in modelo.encoder.parameters() if p.grad is not None]
            assert gradientes and all(torch.isfinite(g).all().item() for g in gradientes), 'Gradientes ausentes/não finitos.'
            assert any(torch.count_nonzero(g).item() > 0 for g in gradientes), 'Gradientes todos zero.'
            otimizador.step()
            assert all(torch.isfinite(p).all().item() for p in modelo.parameters()), 'Parâmetros não finitos.'
            linha = {'passo': numero, 'indice': r['indice'], 'imagem': r['imagem'],
                     'regiao': r['regiao'], 'ce': ce.detach().item(),
                     'alinhamento': float(alinhamento.detach()) if torch.is_tensor(alinhamento) else float(alinhamento),
                     'total': total.detach().item(),
                     'fracao_fg_predita': (logits.detach().argmax(1) == 1).float().mean().item(),
                     'fracao_fg_alvo': (alvo == 1).float().mean().item(),
                     'segundos_diagnostico': time.perf_counter() - inicio}
            with log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(linha, ensure_ascii=False) + '\n')
            passo = numero
            if numero % 10 == 0 or numero == 1:
                print(f'{numero:03d}/200 | CE={linha["ce"]:.6f} | alinhamento={linha["alinhamento"]:.6f} | total={linha["total"]:.6f}', flush=True)
            del retorno, logits, alinhamento, ce, total, gradientes
            if numero % 25 == 0:
                salvar_checkpoint(pasta, modelo, otimizador, numero, config, baseline)
        otimizador.zero_grad(set_to_none=True)
        print('Validação final das mesmas pseudo-máscaras.', flush=True)
        final = avaliar(modelo, dados, planos['validacao'], pasta / 'validacao_final.json')
        resumo = {'variante': variante, 'atualizacoes': passo,
                  'ce_validacao_antes': baseline['ce_media'],
                  'ce_validacao_depois': final['ce_media'],
                  'dice_pseudo_fg_antes': baseline['dice_pseudo_fg_medio'],
                  'dice_pseudo_fg_depois': final['dice_pseudo_fg_medio'],
                  'n_validacao': final['n'], 'checkpoint': 'checkpoint_latest.pt',
                  'limite': 'Piloto EFT-SP; métricas de superpixels, não de pólipos; sem teste clínico.'}
        gravar_json(pasta / 'resumo.json', resumo)
        print(json.dumps(resumo, ensure_ascii=False, indent=2))
        return pasta
    except BaseException:
        (pasta / f'erro_{tentativa}.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Execução interrompida. Envie o traceback. Pasta:', pasta, flush=True)
        print('Retomada usa checkpoint_latest.pt; passos posteriores a ele serão refeitos.', flush=True)
        raise
    finally:
        if modelo is not None:
            modelo.zero_grad(set_to_none=True)
        del modelo, otimizador
