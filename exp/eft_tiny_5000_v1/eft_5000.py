
from pathlib import Path
import json, shutil, gc, traceback
import torch
import torch.nn.functional as F
from eft_treino import (
    construir_modelo, versoes, avaliar, estado_rng, restaurar_rng,
    salvar_checkpoint, carregar_episodio, gravar_json, assinatura, sha256, agora,
)
from eft_plano_2000 import estender
from eft_alinhamento_seguro import instalar

REPO = Path('/workspace/ProtoSAM-repro')
ORIGEM = REPO / 'reproducao/eft_tiny_2000_v2_20261011T204725195685Z'

def executar(retomar=None):
    anterior = json.loads((ORIGEM / 'config.json').read_text())
    ckpt_origem = ORIGEM / 'checkpoint_2000.pt'
    dados = Path(anterior['dados'])
    ambiente = anterior['ambiente']

    torch.backends.cuda.matmul.allow_tf32 = ambiente['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = ambiente['tf32_cudnn']
    torch.backends.cudnn.benchmark = ambiente['cudnn_benchmark']
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(ambiente['deterministico'])

    assert versoes() == ambiente, 'Ambiente diferente do treinamento anterior.'
    assert sha256(dados / 'planos.json') == anterior['planos_sha256']

    for arquivo, digest in anterior['fontes_continuacao_sha256'].items():
        assert sha256(arquivo) == digest, f'Fonte anterior alterada: {arquivo}'
    for arquivo, digest in anterior['fontes_sha256'].items():
        assert sha256(arquivo) == digest, f'Fonte original alterada: {arquivo}'

    assert sha256(anterior['checkpoint_origem']) == anterior['checkpoint_origem_sha256']

    antigos = json.loads((ORIGEM / 'planos.json').read_text())
    registros = []
    for item in json.loads((dados / 'divisao.json').read_text()):
        if item['split'] == 'treino':
            r = json.loads(
                (dados / 'cache' / f"{item['indice']:04d}.json").read_text()
            )
            assert all(r[k] == v for k, v in item.items())
            registros.append(r)

    planos = {
        'treino': estender(
            registros, antigos['treino'][:200],
            antigos['validacao'], total=5000
        ),
        'validacao': antigos['validacao'],
    }
    assert len(antigos['treino']) == 2000
    assert planos['treino'][:2000] == antigos['treino'], 'Sequência anterior divergiu.'

    for r in registros + planos['validacao']:
        assert sha256(dados / r['cache']) == r['cache_sha256']

    fontes = sorted(Path(__file__).parent.glob('*.py'))
    config = dict(anterior)
    config.update(
        versao=4, atualizacoes=5000, passo_origem=2000,
        continuacao_origem=str(ORIGEM),
        checkpoint_continuacao=str(ckpt_origem),
        checkpoint_continuacao_sha256=sha256(ckpt_origem),
        plano_estendido_sha256=assinatura(planos),
        fontes_continuacao_sha256={str(f):sha256(f) for f in fontes},
        checkpoints_preservados=list(range(2500,5001,500)),
        checkpoint_a_cada=50,
        validacao='pseudo-mascaras fixas a cada 250 passos',
    )

    pasta = (
        Path(retomar).resolve() if retomar
        else REPO / 'reproducao' / f'eft_tiny_5000_{agora()}'
    )
    assert pasta != ORIGEM

    if retomar:
        assert json.loads((pasta / 'config.json').read_text()) == config, 'Configuração mudou.'
        assert json.loads((pasta / 'planos.json').read_text()) == planos
    else:
        pasta.mkdir(parents=True, exist_ok=False)
        gravar_json(pasta / 'config.json', config)
        gravar_json(pasta / 'planos.json', planos)
        (pasta / 'fontes').mkdir()
        for f in fontes:
            shutil.copy2(f, pasta / 'fontes' / f.name)
        shutil.copy2(
            ORIGEM / 'validacao_2000.json',
            pasta / 'validacao_2000.json',
        )

    print('EXECUÇÃO:', pasta, flush=True)
    print('Guarde este caminho para RETOMAR. São 3.000 atualizações adicionais.', flush=True)

    if (pasta / 'resumo.json').exists():
        print((pasta / 'resumo.json').read_text())
        return pasta

    modelo = otimizador = None
    try:
        modelo = construir_modelo('tiny', REPO)
        estado_align = instalar(modelo)
        otimizador = torch.optim.AdamW(
            modelo.parameters(), lr=anterior['lr'],
            betas=tuple(anterior['betas']), eps=anterior['eps'],
            weight_decay=anterior['weight_decay'],
        )

        ultimo = pasta / 'checkpoint_latest.pt'
        salvo = torch.load(
            ultimo if ultimo.exists() else ckpt_origem,
            map_location='cpu', weights_only=True,
        )
        assert salvo['config'] == (config if ultimo.exists() else anterior)
        assert salvo['formato'] == 1 and salvo['scheduler'] is None

        passo = int(salvo['passo'])
        assert 2000 <= passo <= 5000
        if not ultimo.exists():
            assert passo == 2000

        modelo.load_state_dict(salvo['modelo'], strict=True)
        otimizador.load_state_dict(salvo['otimizador'])
        baseline = salvo['baseline']
        restaurar_rng(salvo['rng'])
        del salvo

        if not ultimo.exists():
            salvar_checkpoint(pasta, modelo, otimizador, passo, config, baseline)

        print(
            f'Retomando após passo {passo}: pesos, AdamW e RNG restaurados.',
            flush=True,
        )
        log = pasta / f'passos_{agora()}.jsonl'
        gravar_json(
            pasta / f'tentativa_{agora()}.json',
            {'passo_inicial':passo, 'log':log.name},
        )

        def validar(n):
            if n in config['checkpoints_preservados']:
                destino = pasta / f'checkpoint_{n:04d}.pt'
                if not destino.exists():
                    import os
                    try:
                        os.link(ultimo, destino)
                    except OSError:
                        tmp = destino.with_suffix('.tmp')
                        shutil.copy2(ultimo, tmp)
                        os.replace(tmp, destino)

            arquivo = pasta / f'validacao_{n:04d}.json'
            if not arquivo.exists():
                rng = estado_rng()
                try:
                    v = avaliar(modelo, dados, planos['validacao'], arquivo)
                finally:
                    restaurar_rng(rng)
                print(
                    f"Validação {n}: CE={v['ce_media']:.6f}; "
                    f"Dice pseudo-FG={v['dice_pseudo_fg_medio']:.6f}",
                    flush=True,
                )

        if passo % 250 == 0:
            validar(passo)

        modelo.train()
        for numero in range(passo+1, 5001):
            r = planos['treino'][numero-1]
            s, fg, q, alvo = carregar_episodio(dados, r, 'cuda')
            otimizador.zero_grad(set_to_none=True)
            estado_align['ultimo'] = None

            retorno = modelo(
                [[s]], [[fg]], [[1-fg]], [q],
                isval=False, val_wsize=2, show_viz=False,
            )
            logits, alinhamento = retorno[:2]
            assert logits.shape == (1,2,256,256)
            assert torch.isfinite(logits).all().item()

            ce = F.cross_entropy(logits, alvo)
            total = ce + alinhamento
            assert torch.isfinite(total).item()
            total.backward()

            gradientes = [
                p.grad for p in modelo.encoder.parameters()
                if p.grad is not None
            ]
            assert gradientes
            assert all(torch.isfinite(g).all().item() for g in gradientes)
            assert any(torch.count_nonzero(g).item() > 0 for g in gradientes)

            otimizador.step()
            assert all(torch.isfinite(p).all().item() for p in modelo.parameters())

            linha = {
                'passo':numero, 'indice':r['indice'],
                'imagem':r['imagem'], 'regiao':r['regiao'],
                'ce':ce.item(), 'alinhamento':float(alinhamento),
                'total':total.item(),
                'alinhamento_excepcao':estado_align['ultimo'],
            }
            with log.open('a') as f:
                f.write(json.dumps(linha, ensure_ascii=False)+'\n')

            if numero % 10 == 0 or estado_align['ultimo'] is not None:
                print(
                    f"{numero}/5000 | CE={linha['ce']:.6f} | "
                    f"alinhamento={linha['alinhamento']:.6f} | "
                    f"total={linha['total']:.6f} | "
                    f"exceção={estado_align['ultimo']}",
                    flush=True,
                )

            del retorno, logits, alinhamento, ce, total, gradientes

            if numero % 50 == 0:
                salvar_checkpoint(
                    pasta, modelo, otimizador, numero, config, baseline
                )
            if numero % 250 == 0:
                validar(numero)

        curva = []
        for n in range(2000,5001,250):
            v = json.loads((pasta / f'validacao_{n:04d}.json').read_text())
            curva.append({
                'passo':n, 'ce':v['ce_media'],
                'dice_pseudo_fg':v['dice_pseudo_fg_medio'],
            })

        resumo = {
            'variante':'tiny', 'atualizacoes':5000,
            'adicionais':3000, 'n_validacao':145,
            'curva':curva, 'origem':str(ORIGEM),
            'regra_degenerada':config['alinhamento_degenerado'],
            'limite':'Pseudo-máscaras; avaliação ALP + SAM separada.',
        }
        shutil.copy2(
            pasta / 'validacao_5000.json',
            pasta / 'validacao_final.json',
        )
        gravar_json(pasta / 'resumo.json', resumo)
        print(json.dumps(resumo, indent=2, ensure_ascii=False))
        return pasta

    except BaseException:
        (pasta / f'erro_{agora()}.txt').write_text(traceback.format_exc())
        print('Para retomar: RETOMAR =', repr(str(pasta)), flush=True)
        raise

    finally:
        if modelo is not None:
            modelo.zero_grad(set_to_none=True)
        del modelo, otimizador
        gc.collect()
        torch.cuda.empty_cache()
