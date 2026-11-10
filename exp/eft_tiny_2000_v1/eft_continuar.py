"""Continuação isolada do Tiny, do passo 200 ao 2000."""
from eft_treino import *
import eft_treino
import eft_plano_2000
from eft_plano_2000 import estender

PILOTO = Path('/workspace/ProtoSAM-repro/reproducao/eft_tiny_piloto_20261011T152746615659Z')
DADOS = Path('/workspace/ProtoSAM-repro/reproducao/eft_sp_piloto_dados_20261011T152315570066Z')

def executar(retomar=None, piloto=PILOTO, dados=DADOS, repo=Path('/workspace/ProtoSAM-repro')):
    piloto, dados, repo = map(lambda p: Path(p).resolve(), (piloto, dados, repo))
    origem = json.loads((piloto / 'config.json').read_text())
    assert origem['variante'] == 'tiny' and origem['atualizacoes'] == 200
    assert origem['dados'] == str(dados)
    assert origem['planos_sha256'] == sha256(dados / 'planos.json')
    assert sha256(origem['checkpoint_origem']) == origem['checkpoint_origem_sha256']
    # As fontes do piloto permanecem intactas. Cópias locais também devem ser idênticas.
    for nome, digest in origem['fontes_sha256'].items():
        assert sha256(nome) == digest, f'Fonte original alterada: {nome}'
        if Path(nome).name in ('eft_dados.py', 'eft_modelo.py', 'eft_treino.py'):
            assert sha256(Path(__file__).parent / Path(nome).name) == digest
    for modulo in (eft_dados, eft_modelo, eft_treino, eft_plano_2000):
        assert Path(modulo.__file__).resolve().parent == Path(__file__).resolve().parent, 'Reinicie o kernel para importar este pacote.'
    ambiente = origem['ambiente']
    torch.backends.cuda.matmul.allow_tf32 = ambiente['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = ambiente['tf32_cudnn']
    torch.backends.cudnn.benchmark = ambiente['cudnn_benchmark']
    torch.use_deterministic_algorithms(ambiente['deterministico'])
    assert versoes() == ambiente, 'Ambiente diferente do piloto; interrompido.'
    antigos = json.loads((dados / 'planos.json').read_text())
    divisao = json.loads((dados / 'divisao.json').read_text())
    treino = []
    for item in divisao:
        if item['split'] == 'treino':
            r = json.loads((dados / 'cache' / f"{item['indice']:04d}.json").read_text())
            assert all(r[k] == v for k, v in item.items())
            treino.append(r)
    planos = {'treino': estender(treino, antigos['treino'], antigos['validacao']),
              'validacao': antigos['validacao']}
    for r in treino + planos['validacao']:
        assert sha256(dados / r['cache']) == r['cache_sha256'], f"Cache alterado: {r['cache']}"
    fontes = sorted(Path(__file__).resolve().parent.glob('*.py'))
    config = dict(origem)
    config.update(versao=2, atualizacoes=2000, passo_origem=200,
                  piloto_origem=str(piloto),
                  checkpoint_piloto_sha256=sha256(piloto / 'checkpoint_latest.pt'),
                  plano_estendido_sha256=assinatura(planos),
                  fontes_continuacao_sha256={str(f):sha256(f) for f in fontes},
                  checkpoint_a_cada=50, validacao='pseudo-mascaras fixas a cada 250 passos',
                  checkpoints_preservados=[500,1000,1500,2000])
    pasta = Path(retomar).resolve() if retomar else repo / 'reproducao' / f'eft_tiny_2000_{agora()}'
    assert pasta != piloto, 'Retome a nova execução, nunca a pasta do piloto.'
    if retomar:
        assert json.loads((pasta / 'config.json').read_text()) == config, 'Configuração alterada.'
        assert json.loads((pasta / 'planos.json').read_text()) == planos
    else:
        pasta.mkdir(parents=True, exist_ok=False)
        gravar_json(pasta / 'config.json', config)
        gravar_json(pasta / 'planos.json', planos)
        (pasta / 'fontes').mkdir()
        for f in fontes:
            shutil.copy2(f, pasta / 'fontes' / f.name)
        shutil.copy2(piloto / 'validacao_inicial.json', pasta / 'validacao_inicial.json')
        shutil.copy2(piloto / 'validacao_final.json', pasta / 'validacao_0200.json')
    print('EXECUÇÃO:', pasta, flush=True)
    print('Guarde este caminho para retomar. Total: 2000; adicionais: 1800.', flush=True)
    if (pasta / 'resumo.json').exists():
        print((pasta / 'resumo.json').read_text())
        return pasta
    tentativa = agora()
    log = pasta / f'passos_{tentativa}.jsonl'
    modelo = otimizador = None
    try:
        modelo = construir_modelo('tiny', repo)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-5,
                                      betas=(0.9,0.999), eps=1e-8, weight_decay=0.)
        proprio = pasta / 'checkpoint_latest.pt'
        salvo = torch.load(proprio if proprio.exists() else piloto / 'checkpoint_latest.pt',
                           map_location='cpu', weights_only=True)
        assert salvo['formato'] == 1 and salvo['scheduler'] is None
        assert salvo['config'] == (config if proprio.exists() else origem)
        passo = int(salvo['passo'])
        assert 200 <= passo <= 2000
        if not proprio.exists():
            assert passo == 200
        modelo.load_state_dict(salvo['modelo'], strict=True)
        otimizador.load_state_dict(salvo['otimizador'])
        baseline = salvo['baseline']
        restaurar_rng(salvo['rng'])
        del salvo
        if not proprio.exists():
            salvar_checkpoint(pasta, modelo, otimizador, passo, config, baseline)
        print(f'Retomando após passo {passo}; AdamW e RNG restaurados.', flush=True)

        def validar_passo(n):
            if n in config['checkpoints_preservados']:
                destino = pasta / f'checkpoint_{n:04d}.pt'
                if not destino.exists():
                    try:
                        os.link(proprio, destino)
                    except OSError:
                        tmp = destino.with_suffix('.tmp')
                        shutil.copy2(proprio, tmp)
                        os.replace(tmp, destino)
            arquivo = pasta / f'validacao_{n:04d}.json'
            if not arquivo.exists():
                rng = estado_rng()
                try:
                    resultado = avaliar(modelo, dados, planos['validacao'], arquivo)
                finally:
                    restaurar_rng(rng)
                print(f"Passo {n}: CE validação={resultado['ce_media']:.6f}; Dice pseudo-FG={resultado['dice_pseudo_fg_medio']:.6f}", flush=True)
        if passo % 250 == 0:
            validar_passo(passo)
        gravar_json(pasta / f'tentativa_{tentativa}.json', {'passo_inicial':passo,'log':log.name})
        modelo.train()
        for numero in range(passo + 1, 2001):
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
                print(f'{numero:03d}/2000 | CE={linha["ce"]:.6f} | alinhamento={linha["alinhamento"]:.6f} | total={linha["total"]:.6f}', flush=True)
            del retorno, logits, alinhamento, ce, total, gradientes
            if numero % 50 == 0:
                salvar_checkpoint(pasta, modelo, otimizador, numero, config, baseline)
            if numero % 250 == 0:
                validar_passo(numero)
        final = json.loads((pasta / 'validacao_2000.json').read_text())
        shutil.copy2(pasta / 'validacao_2000.json', pasta / 'validacao_final.json')
        curva = []
        for n in [200] + list(range(250,2001,250)):
            v = json.loads((pasta / f'validacao_{n:04d}.json').read_text())
            curva.append({'passo':n, 'ce':v['ce_media'], 'dice_pseudo_fg':v['dice_pseudo_fg_medio']})
        resumo = {'variante':'tiny', 'atualizacoes':2000, 'adicionais':1800,
                  'imagens_treino_distintas':1304, 'n_validacao':145,
                  'ce_validacao_passo_0':baseline['ce_media'],
                  'dice_pseudo_fg_passo_0':baseline['dice_pseudo_fg_medio'],
                  'curva':curva, 'checkpoint':'checkpoint_latest.pt',
                  'limite':'Validação de pseudo-máscaras; avaliar checkpoints no ALP + SAM separadamente.'}
        gravar_json(pasta / 'resumo.json', resumo)
        print(json.dumps(resumo, ensure_ascii=False, indent=2))
        return pasta
    except BaseException:
        (pasta / f'erro_{tentativa}.txt').write_text(traceback.format_exc())
        print('Retome com RETOMAR =', repr(str(pasta)), flush=True)
        raise
    finally:
        if modelo is not None:
            modelo.zero_grad(set_to_none=True)
        del modelo, otimizador
