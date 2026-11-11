"""Ajuste auto-supervisionado do encoder com regiões como pseudo-máscaras, a 672 px.

É a receita do melhor VMamba de out/2026 (ProtoSAM-repro, eft_SAM_vs_SP256_1231), agora para
qualquer encoder de construir_alp. Um episódio = uma imagem sem rótulo e uma região dela:
o suporte é a imagem com a região como máscara, a query é o espelho horizontal dos dois.
Perda: entropia cruzada + alinhamento oficial (pesos 1 e 1), AdamW, um episódio por passo.

O plano (pasta com planos.json e config.json) diz que região usar em cada passo. Dois tipos:
    SAM    região = d['candidatas'][regiao] do cache (máscaras geradas pelo SAM a 512)
    SP256  região = (d['segmentos'] == regiao) & d['util'] (superpixels de Felzenszwalb a 256)

Saída em runs/<data>_treino_<nome>/: config.json, planos.json, passos_*.jsonl,
checkpoint_NNNN.pt (marcos) e checkpoint_latest.pt (reexecutar com --run continua daqui).
"""
from pathlib import Path
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from protovm import RAIZ
from protovm.eft_treino import estado_rng, instalar_alinhamento_seguro, restaurar_rng
from protovm.encoder import construir_alp
from protovm.util import agora, ambiente, fixar_numerica, git_commit, gravar_json, ler_json, sha256

LADO = 672
MEDIA = [123.675, 116.28, 103.53]
DESVIO = [58.395, 57.12, 57.375]


def mascara_da_regiao(registro, tipo, pasta_plano):
    """Pseudo-máscara booleana (672, 672) do episódio."""
    with np.load(Path(pasta_plano) / registro['cache'], allow_pickle=False) as d:
        if tipo == 'SAM':
            m = d['candidatas'][registro['regiao']].copy()
            assert m.shape == (512, 512)
        elif tipo == 'SP256':
            m512 = (d['segmentos'] == registro['regiao']) & d['util']
            m = m512[::2, ::2].copy()  # a região de 256 que originou o cache de 512
            assert np.array_equal(m512, m.repeat(2, axis=0).repeat(2, axis=1))
        else:
            raise ValueError(f'Tipo de pseudo-máscara desconhecido: {tipo}')
    m = np.array(Image.fromarray(m.astype(np.uint8)).resize((LADO, LADO), Image.Resampling.NEAREST)) > 0
    assert m.any()
    return m


def episodio(registro, tipo, pasta_plano):
    """(suporte, máscara do suporte, query, alvo da query), tudo na GPU."""
    mask = mascara_da_regiao(registro, tipo, pasta_plano)
    assert sha256(registro['imagem']) == registro['imagem_sha256']
    with Image.open(registro['imagem']) as im:
        rgb = np.array(im.convert('RGB').resize((LADO, LADO), Image.Resampling.BILINEAR))

    def imagem(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].to('cuda', dtype=torch.float32)
        return (t - t.new_tensor(MEDIA)[None, :, None, None]) / t.new_tensor(DESVIO)[None, :, None, None]

    return (imagem(rgb), torch.from_numpy(mask.copy())[None].to('cuda', dtype=torch.float32),
            imagem(np.fliplr(rgb)), torch.from_numpy(np.fliplr(mask).copy())[None].to('cuda', dtype=torch.long))


def treinar(cfg, pasta=None, max_passos=None):
    """cfg: dict do YAML (ver configs/eft_regioes). max_passos interrompe antes do fim (ensaio)."""
    fixar_numerica()
    pasta_plano = Path(cfg['plano']).expanduser()
    tipo = ler_json(pasta_plano / 'config.json')['tipo_pseudolabel']
    plano = ler_json(pasta_plano / 'planos.json')['treino']
    total = cfg['passos']
    assert len(plano) >= total, (len(plano), total)
    marcos = set(cfg['checkpoints']) | {0}

    registro_cfg = dict(cfg, tipo_pseudolabel=tipo, plano_sha256=sha256(pasta_plano / 'planos.json'),
                        resolucao=LADO, loss='CE + alinhamento; pesos 1 e 1', otimizador='AdamW')
    registro_cfg.setdefault('estagio', None)  # scripts/avaliar.py lê estes dois do treino
    registro_cfg.setdefault('norm', None)
    if pasta is None:
        pasta = RAIZ / 'runs' / f"{agora()}_treino_{cfg['nome']}"
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_cfg = pasta / 'config.json'
    if arquivo_cfg.exists():
        salvo = ler_json(arquivo_cfg)
        assert {k: salvo.get(k) for k in registro_cfg} == registro_cfg, 'A configuração mudou; retomada recusada.'
    else:
        gravar_json(arquivo_cfg, dict(registro_cfg, git=git_commit(), ambiente=ambiente()))
        shutil.copy2(pasta_plano / 'planos.json', pasta / 'planos.json')
    print('TREINO:', pasta, flush=True)

    alp = construir_alp(cfg['modelo'], cfg.get('estagio') or 3, cfg.get('norm') or 'checkpoint', LADO,
                        grade=cfg.get('grade', 32), treino=True, semente=cfg['semente'])
    estado_align = instalar_alinhamento_seguro(alp)
    otimizador = torch.optim.AdamW(alp.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=cfg['weight_decay'])
    latest = pasta / 'checkpoint_latest.pt'

    def salvar(n):
        tmp = latest.with_suffix('.tmp')
        torch.save({'formato': 1, 'passo': n, 'modelo': alp.state_dict(), 'otimizador': otimizador.state_dict(),
                    'rng': estado_rng(), 'config': registro_cfg}, tmp)
        os.replace(tmp, latest)
        destino = pasta / f'checkpoint_{n:04d}.pt'
        if n in marcos and not destino.exists():
            try:
                os.link(latest, destino)
            except OSError:
                shutil.copy2(latest, destino)

    passo = 0
    if latest.exists():
        salvo = torch.load(latest, map_location='cpu', weights_only=True)
        assert salvo['config'] == registro_cfg
        alp.load_state_dict(salvo['modelo'], strict=True)
        otimizador.load_state_dict(salvo['otimizador'])
        passo = int(salvo['passo'])
        restaurar_rng(salvo['rng'])
        del salvo
        print(f'Retomada após o passo {passo}.', flush=True)
    else:
        salvar(0)

    alp.train()
    torch.cuda.reset_peak_memory_stats()
    inicio, primeiro = time.perf_counter(), passo
    fim = min(total, max_passos or total)
    with (pasta / f'passos_{time.time_ns()}.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for n in range(passo + 1, fim + 1):
            t0 = time.perf_counter()
            r = plano[n - 1]
            s, fg, q, alvo = episodio(r, tipo, pasta_plano)
            otimizador.zero_grad(set_to_none=True)
            estado_align['ultimo'] = None
            logits, alinhamento = alp([[s]], [[fg]], [[1 - fg]], [q], isval=False, val_wsize=2, show_viz=False)[:2]
            assert tuple(logits.shape) == (1, 2, LADO, LADO) and torch.isfinite(logits).all()
            ce = F.cross_entropy(logits, alvo)
            loss = ce + alinhamento
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in alp.encoder.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            otimizador.step()
            total_passo, ce_passo = loss.item(), ce.item()
            log.write(json.dumps({
                'passo': n, 'indice': r['indice'], 'regiao': r['regiao'], 'ce': ce_passo,
                'alinhamento': float(alinhamento.detach()) if torch.is_tensor(alinhamento) else float(alinhamento),
                'total': total_passo, 'alinhamento_excecao': estado_align['ultimo'],
                'segundos': time.perf_counter() - t0}, ensure_ascii=False) + '\n')
            del logits, alinhamento, ce, loss, grads, s, fg, q, alvo
            if n % cfg['salvar_a_cada'] == 0 or n in marcos or n == fim:
                salvar(n)
            if n == primeiro + 1 or n % 10 == 0:
                media = (time.perf_counter() - inicio) / (n - primeiro)
                print(f'{n}/{total} | loss={total_passo:.6f} | CE={ce_passo:.6f} | {media:.2f}s/passo | '
                      f'restante={media * (total - n) / 3600:.2f}h | '
                      f'pico={torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB', flush=True)
    if fim == total:
        gravar_json(pasta / 'concluido.json', {'passos': total,
                                               'pico_memoria_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3})
        print('TREINO CONCLUIDO', flush=True)
    return pasta
