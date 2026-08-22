"""
Exporta as runs do W&B e gera as figuras da dissertação.

Regra do guia: nenhuma figura sai de CSV avulso. Tudo vem de `wandb.Api()`, de
modo que as figuras sempre correspondem ao que foi de fato registrado, e são
regeneráveis por qualquer pessoa com acesso ao projeto.

Saída:
    exp/figures/*.png e *.pdf   (PDF para entrar no LaTeX)
    exp/out/export_*.csv        (a "visão de tabela" — todo dado de toda figura)

Uso:
    python -m exp.export_runs
    python -m exp.export_runs --only f1 f6
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.envinfo import REPO_ROOT

FIG_DIR = os.path.join(REPO_ROOT, "exp/figures")
OUT_DIR = os.path.join(REPO_ROOT, "exp/out")

# --- paleta -----------------------------------------------------------------
# Validada com scripts/validate_palette.js (modo light, superfície #fcfcfb):
# banda de luminosidade PASS, piso de croma PASS, separação CVD PASS
# (pior par adjacente ΔE 9.1), piso de visão normal PASS (ΔE 19.6).
# Aviso de contraste <3:1 em aqua/amarelo/magenta -> regra de alívio: rótulos
# diretos visíveis + CSV de apoio, ambos entregues aqui.
# A cor segue a ENTIDADE (o encoder), nunca a ordem/posto na figura.
SERIES = {
    "dinov2_l14":          {"c": "#2a78d6", "m": "o", "ls": "-",  "label": "DINOv2-L"},
    "dinov2_b14":          {"c": "#eb6834", "m": "s", "ls": "--", "label": "DINOv2-B"},
    "dinov2_s14":          {"c": "#1baf7a", "m": "^", "ls": "-.", "label": "DINOv2-S"},
    "vmamba_tiny":         {"c": "#eda100", "m": "D", "ls": "-",  "label": "VMamba-T"},
    "sam_h_image_encoder": {"c": "#e87ba4", "m": "*", "ls": ":",  "label": "SAM-H (encoder)"},
}
# Segmentos do E4 — parte-de-todo, ordem fixa (slots 1..4).
STAGES = [("encoder_ms", "#2a78d6", "Encoder"),
          ("alp_ms",     "#eb6834", "ALP (protótipos)"),
          ("prompts_ms", "#1baf7a", "Prompts (CCA, bbox)"),
          ("sam_ms",     "#eda100", "SAM-H")]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def style_axes(ax, xlabel, ylabel, title=None):
    """Grade e eixos recessivos; sem moldura; tinta de texto, nunca cor de série."""
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, pad=10, loc="left")


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  -> exp/figures/{name}.png / .pdf")


def dump(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"export_{name}.csv")
    df.to_csv(p, index=False)
    print(f"  -> exp/out/export_{name}.csv  ({len(df)} linhas)")


# --- coleta -----------------------------------------------------------------

def fetch(entity, project):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=500)
    rows = []
    for r in runs:
        if r.state != "finished":
            continue
        d = {"name": r.name, "group": r.group or "", "id": r.id,
             "created_at": str(r.created_at)}
        d.update({f"cfg.{k}": v for k, v in r.config.items()
                  if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))})
        d.update({f"sum.{k}": v for k, v in r.summary.items()
                  if not k.startswith("_") and isinstance(v, (int, float, str, bool))})
        rows.append(d)
    df = pd.DataFrame(rows)
    print(f"{len(df)} runs concluidas em {entity}/{project}")

    # Tentativas superadas continuam no projeto (ex.: as duas primeiras versões do
    # E1b, com contabilidade errada). O nome da run codifica a identidade do
    # experimento, então a mais recente com o mesmo nome é a que vale.
    before = len(df)
    df = (df.sort_values("created_at")
            .drop_duplicates(subset=["group", "name"], keep="last")
            .reset_index(drop=True))
    if before != len(df):
        print(f"  {before - len(df)} runs superadas descartadas (mesmo nome, versao antiga)")
    return df


def pick(df, group_prefix):
    return df[df["group"].str.startswith(group_prefix)].copy() if len(df) else df


# --- figuras ----------------------------------------------------------------

def f1_dice_resolution(df):
    """Dice × resolução, com desvio entre suportes. A figura principal."""
    for dom, titulo in [("kvasir", "suporte do mesmo domínio (Kvasir)"),
                        ("clinicdb", "suporte de outro domínio (CVC-ClinicDB)")]:
        sub = pick(df, f"E2-quality-{dom}")
        if sub.empty:
            print(f"  [skip] f1 {dom}: sem runs")
            continue
        g = (sub.groupby(["cfg.backbone", "cfg.input_size"])["sum.mean_dice"]
             .agg(["mean", "std", "count"]).reset_index())
        dump(g, f"f1_dice_{dom}")

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        style_axes(ax, "Resolução de entrada (px)", "Dice médio",
                   f"Dice × resolução — {titulo}")
        for enc in ["dinov2_l14", "vmamba_tiny"]:
            s = g[g["cfg.backbone"] == enc].sort_values("cfg.input_size")
            if s.empty:
                continue
            st = SERIES[enc]
            ax.errorbar(s["cfg.input_size"], s["mean"], yerr=s["std"],
                        color=st["c"], marker=st["m"], linestyle=st["ls"],
                        linewidth=2, markersize=7, capsize=3, capthick=1,
                        elinewidth=1, label=st["label"], zorder=3)
            # rótulo direto no último ponto (regra de alívio do contraste)
            last = s.iloc[-1]
            ax.annotate(f'{last["mean"]:.3f}', (last["cfg.input_size"], last["mean"]),
                        textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=9, color=INK2)
        ax.set_xticks([256, 384, 512, 672, 768, 1024])
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="best")
        save(fig, f"f1_dice_resolucao_{dom}")


def f2_latency(df):
    """Latência do encoder × resolução (E1)."""
    sub = pick(df, "E1-cost")
    sub = sub[sub["group"] == "E1-cost"]
    sub = sub[sub["cfg.encoder"].notna()] if "cfg.encoder" in sub else sub
    if sub.empty:
        print("  [skip] f2: sem runs")
        return
    cols = ["cfg.encoder", "cfg.input_size", "cfg.latency_ms_median",
            "cfg.latency_ms_p25", "cfg.latency_ms_p75", "cfg.gflops",
            "cfg.vram_weights_mb", "cfg.vram_activation_mb", "cfg.tokens_real"]
    g = sub[[c for c in cols if c in sub]].dropna(subset=["cfg.latency_ms_median"])
    g = g.sort_values(["cfg.encoder", "cfg.input_size"])
    dump(g, "f2_custo_encoder")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style_axes(ax, "Resolução de entrada (px)", "Latência mediana (ms)",
               "Custo do encoder isolado — latência")
    for enc in ["dinov2_l14", "dinov2_b14", "dinov2_s14", "vmamba_tiny"]:
        s = g[g["cfg.encoder"] == enc].sort_values("cfg.input_size")
        s = s[s["cfg.input_size"].isin([256, 384, 512, 672, 768, 1024])]
        if s.empty:
            continue
        st = SERIES[enc]
        ax.fill_between(s["cfg.input_size"], s["cfg.latency_ms_p25"], s["cfg.latency_ms_p75"],
                        color=st["c"], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(s["cfg.input_size"], s["cfg.latency_ms_median"], color=st["c"],
                marker=st["m"], linestyle=st["ls"], linewidth=2, markersize=7,
                label=st["label"], zorder=3)
        last = s.iloc[-1]
        ax.annotate(f'{last["cfg.latency_ms_median"]:.0f} ms',
                    (last["cfg.input_size"], last["cfg.latency_ms_median"]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=9, color=INK2)
    sam = g[g["cfg.encoder"] == "sam_h_image_encoder"]
    if not sam.empty:
        y = float(sam["cfg.latency_ms_median"].iloc[0])
        ax.axhline(y, color=MUTED, linestyle=":", linewidth=1.2, zorder=1)
        ax.annotate(f"SAM-H (encoder) @1024: {y:.0f} ms", (260, y),
                    textcoords="offset points", xytext=(0, 5), fontsize=8.5, color=MUTED)
    ax.set_xticks([256, 384, 512, 672, 768, 1024])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    save(fig, "f2_latencia_encoder")


def f3_vram(df):
    """VRAM separada em pesos e ativação — o conserto do D2. Parte-de-todo."""
    sub = df[df["group"] == "E1-cost"]
    if sub.empty or "cfg.vram_weights_mb" not in sub:
        print("  [skip] f3: sem runs")
        return
    encs = ["dinov2_l14", "dinov2_b14", "dinov2_s14", "vmamba_tiny"]
    sizes = [256, 384, 512, 672, 768, 1024]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    style_axes(ax, "", "VRAM (MB)", "Memória do encoder — pesos vs ativação")
    x, rows, ticks, groups = 0, [], [], []
    for enc in encs:
        s = sub[sub["cfg.encoder"] == enc].sort_values("cfg.input_size")
        s = s[s["cfg.input_size"].isin(sizes)]
        if s.empty:
            continue
        start = x
        for _, r in s.iterrows():
            w, a = float(r["cfg.vram_weights_mb"]), float(r["cfg.vram_activation_mb"])
            ax.bar(x, w, color="#c3c2b7", width=0.78, zorder=3)
            # espaçador de superfície entre os segmentos empilhados
            ax.bar(x, a, bottom=w + 8, color=SERIES[enc]["c"], width=0.78, zorder=3)
            ticks.append((x, str(int(r["cfg.input_size"]))))
            rows.append({"encoder": enc, "input_size": int(r["cfg.input_size"]),
                         "vram_weights_mb": w, "vram_activation_mb": a})
            x += 1
        groups.append(((start + x - 1) / 2, SERIES[enc]["label"]))
        x += 1
    dump(pd.DataFrame(rows), "f3_vram")

    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=8, rotation=0)
    ax.set_xlim(-1, x - 1)
    # Identidade do grupo por texto, nunca só por cor.
    ymin = ax.get_ylim()[0]
    for cx, label in groups:
        ax.annotate(label, (cx, ymin), xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=10, color=INK, annotation_clip=False)
    ax.annotate("Resolução de entrada (px)", (0.5, -0.10), xycoords="axes fraction",
                ha="center", fontsize=9, color=MUTED, annotation_clip=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7"),
               *[plt.Rectangle((0, 0), 1, 1, color=SERIES[e]["c"]) for e in encs]]
    ax.legend(handles, ["Pesos (constantes)"] + [f'Ativação — {SERIES[e]["label"]}' for e in encs],
              frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.42))
    save(fig, "f3_vram_pesos_ativacao")


def f4_e4_breakdown(df):
    """Decomposição do tempo por imagem — barra empilhada horizontal."""
    sub = df[df["group"] == "E4-e2e"]
    if sub.empty or "sum.e4_total_ms" not in sub:
        print("  [skip] f4: sem runs")
        return
    sub = sub.sort_values(["cfg.backbone", "cfg.input_size"], ascending=[False, True])
    rows, ylabels = [], []
    for _, r in sub.iterrows():
        rows.append({s[0]: float(r.get(f"sum.e4_{s[0]}", 0.0)) for s in STAGES}
                    | {"total_ms": float(r["sum.e4_total_ms"]),
                       "encoder": r["cfg.backbone"], "input_size": int(r["cfg.input_size"])})
        ylabels.append(f'{SERIES[r["cfg.backbone"]]["label"]}\n{int(r["cfg.input_size"])} px')
    d = pd.DataFrame(rows)
    dump(d, "f4_e2e")

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    style_axes(ax, "Tempo por imagem (ms)", "", "Onde o tempo é gasto, fim a fim")
    ax.grid(axis="y", visible=False)
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for key, color, label in STAGES:
        ax.barh(y, d[key], left=left, color=color, height=0.62, label=label, zorder=3)
        left = left + d[key].values + 3.0  # 3px de superfície entre segmentos
    for i, r in d.iterrows():
        frac = r["encoder_ms"] / r["total_ms"]
        # O rótulo só cabe dentro quando o segmento é largo; senão vai para fora,
        # em tinta de texto, para não invadir os segmentos vizinhos.
        if frac > 0.25:
            ax.annotate(f'encoder {frac*100:.0f}%', (r["encoder_ms"] / 2, i),
                        ha="center", va="center", fontsize=8.5, color="#ffffff", zorder=5)
            ax.annotate(f'{r["total_ms"]:.0f} ms', (left[i], i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
        else:
            ax.annotate(f'{r["total_ms"]:.0f} ms  ·  encoder {frac*100:.0f}%',
                        (left[i], i), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, -0.32))
    save(fig, "f4_decomposicao_e2e")


def f5_token_parity(df):
    """Paridade de tokens — o desconfundimento do D3."""
    e2b = pick(df, "E2b-tokenparity")
    e2k = pick(df, "E2-quality-kvasir")
    if e2b.empty or e2k.empty:
        print("  [skip] f5: sem runs")
        return
    pares = [(16, 224, 512), (24, 336, 768), (32, 448, 1024)]
    rows = []
    for tok, ds, vs in pares:
        a = e2b[(e2b["cfg.backbone"] == "dinov2_l14") & (e2b["cfg.input_size"] == ds)]["sum.mean_dice"]
        b = e2k[(e2k["cfg.backbone"] == "vmamba_tiny") & (e2k["cfg.input_size"] == vs)]["sum.mean_dice"]
        if a.empty or b.empty:
            continue
        rows.append({"tokens": tok, "dinov2_input": ds, "vmamba_input": vs,
                     "dinov2_dice": a.mean(), "dinov2_std": a.std(),
                     "vmamba_dice": b.mean(), "vmamba_std": b.std()})
    if not rows:
        print("  [skip] f5: pares incompletos")
        return
    d = pd.DataFrame(rows)
    dump(d, "f5_paridade_tokens")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    style_axes(ax, "Tokens reais entregues ao ALP", "Dice médio",
               "Mesma granularidade: representação, sem confundimento")
    x = np.arange(len(d))
    w = 0.36
    for i, (enc, key, sd) in enumerate([("dinov2_l14", "dinov2_dice", "dinov2_std"),
                                        ("vmamba_tiny", "vmamba_dice", "vmamba_std")]):
        st = SERIES[enc]
        off = (i - 0.5) * (w + 0.03)   # espaçador de superfície entre barras vizinhas
        ax.bar(x + off, d[key], width=w, color=st["c"], zorder=3, label=st["label"],
               yerr=d[sd], capsize=3, error_kw=dict(elinewidth=1, capthick=1, ecolor=MUTED))
        for xi, v in zip(x + off, d[key]):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r.tokens}²\n({r.dinov2_input} vs {r.vmamba_input} px)'
                        for r in d.itertuples()], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    save(fig, "f5_paridade_tokens")


def f6_launchbound(df):
    """Fração de tempo em que a GPU fica ociosa (E1b)."""
    sub = df[df["group"] == "E1b-launchbound"]
    if sub.empty or "cfg.gpu_idle_pct" not in sub:
        print("  [skip] f6: sem runs")
        return
    g = sub[["cfg.encoder", "cfg.input_size", "cfg.batch", "cfg.gpu_idle_pct",
             "cfg.cuda_over_wall", "cfg.wall_per_image_ms", "cfg.kernels_per_iter"]].dropna()
    g = g.sort_values(["cfg.encoder", "cfg.input_size", "cfg.batch"])
    dump(g, "f6_launchbound")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    style_axes(ax, "Batch", "GPU ociosa (%)",
               "A GPU está esperando ou trabalhando?")
    for enc in ["vmamba_tiny", "dinov2_l14"]:
        for size, alpha, ls in [(512, 1.0, "-"), (1024, 0.55, "--")]:
            s = g[(g["cfg.encoder"] == enc) & (g["cfg.input_size"] == size)].sort_values("cfg.batch")
            if s.empty:
                continue
            st = SERIES[enc]
            ax.plot(s["cfg.batch"], s["cfg.gpu_idle_pct"].clip(lower=0), color=st["c"],
                    marker=st["m"], linestyle=ls, linewidth=2, markersize=7, alpha=alpha,
                    label=f'{st["label"]} @{size}', zorder=3)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    save(fig, "f6_gpu_ociosa")


FIGS = {"f1": f1_dice_resolution, "f2": f2_latency, "f3": f3_vram,
        "f4": f4_e4_breakdown, "f5": f5_token_parity, "f6": f6_launchbound}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="leodegario")
    p.add_argument("--project", default="qualification")
    p.add_argument("--only", nargs="+", default=None, choices=list(FIGS))
    a = p.parse_args()

    df = fetch(a.entity, a.project)
    dump(df, "todas_as_runs")
    for k, fn in FIGS.items():
        if a.only and k not in a.only:
            continue
        print(f"[{k}] {fn.__doc__.splitlines()[0]}")
        try:
            fn(df)
        except Exception as e:
            print(f"  ERRO em {k}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
