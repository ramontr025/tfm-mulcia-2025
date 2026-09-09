"""Snakemake ``script:`` for compare (Stage 3).

Agrega las métricas JSON de todos los modelos evaluados y genera:
1. results/comparison.csv: tabla con los estimandos AGRUPADOS (pooled) por split
   para cada modelo, ordenada de forma descendente por el C-index en holdout.
2. results/comparison.png: C-index por edad de landmark (holdout) con banda de
   IC 95% bootstrap, una curva por modelo.
3. results/{auroc,brier,cindex_ipcw,ipa}.png: la misma curva por landmark con su
   banda de IC para cada una de las métricas nuevas (subproductos no declarados).
4. results/figure_main.png/.pdf: lámina central para paper/memoria con los
   paneles A (AUROC por landmark + eventos), B (IPA frente al modelo nulo) y
   C (forest de los estimandos agrupados con IC 95% de cluster bootstrap).

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecución.

Reads:  snakemake.input.metrics   lista de JSONs de métricas (results/metrics/{name}.json)
Writes: snakemake.output.csv      results/comparison.csv
        snakemake.output.png      results/comparison.png
        snakemake.output.main_png results/figure_main.png
        snakemake.output.main_pdf results/figure_main.pdf
"""

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

# Redirección de logs si Snakemake proporciona ruta de log
if hasattr(snakemake, "log") and len(snakemake.log) > 0:
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

# ── Estilo: paleta Okabe-Ito (segura para daltonismo) + marcador y trazo ─────
# ── distintos por modelo, para que la figura sea legible impresa en B/N. ─────
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

METRIC_LABELS = {
    "c_index": "C-index (Harrell)",
    "auroc": "AUROC a 5 años",
    "c_index_ipcw": "C-index IPCW (Uno)",
    "brier": "Brier score a 5 años",
    "ipa": "IPA a 5 años",
}
# PNG por métrica (subproductos en el mismo directorio que comparison.png)
METRIC_PNGS = {
    "auroc": "auroc.png",
    "brier": "brier.png",
    "c_index_ipcw": "cindex_ipcw.png",
    "ipa": "ipa.png",
}
FOREST_SPECS = [
    ("auroc", "AUROC (5 años)", ".3f", 0.5),
    ("c_index_ipcw", "C-index IPCW (Uno)", ".3f", 0.5),
    ("ipa", "IPA (5 años)", ".3f", 0.0),
    ("brier", "Brier (5 años)", ".4f", None),
]

# 1. Obtener archivos de entrada y salida
metric_files = (
    snakemake.input.metrics if hasattr(snakemake.input, "metrics") else snakemake.input
)
csv_output = getattr(snakemake.output, "csv", snakemake.output[0])
png_output = getattr(
    snakemake.output,
    "png",
    snakemake.output[1] if len(snakemake.output) > 1 else "results/comparison.png",
)
main_png_output = getattr(snakemake.output, "main_png", "results/figure_main.png")
main_pdf_output = getattr(snakemake.output, "main_pdf", "results/figure_main.pdf")
results_dir = Path(csv_output).parent

# 2. Leer y parsear cada JSON de métricas
MODELS_DATA = {}
for fpath in metric_files:
    fpath = Path(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_name = data.get("model_name", fpath.stem)
    MODELS_DATA[model_name] = data


def _split(data):
    """Split de referencia para las figuras: holdout si existe, si no val."""
    return "holdout" if "holdout" in data.get("c_index", {}) else "val"


def _pooled_point(data, metric_key, split):
    """Punto agrupado (mean_*) de un modelo; None si ausente o nulo."""
    value = data.get(metric_key, {}).get(split, {}).get(f"mean_{metric_key}")
    return float(value) if value is not None else np.nan


def _extract_pooled(data, metric_key, split):
    """Punto agrupado (mean_*) y su IC 95% de cluster bootstrap (pooled_ci)."""
    point = _pooled_point(data, metric_key, split)
    ci = data.get("pooled_ci", {}).get(split, {}).get(metric_key) or [None, None]
    lo, hi = ci[0], ci[1]
    lo = float(lo) if lo is not None else np.nan
    hi = float(hi) if hi is not None else np.nan
    return point, lo, hi


def _landmark_series(data, metric_key, split):
    """Serie por landmark de una métrica + su banda de IC 95% bootstrap."""
    lm_dict = data.get(metric_key, {}).get(split, {}).get("landmarks", {})
    ci_dict = data.get("bootstrap_ci", {}).get(split, {})
    lms = sorted(int(k) for k in lm_dict.keys())
    vals, los, his = [], [], []
    for lm in lms:
        v = lm_dict[str(lm)].get(metric_key)
        vals.append(float(v) if v is not None else np.nan)
        ci_lm = ci_dict.get(str(lm), {}).get(metric_key) or [None, None]
        los.append(float(ci_lm[0]) if ci_lm[0] is not None else np.nan)
        his.append(float(ci_lm[1]) if ci_lm[1] is not None else np.nan)
    return np.array(lms), np.array(vals), np.array(los), np.array(his)


def _events_per_landmark(split):
    """Eventos observados por landmark en el split (iguales para todos los modelos)."""
    for data in MODELS_DATA.values():
        lm_dict = data.get("c_index", {}).get(split, {}).get("landmarks", {})
        if lm_dict:
            lms = sorted(int(k) for k in lm_dict.keys())
            return np.array(lms), np.array(
                [lm_dict[str(lm)]["n_events"] for lm in lms], dtype=float
            )
    return np.array([]), np.array([])


# 3. Orden global de modelos: por C-index agrupado en holdout (estilo del
# proyecto: leyendas ordenadas por rendimiento medio en holdout). El color,
# marcador y trazo de cada modelo se fijan por este rango y son consistentes
# en todas las figuras.
SPLIT = "holdout"
model_rank = sorted(
    MODELS_DATA.keys(),
    key=lambda m: (
        -_pooled_point(MODELS_DATA[m], "c_index", SPLIT)
        if np.isfinite(_pooled_point(MODELS_DATA[m], "c_index", SPLIT))
        else float("inf")
    ),
)
MODEL_STYLE = {
    name: {
        "color": PALETTE[i % len(PALETTE)],
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)],
    }
    for i, name in enumerate(model_rank)
}

# 4. Tabla CSV: estimandos agrupados por split, ordenados por holdout
records = []
for model_name in model_rank:
    data = MODELS_DATA[model_name]
    row = {"model": model_name}
    for split_name in ["train", "val", "holdout"]:
        row[split_name] = _pooled_point(data, "c_index", split_name)
    for metric_key in ["auroc", "c_index_ipcw", "brier", "ipa"]:
        row[f"holdout_{metric_key}"] = _pooled_point(data, metric_key, "holdout")
    records.append(row)

df = pd.DataFrame(records)
df = df.sort_values(by="holdout", ascending=False).reset_index(drop=True)
df.to_csv(csv_output, index=False)
print("=== Comparación de Modelos (estimando agrupado por split) ===", flush=True)
print(df.to_string(index=False), flush=True)
print(f"\n-> Tabla guardada en {csv_output}", flush=True)


def _panel_landmark(ax, metric_key, ylabel, hline=None, show_events=False):
    """Draw one curve per model against landmark age, with its 95% bootstrap CI band."""
    if show_events:
        lms_ev, events = _events_per_landmark(SPLIT)
        if len(lms_ev):
            ax2 = ax.twinx()
            ax2.bar(
                lms_ev,
                events,
                width=0.7,
                color="gray",
                alpha=0.25,
                zorder=1,
                label="Eventos (eje dcho.)",
            )
            ax2.set_ylabel("Eventos observados", color="gray")
            ax2.tick_params(axis="y", colors="gray")
            ax2.set_zorder(ax.get_zorder())
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

    for m_name in model_rank:
        lms, vals, los, his = _landmark_series(MODELS_DATA[m_name], metric_key, SPLIT)
        if len(lms) == 0:
            continue
        st = MODEL_STYLE[m_name]
        ax.plot(
            lms,
            vals,
            marker=st["marker"],
            linestyle=st["linestyle"],
            color=st["color"],
            linewidth=2,
            markersize=6,
            label=m_name,
            zorder=3,
        )
        band = np.isfinite(los) & np.isfinite(his)
        if np.any(band):
            ax.fill_between(
                lms[band],
                los[band],
                his[band],
                color=st["color"],
                alpha=0.15,
                zorder=2,
            )

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle=":", linewidth=1, zorder=2)
        if hline == 0.0:
            ymin, ymax = ax.get_ylim()
            if ymin < 0:
                ax.axhspan(ymin, 0, color="red", alpha=0.05, zorder=0)
    ax.set_xlabel("Edad del landmark")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    return ax


def _panel_forest(ax, ax_txt, metric_key, title, fmt, refline):
    """Forest plot of the pooled estimand + 95% cluster-bootstrap CI, one row per model."""
    rows = []
    for m_name in model_rank:
        point, lo, hi = _extract_pooled(MODELS_DATA[m_name], metric_key, SPLIT)
        if np.isfinite(point):
            rows.append((m_name, point, lo, hi))
    rows.sort(key=lambda r: r[1], reverse=True)  # mejor modelo arriba en este panel

    ys = np.arange(len(rows))[::-1]
    for y, (m_name, point, lo, hi) in zip(ys, rows):
        st = MODEL_STYLE[m_name]
        if np.isfinite(lo) and np.isfinite(hi):
            xerr = np.array([[point - lo], [hi - point]])
            ax.errorbar(
                [point],
                [y],
                xerr=xerr,
                fmt=st["marker"],
                color=st["color"],
                markersize=7,
                capsize=4,
                linewidth=1.8,
            )
            ci_txt = f"{point:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"
        else:
            ax.plot(
                [point],
                [y],
                marker=st["marker"],
                color=st["color"],
                markersize=7,
            )
            ci_txt = f"{point:{fmt}}"
        # Las cifras viven en el eje fantasma de la derecha (axis off), no dentro
        # del eje del forest: meterlas dentro obligaría a inflar el xlim y la
        # escala de la métrica quedaría deformada.
        ax_txt.text(0.0, y, m_name, ha="left", va="center", fontsize=9)
        ax_txt.text(
            0.98, y, ci_txt, ha="right", va="center", fontsize=9, family="monospace"
        )

    if refline is not None:
        ax.axvline(refline, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax_txt.set_ylim(ax.get_ylim())
    ax_txt.axis("off")
    return ax


def plot_main_figure(out_stem):
    """Lámina central para paper/memoria: paneles A, B y C en una sola figura."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.25)

    # Panel A — discriminación por landmark: AUROC a 5 años con banda de IC 95%
    # y eje secundario con los eventos observados por landmark (barras grises).
    # Contesta por adelantado por qué la banda se ensancha a edades altas: va con
    # el número de casos, no con inestabilidad del modelo.
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_landmark(ax_a, "auroc", METRIC_LABELS["auroc"], show_events=True)
    ax_a.set_title("A. Discriminación por landmark", loc="left", fontweight="bold")
    ax_a.legend(fontsize=9, loc="best", framealpha=0.9)

    # Panel B — utilidad frente al modelo nulo: IPA a 5 años con línea de
    # referencia en 0 y la zona inferior sombreada.
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_landmark(ax_b, "ipa", METRIC_LABELS["ipa"], hline=0.0)
    ax_b.set_title("B. Utilidad frente al modelo nulo", loc="left", fontweight="bold")

    # Panel C — resumen agregado: forest de cuatro métricas, un modelo por fila,
    # con el punto agrupado, su IC 95% de cluster bootstrap y las cifras exactas.
    gs_c = gs[1, :].subgridspec(1, len(FOREST_SPECS), wspace=0.45)
    for i, (metric_key, title, fmt, refline) in enumerate(FOREST_SPECS):
        sub = gs_c[0, i].subgridspec(1, 2, width_ratios=[2.2, 1.8], wspace=0.05)
        ax_f = fig.add_subplot(sub[0])
        ax_txt = fig.add_subplot(sub[1])
        _panel_forest(ax_f, ax_txt, metric_key, title, fmt, refline)
    fig.text(
        0.01,
        0.47,
        "C. Resumen agregado (estimando pooled + IC 95% cluster bootstrap)",
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.align_ylabels()
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"-> Figura central guardada en {out_stem}.png / {out_stem}.pdf", flush=True)


# 5. comparison.png: C-index por landmark (holdout) con banda de IC 95%
fig, ax = plt.subplots(figsize=(10, 6))
_panel_landmark(ax, "c_index", METRIC_LABELS["c_index"])
ax.set_title("Comparativa de C-index por Landmark (Holdout)", fontsize=14, pad=12)
ax.legend(
    title="Modelos (ordenados por C-index Holdout)",
    fontsize=10,
    title_fontsize=11,
    loc="best",
    framealpha=0.9,
)
fig.tight_layout()
fig.savefig(png_output, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"-> Figura guardada en {png_output}", flush=True)

# 6. PNG por métrica nueva (subproductos no declarados en el mismo directorio)
for metric_key, png_name in METRIC_PNGS.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    hline = (
        0.0
        if metric_key == "ipa"
        else (0.5 if "index" in metric_key or metric_key == "auroc" else None)
    )
    show_events = metric_key == "auroc"
    _panel_landmark(
        ax, metric_key, METRIC_LABELS[metric_key], hline=hline, show_events=show_events
    )
    ax.set_title(
        f"{METRIC_LABELS[metric_key]} por Landmark (Holdout)", fontsize=14, pad=12
    )
    ax.legend(
        title="Modelos (ordenados por C-index Holdout)",
        fontsize=10,
        title_fontsize=11,
        loc="best",
        framealpha=0.9,
    )
    fig.tight_layout()
    out_path = results_dir / png_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> Figura guardada en {out_path}", flush=True)

# 7. Figura central (paneles A, B y C)
main_stem = str(Path(main_png_output).with_suffix(""))
plot_main_figure(main_stem)
# Garantizar que el PDF declarado existe aunque difiera del stem del PNG
if Path(main_pdf_output) != Path(f"{main_stem}.pdf"):
    import shutil

    shutil.copyfile(f"{main_stem}.pdf", main_pdf_output)
