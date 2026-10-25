"""Figuras del EDA: una funcion por figura, un mensaje por figura.

Convenciones (C7): ancho de diseno 12,8 cm, fuentes >= 8 pt, etiquetas en castellano
con tildes, cada analitica rotulada ``Nombre corto (LOINC)`` y nunca solo el codigo.
Cada figura guarda PNG a 300 dpi y PDF vectorial, y declara sus paneles para que el
verificador pueda rechazar datos repetidos con unidades distintas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURE_WIDTH_CM = 12.8
FIGURE_HEIGHT_CM = 8.0
DPI = 300

COLORS = {
    "train": "#0072B2",
    "val": "#009E73",
    "holdout": "#D55E00",
    "discovery": "#CC79A7",
    "event0": "#0072B2",
    "event1": "#D55E00",
    "neutral": "#4D4D4D",
    "grid": "#DDDDDD",
}

GROUP_COLORS = {
    "Signos vitales": "#0072B2",
    "Antropometría": "#56B4E9",
    "Cuestionario": "#E69F00",
    "Hemograma": "#009E73",
    "Lípidos": "#CC79A7",
}


def apply_common_style() -> None:
    """Unica llamada de estilo. Se ejecuta una vez, al principio del script."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def _cm(value: float) -> float:
    return value / 2.54


def _save(fig: plt.Figure, png: Path, pdf: Path) -> None:
    fig.tight_layout(pad=0.6, h_pad=1.4, w_pad=0.8)
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)


def _short(labels: dict[str, dict[str, str]], code: str) -> str:
    meta = labels.get(code, {})
    name = meta.get("lab_short_es") or code
    return f"{name} ({code})"


# ── F0: cohorte y supervivencia ─────────────────────────────────────────────


def plot_cohort(
    km: dict, landmarks_ticks: list[int], png: Path, pdf: Path
) -> list[dict]:
    """Incidencia acumulada por edad, con el primer y el ultimo evento marcados.

    El eje x es la edad al final del seguimiento (nacimiento -> evento o censura),
    no el tiempo desde la fecha indice. Se recorta cinco años antes del primer
    evento y cinco despues del ultimo, y las dos edades se marcan con linea
    discontinua. La banda sombreada marca la ventana de landmarks.

    El numero en riesgo ya no se dibuja: vive en el artefacto
    (``blocks.C.km[split].at_risk``) y lo comprueba ``check_eda.py``.
    """
    ages_min = [d["event_age_min"] for d in km.values() if d.get("event_age_min")]
    ages_max = [d["event_age_max"] for d in km.values() if d.get("event_age_max")]
    first = min(ages_min) if ages_min else float(min(km["train"]["grid_age_years"]))
    last = max(ages_max) if ages_max else float(max(km["train"]["grid_age_years"]))

    fig, ax = plt.subplots(figsize=(_cm(FIGURE_WIDTH_CM), _cm(7.5)))
    ax.axvspan(
        min(landmarks_ticks),
        max(landmarks_ticks),
        color="#999999",
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )
    for split in ("train", "val", "holdout"):
        if split not in km:
            continue
        data = km[split]
        grid = np.asarray(data["grid_age_years"])
        ax.plot(
            grid,
            100.0 * (1.0 - np.asarray(data["survival"])),
            color=COLORS[split],
            lw=1.4,
            label=split,
            zorder=3,
        )
        ax.fill_between(
            grid,
            100.0 * (1.0 - np.asarray(data["ci_upper"])),
            100.0 * (1.0 - np.asarray(data["ci_lower"])),
            color=COLORS[split],
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )

    ax.axvline(first, color="#555555", ls="--", lw=0.9, zorder=4)
    ax.axvline(last, color="#555555", ls="--", lw=0.9, zorder=4)

    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("% de pacientes diagnosticados")
    ax.set_title("Incidencia acumulada de cáncer de pulmón por edad")
    ax.set_xlim(
        first - 5.0,
        min(last + 5.0, float(max(km["train"]["grid_age_years"]))),
    )
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, loc="upper left")
    for age in (first, last):
        ax.text(
            age,
            0.25,
            f"{age:.1f}".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#555555",
        )
    ax.text(
        max(landmarks_ticks) - 1.0,
        9.4,
        "ventana de landmarks",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#555555",
    )
    _save(fig, png, pdf)
    return [
        {
            "id": "acumulada",
            "dataset": "km_survival",
            "unit": "porcentaje acumulado",
        },
    ]


# ── F1: lo que ve el modelo en cada corte ───────────────────────────────────


def plot_sample_coverage(coverage: pd.DataFrame, png: Path, pdf: Path) -> list[dict]:
    """Muestras por corte, proporcion sin analítica y magnitudes por muestra."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(_cm(FIGURE_WIDTH_CM), _cm(14.5)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2]},
    )
    lm = coverage["landmark_age"].to_numpy()
    ax = axes[0]
    ax.bar(lm, coverage["n_samples"], color=COLORS["train"], width=0.8)
    ax.set_ylabel("Muestras")
    ax.set_title("Muestra paciente-landmark por edad de corte", loc="left")

    ax = axes[1]
    ax.plot(
        lm,
        coverage["pct_without_labs"],
        color=COLORS["event1"],
        marker="o",
        ms=3,
        lw=1.4,
    )
    ax.set_ylabel("% sin analítica")
    ax.set_ylim(0, 100)
    ax.set_title("Muestras sin ninguna analítica previa al corte", loc="left")
    first = coverage.iloc[0]
    last = coverage.iloc[-1]
    ax.annotate(
        f"{first['pct_without_labs']:.1f} % a los {int(first['landmark_age'])} años",
        xy=(first["landmark_age"], first["pct_without_labs"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )
    ax.annotate(
        f"{last['pct_without_labs']:.1f} % a los {int(last['landmark_age'])} años",
        xy=(last["landmark_age"], last["pct_without_labs"]),
        xytext=(-70, -12),
        textcoords="offset points",
        fontsize=7,
    )

    ax = axes[2]
    ax.plot(
        lm,
        coverage["median_labs_if_any"],
        color=COLORS["train"],
        lw=1.4,
        label="mediciones observadas, si hay alguna",
    )
    ax.fill_between(
        lm,
        coverage["p25_labs_if_any"],
        coverage["p75_labs_if_any"],
        color=COLORS["train"],
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        lm,
        coverage["median_icd_codes"],
        color=COLORS["event1"],
        lw=1.4,
        label="códigos ICD-10-CM",
    )
    ax.set_ylabel("Mediana por muestra")
    ax.set_xlabel("Edad del corte (años)")
    ax.set_title("Cuando hay datos: analíticas y códigos disponibles", loc="left")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, png, pdf)
    return [
        {"id": "muestras", "dataset": "n_samples", "unit": "muestra"},
        {"id": "sin_analitica", "dataset": "pct_without_labs", "unit": "porcentaje"},
        {
            "id": "magnitudes",
            "dataset": "labs_and_codes_per_sample",
            "unit": "recuento por muestra",
        },
    ]


# ── F2: estructura temporal ─────────────────────────────────────────────────


def plot_time_structure(
    deltas_days: np.ndarray,
    rates_by_event: dict[int, np.ndarray],
    modal_days: float,
    p_value: float | None,
    delta_cliff: float | None,
    png: Path,
    pdf: Path,
) -> list[dict]:
    """Distribución de Δt en escala log y tasa de visitas por grupo de evento."""
    fig, axes = plt.subplots(
        1, 2, figsize=(_cm(FIGURE_WIDTH_CM), _cm(9.0)), gridspec_kw={"wspace": 0.35}
    )
    ax = axes[0]
    positive = deltas_days[deltas_days > 0.5]
    bins = np.logspace(np.log10(0.5), np.log10(max(positive.max(), 10)), 60)
    ax.hist(positive, bins=bins, color=COLORS["train"], edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Días entre timesteps (escala log)")
    ax.set_ylabel("N de pares")
    ax.set_title("Espaciado entre timesteps", loc="left")
    ax.axvline(7, color=COLORS["neutral"], ls=":", lw=0.8)
    ax.axvline(modal_days, color=COLORS["event1"], ls="--", lw=1.0)
    ax.annotate(
        "7 d", xy=(7, ax.get_ylim()[1] * 0.92), fontsize=7, color=COLORS["neutral"]
    )
    ax.annotate(
        f"{modal_days:.0f} d",
        xy=(modal_days, ax.get_ylim()[1] * 0.78),
        fontsize=7,
        color=COLORS["event1"],
    )

    ax = axes[1]
    groups = [(0, "sin evento"), (1, "con evento")]
    data = [rates_by_event.get(key, np.zeros(0)) for key, _ in groups]
    ax.boxplot(
        [d for d in data],
        vert=True,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#EEEEEE", "edgecolor": COLORS["neutral"]},
        medianprops={"color": COLORS["neutral"], "lw": 1.2},
    )
    for i, (values, color) in enumerate(
        zip(data, [COLORS["event0"], COLORS["event1"]]), start=1
    ):
        if values.size:
            jitter = np.random.default_rng(0).normal(
                i, 0.04, size=min(values.size, 800)
            )
            ax.scatter(
                jitter,
                values[: jitter.size],
                s=3,
                color=color,
                alpha=0.35,
                linewidths=0,
            )
    ax.set_xticks(
        [1, 2], [f"{label}\n(n={d.size})" for (_, label), d in zip(groups, data)]
    )
    ax.set_ylabel("Visitas por año de seguimiento")
    ax.set_title("Medición según desenlace", loc="left")
    caption = []
    if delta_cliff is not None:
        caption.append(f"δ de Cliff = {delta_cliff:+.2f}")
    if p_value is not None:
        caption.append(f"p = {p_value:.1e}")
    if caption:
        ax.text(
            0.02,
            0.98,
            "   ".join(caption),
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    _save(fig, png, pdf)
    return [
        {"id": "delta", "dataset": "delta_days", "unit": "días"},
        {"id": "tasas", "dataset": "lab_visits_per_year", "unit": "visitas por año"},
    ]


# ── F3: densidad de observación ─────────────────────────────────────────────


def plot_observation_density(
    labs_per_visit: np.ndarray,
    lab_coverage: pd.DataFrame,
    labels: dict[str, dict[str, str]],
    lab_codes: list[str],
    bands: list[str],
    png: Path,
    pdf: Path,
) -> list[dict]:
    """Analíticas por visita, cobertura por analítica e intensidad por edad."""
    fig = plt.figure(figsize=(_cm(FIGURE_WIDTH_CM), _cm(16.5)))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 2.6, 1.4])

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(
        labs_per_visit,
        bins=np.arange(0.5, 24.5, 1),
        color=COLORS["train"],
        edgecolor="none",
    )
    ax.set_xlabel("Analíticas medidas en la visita")
    ax.set_ylabel("N de visitas")
    ax.set_title("Carga por visita", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    med = float(np.median(labs_per_visit)) if labs_per_visit.size else 0.0
    ax.axis("off")
    ax.text(
        0.0,
        0.8,
        f"Mediana: {med:.0f} de {len(lab_codes)} analíticas\n"
        f"Media: {np.mean(labs_per_visit):.1f}\n"
        f"Visitas con analítica: {labs_per_visit.size}",
        va="top",
        fontsize=8,
    )

    ax = fig.add_subplot(gs[1, :])
    lab_rows = lab_coverage[lab_coverage["scope"] == "lab"]
    pct_pat = (
        lab_rows[lab_rows["metric"] == "pct_patients_ever_measured"]
        .set_index("lab_code")["value"]
        .astype(float)
    )
    pct_vis = (
        lab_rows[lab_rows["metric"] == "pct_visits_with_lab"]
        .set_index("lab_code")["value"]
        .astype(float)
    )
    order = list(reversed(lab_codes))
    y = np.arange(len(order))
    ax.hlines(
        y, pct_vis[order].to_numpy(), pct_pat[order].to_numpy(), color="#BBBBBB", lw=1.0
    )
    ax.scatter(
        pct_vis[order], y, s=18, color=COLORS["train"], label="% de visitas", zorder=3
    )
    ax.scatter(
        pct_pat[order],
        y,
        s=18,
        color=COLORS["event1"],
        label="% de pacientes",
        zorder=3,
    )
    ax.set_yticks(y, [_short(labels, c) for c in order], fontsize=6)
    ax.set_xlabel("Porcentaje")
    ax.set_xlim(0, 105)
    ax.set_title("Presencia de cada analítica", loc="left")
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
    )

    ax = fig.add_subplot(gs[2, :])
    pivot = (
        lab_coverage[lab_coverage["metric"] == "observations_per_patient_year"]
        .pivot(index="lab_code", columns="age_band", values="value")
        .reindex(lab_codes)[bands]
        .T
    )
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(lab_codes)),
        [_short(labels, c) for c in lab_codes],
        rotation=90,
        fontsize=5,
    )
    ax.set_yticks(range(len(bands)), bands, fontsize=6)
    ax.grid(False)
    ax.set_title("Observaciones por paciente y año, por banda de edad", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    _save(fig, png, pdf)
    return [
        {"id": "carga", "dataset": "labs_per_visit", "unit": "analíticas por visita"},
        {"id": "presencia", "dataset": "lab_coverage", "unit": "porcentaje"},
        {
            "id": "intensidad",
            "dataset": "observations_per_patient_year",
            "unit": "observaciones por paciente-año",
        },
    ]


# ── F4: redundancia ─────────────────────────────────────────────────────────


def plot_lab_redundancy(
    jaccard: np.ndarray,
    order: list[str],
    n_flagged: int,
    threshold: float,
    labels: dict[str, dict[str, str]],
    png: Path,
    pdf: Path,
) -> list[dict]:
    """Mapa de calor de Jaccard, ya reordenado por bloques en ``order``."""
    codes = list(order)
    fig, ax = plt.subplots(figsize=(_cm(FIGURE_WIDTH_CM), _cm(11.5)))
    im = ax.imshow(jaccard, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(
        range(len(codes)), [_short(labels, c) for c in codes], rotation=90, fontsize=6
    )
    ax.set_yticks(range(len(codes)), [_short(labels, c) for c in codes], fontsize=6)
    ax.grid(False)
    ax.set_title(
        f"Jaccard por visita entre analíticas ({n_flagged} pares ≥ {threshold:.2f})"
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Jaccard")
    _save(fig, png, pdf)
    return [{"id": "jaccard", "dataset": "jaccard_matrix", "unit": "índice"}]


# ── F5: magnitudes ──────────────────────────────────────────────────────────


def plot_value_scale(
    values_by_lab: dict[str, np.ndarray],
    lab_codes: list[str],
    labels: dict[str, dict[str, str]],
    sigma_limit: float,
    png: Path,
    pdf: Path,
) -> list[dict]:
    """Distribución de z-scores observados por analítica."""
    codes = list(reversed(lab_codes))
    data = [values_by_lab.get(code, np.zeros(0)) for code in codes]
    fig, ax = plt.subplots(figsize=(_cm(FIGURE_WIDTH_CM), _cm(15.0)))
    positions = np.arange(1, len(codes) + 1)
    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        whis=(1, 99),
        boxprops={"facecolor": "#DDDDDD", "edgecolor": COLORS["neutral"]},
        medianprops={"color": COLORS["neutral"]},
    )
    for patch, code in zip(bp["boxes"], codes):
        group = labels.get(code, {}).get("group", "")
        patch.set_facecolor(GROUP_COLORS.get(group, "#DDDDDD"))
        patch.set_alpha(0.8)
    ax.axvline(0, color=COLORS["neutral"], lw=0.8)
    for limit in (-sigma_limit, sigma_limit):
        ax.axvline(limit, color=COLORS["event1"], ls=":", lw=0.8)
    ax.set_yticks(positions, [_short(labels, c) for c in codes], fontsize=6)
    ax.set_xlabel("Valor estandarizado (z)  ·  bigotes p1-p99, límites ±5")
    ax.set_title("Distribución de los valores que recibe el modelo", loc="left")
    _save(fig, png, pdf)
    return [{"id": "valores", "dataset": "observed_z", "unit": "z"}]
