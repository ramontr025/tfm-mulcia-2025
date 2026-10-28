"""Snakemake ``script:`` del analisis exploratorio (Stage 1.5).

Describe la cohorte (§4.1 de la memoria) y las secuencias (§4.2) con el protocolo
declarado en ``MemoriaTFM/docs/03_metodos.md``: seis bloques, tres vistas de la
historia y un artefacto con esquema y procedencia.

Estructura:
    results/eda/eda_summary.json          esquema 2.0, procedencia y bloques
    results/eda/tables/*.csv              siete tablas en formato largo
    results/eda/figures/fig_*.{png,pdf}   seis figuras

Reads:  snakemake.input.sequences  results/{train,val,holdout}/sequences.pkl
        snakemake.input.cohorts    results/{discovery,train,val,holdout}/cohort.parquet
        snakemake.input.lab_stats  results/lab_stats.json
        snakemake.input.labels     resources/loinc_labels.csv
Writes: snakemake.output[0]        directory("results/eda")
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from eda import blocks, io, plots, views  # noqa: E402

sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")  # noqa: F821

STARTED = time.time()
SPLITS = ("train", "val", "holdout")
COHORT_SPLITS = ("discovery", "train", "val", "holdout")

OUT = Path(snakemake.output[0])  # noqa: F821
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def param(name: str, default=None):
    """Parametro de Snakemake con valor por defecto."""
    return getattr(snakemake.params, name, default)  # noqa: F821


# ── 1. Parametros ───────────────────────────────────────────────────────────

landmarks = [int(x) for x in param("landmarks")]
index_age = float(param("index_age", 18.0))
horizon = float(param("horizon", 5.0))
max_events = int(param("max_events", 200))
age_weight_gamma = float(param("age_weight_gamma", 1.0))
seed = int(param("seed", 0))
dpi = int(param("dpi", plots.DPI))
width_cm = float(param("figure_width_cm", plots.FIGURE_WIDTH_CM))
week_days = float(param("week_modulus_days", 7.0))
week_tol = float(param("day_multiple_tolerance", 0.1))
annual_center = float(param("annual_mode_center_days", 371.0))
jaccard_threshold = float(param("jaccard_flag_threshold", 0.95))
sigma_limit = float(param("value_sigma_limit", 5.0))
mean_tol = float(param("labstats_z_mean_tol", 0.01))
std_tol = float(param("labstats_z_std_tol", 0.01))
bands = [(str(b[0]), float(b[1]), float(b[2])) for b in param("age_bands")]

plots.DPI = dpi
plots.FIGURE_WIDTH_CM = width_cm
np.random.seed(seed)
plots.apply_common_style()

# ── 2. Entradas ─────────────────────────────────────────────────────────────

seq_paths = dict(zip(SPLITS, list(snakemake.input.sequences)))  # noqa: F821
cohort_paths = dict(zip(COHORT_SPLITS, list(snakemake.input.cohorts)))  # noqa: F821

history_by_split: dict[str, dict] = {}
lab_codes: list[str] = []
cohorts: dict[str, pd.DataFrame] = {}
for split in SPLITS:
    sequences, codes, cohort = views.load_history(Path(seq_paths[split]))
    history_by_split[split] = sequences
    lab_codes = codes
    cohorts[split] = cohort
for split in COHORT_SPLITS:
    if split in cohorts:
        continue
    cohorts[split] = pd.read_parquet(cohort_paths[split])

labels = io.read_labels(Path(snakemake.input.labels))  # noqa: F821
with open(snakemake.input.lab_stats, encoding="utf-8") as handle:  # noqa: F821
    lab_stats = json.load(handle)

print(
    f"[eda] train: {len(history_by_split['train'])} secuencias, "
    f"{len(lab_codes)} analíticas; cohorts: "
    f"{ {k: len(v) for k, v in cohorts.items()} }",
    flush=True,
)

# ── 3. Vistas ───────────────────────────────────────────────────────────────

train_sequences = history_by_split["train"]
dataset = views.build_model_view(
    train_sequences,
    cohorts["train"],
    landmarks,
    index_age,
    max_events,
    age_weight_gamma,
)
samples = views.sample_table(dataset)
history = views.history_table(train_sequences)
print(
    f"[eda] muestra del modelo: {len(samples)} pares, "
    f"{samples['pid'].nunique()} pacientes",
    flush=True,
)

# ── 4. Bloques ──────────────────────────────────────────────────────────────

rows_cohort, block_c = blocks.block_cohort(
    cohorts, history_by_split, index_age, landmarks
)
flow_rows = views.funnel(cohorts, train_sequences, dataset)

coverage_rows, block_1 = blocks.block_model_view(samples, landmarks)
coverage = pd.DataFrame(coverage_rows)

timing_rows, block_2, deltas_days, rates_by_event = blocks.block_timing(
    history,
    samples,
    cohorts["train"],
    index_age,
    horizon,
    week_days,
    week_tol,
    annual_center,
)

density_rows, block_3 = blocks.block_density(train_sequences, lab_codes, labels, bands)

redundancy_rows, block_4, jaccard = blocks.block_redundancy(
    train_sequences, lab_codes, labels, jaccard_threshold
)

value_rows, block_5, values_by_lab = blocks.block_values(
    train_sequences, lab_stats, lab_codes, labels, sigma_limit, mean_tol, std_tol
)

print(
    f"[eda] bloques: sin analítica {block_1['pct_samples_without_labs']:.1f} % · "
    f"Δt mediana {block_2['n_consecutive_timestep_pairs']} pares · "
    f"Jaccard {block_4['n_flagged']} pares marcados",
    flush=True,
)

# ── 5. Figuras ──────────────────────────────────────────────────────────────

panels: dict[str, list[dict]] = {}
panels["C"] = plots.plot_cohort(
    block_c["km"],
    [45, 50, 55, 60, 65, 70],
    FIGURES / "fig_cohort.png",
    FIGURES / "fig_cohort.pdf",
)
panels["1"] = plots.plot_sample_coverage(
    coverage, FIGURES / "fig_sample_coverage.png", FIGURES / "fig_sample_coverage.pdf"
)
panels["2"] = plots.plot_time_structure(
    deltas_days,
    rates_by_event,
    float(block_2["modal_delta_days"] or 0.0),
    block_2["informative_sampling"]["mannwhitney_p"],
    block_2["informative_sampling"]["cliffs_delta"],
    FIGURES / "fig_time_structure.png",
    FIGURES / "fig_time_structure.pdf",
)

labs_per_visit = np.concatenate(
    [
        np.asarray(seq["masks"], dtype=np.float32).sum(axis=1)[
            np.asarray(seq["masks"], dtype=np.float32).sum(axis=1) > 0
        ]
        for seq in train_sequences.values()
    ]
)
panels["3"] = plots.plot_observation_density(
    labs_per_visit,
    pd.DataFrame(density_rows),
    labels,
    lab_codes,
    block_3["bands"],
    FIGURES / "fig_observation_density.png",
    FIGURES / "fig_observation_density.pdf",
)
panels["4"] = plots.plot_lab_redundancy(
    jaccard,
    block_4["order"],
    block_4["n_flagged"],
    jaccard_threshold,
    labels,
    FIGURES / "fig_lab_redundancy.png",
    FIGURES / "fig_lab_redundancy.pdf",
)
panels["5"] = plots.plot_value_scale(
    values_by_lab,
    lab_codes,
    labels,
    sigma_limit,
    FIGURES / "fig_value_scale.png",
    FIGURES / "fig_value_scale.pdf",
)
print("[eda] seis figuras escritas", flush=True)

# ── 6. Tablas y JSON ────────────────────────────────────────────────────────

io.write_csv(rows_cohort, TABLES / "cohort_summary.csv")
io.write_csv(
    flow_rows,
    TABLES / "sample_flow.csv",
    [
        "stage_order",
        "stage",
        "unit",
        "n",
        "pct_of_previous",
        "note",
        "population",
    ],
)
io.write_csv(coverage_rows, TABLES / "coverage_by_landmark.csv")
io.write_csv(timing_rows, TABLES / "visits_timing.csv")
io.write_csv(density_rows, TABLES / "lab_coverage.csv")
io.write_csv(
    redundancy_rows,
    TABLES / "lab_redundancy_pairs.csv",
    ["lab_a", "lab_b", "jaccard", "flagged_ge_threshold"],
)
io.write_csv(value_rows, TABLES / "lab_value_stats.csv")

for key, block in (
    ("C", block_c),
    ("1", block_1),
    ("2", block_2),
    ("3", block_3),
    ("4", block_4),
    ("5", block_5),
):
    block["figure"] = {
        "name": io.FIGURE_NAMES[key],
        "panels": panels.get(key, []),
    }

runtime = time.time() - STARTED
provenance = io.build_provenance(
    params={
        "landmarks": landmarks,
        "index_age": index_age,
        "horizon": horizon,
        "max_events": max_events,
        "age_weight_gamma": age_weight_gamma,
        "seed": seed,
        "dpi": dpi,
        "figure_width_cm": width_cm,
        "week_modulus_days": week_days,
        "day_multiple_tolerance": week_tol,
        "annual_mode_center_days": annual_center,
        "jaccard_flag_threshold": jaccard_threshold,
        "value_sigma_limit": sigma_limit,
        "labstats_z_mean_tol": mean_tol,
        "labstats_z_std_tol": std_tol,
        "days_per_year": blocks.DAYS_PER_YEAR,
    },
    inputs={
        "n_sequences_by_split": {k: len(v) for k, v in history_by_split.items()},
        "n_cohort_by_split": {k: len(v) for k, v in cohorts.items()},
        "lab_dim": len(lab_codes),
        "n_landmarks": len(landmarks),
        "input_hashes": {
            Path(p).as_posix(): io.sha256_file(Path(p))
            for p in [
                *seq_paths.values(),
                *cohort_paths.values(),
                snakemake.input.lab_stats,  # noqa: F821
                snakemake.input.labels,  # noqa: F821
            ]
        },
    },
    runtime_seconds=runtime,
    tool_versions={
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "lifelines": __import__("lifelines").__version__,
    },
)

summary = {
    "schema_version": io.SCHEMA_VERSION,
    "provenance": provenance,
    "lab_codes": lab_codes,
    "blocks": {
        "C": {**block_c, "flow": flow_rows},
        "1": block_1,
        "2": block_2,
        "3": block_3,
        "4": block_4,
        "5": block_5,
    },
}
io.write_json(summary, OUT / "eda_summary.json")
print(f"[eda] DONE en {runtime:.1f} s -> {OUT}", flush=True)
