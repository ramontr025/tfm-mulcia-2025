"""Snakemake ``script:`` de la regla seeds: varianza entre réplicas.

Lee los JSON de métricas de cada modelo de la config (ejecución principal, réplica
``splits.seed``) y de sus réplicas ``{name}_s{seed}``, cada una con su partición de
discovery y su semilla de entrenamiento, y escribe:
1. results/seeds/runs.csv: una fila por ejecución con los estimandos agrupados
   en holdout y su IC 95% de cluster bootstrap.
2. results/seeds/summary.csv: por modelo y métrica, media, desviación típica,
   mínimo y máximo entre ejecuciones.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecución.

Reads:  snakemake.input.metrics  results/metrics/{name}_s{seed}.json
        snakemake.input.base     results/metrics/{name}.json
Writes: snakemake.output.runs    results/seeds/runs.csv
        snakemake.output.summary results/seeds/summary.csv
"""

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

METRICS = ["auroc", "ipa", "c_index_ipcw", "brier", "c_index"]
SPLIT = "holdout"

# 1. Una fila por ejecución
rows = []
for fpath in [*snakemake.input.base, *snakemake.input.metrics]:
    fpath = Path(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    match = re.fullmatch(r"(.+)_s(\d+)", fpath.stem)
    model, seed = (
        (match.group(1), int(match.group(2)))
        if match
        else (fpath.stem, int(snakemake.params.main_seed))
    )
    row = {"model": model, "seed": seed}
    for key in METRICS:
        row[key] = data.get(key, {}).get(SPLIT, {}).get(f"mean_{key}")
        ci = data.get("pooled_ci", {}).get(SPLIT, {}).get(key) or [None, None]
        row[f"{key}_lo"], row[f"{key}_hi"] = ci[0], ci[1]
    rows.append(row)

runs = pd.DataFrame(rows).sort_values(["model", "seed"])
runs.to_csv(snakemake.output.runs, index=False)

# 2. Resumen por modelo: media, desviación, mínimo y máximo entre ejecuciones
summary = (
    runs.melt(id_vars=["model", "seed"], value_vars=METRICS, var_name="metric")
    .groupby(["model", "metric"])["value"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
)
summary.to_csv(snakemake.output.summary, index=False)

print(runs[["model", "seed", *METRICS]].to_string(index=False), flush=True)
print(summary.to_string(index=False), flush=True)
