"""Snakemake ``script:`` for compare (Stage 3).

Agrega las métricas JSON de todos los modelos evaluados y genera:
1. results/comparison.csv: Tabla con el C-index medio por split (train, val, holdout)
   para cada modelo, ordenada de forma descendente por el C-index en holdout.
2. results/comparison.png: Gráfico de C-index por edad de landmark (holdout),
   con una curva por modelo (incluyendo modelos Cox y GRU-D), en la línea de la
   celda 25 del notebook de referencia.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecución.

Reads:  snakemake.input.metrics   lista de JSONs de métricas (results/metrics/{name}.json)
Writes: snakemake.output.csv      results/comparison.csv
        snakemake.output.png      results/comparison.png
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

# 2. Leer y parsear cada JSON de métricas
records = []
model_landmarks_data = {}

for fpath in metric_files:
    fpath = Path(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = data.get("model_name", fpath.stem)
    c_index_data = data.get("c_index", {})

    train_c_index = c_index_data.get("train", {}).get("mean_c_index", np.nan)
    val_c_index = c_index_data.get("val", {}).get("mean_c_index", np.nan)
    holdout_c_index = c_index_data.get("holdout", {}).get("mean_c_index", np.nan)

    records.append(
        {
            "model": model_name,
            "train": train_c_index,
            "val": val_c_index,
            "holdout": holdout_c_index,
        }
    )

    # Extraer C-index por landmark en holdout (fallback a val si holdout no estuviera disponible)
    split_for_landmarks = "holdout" if "holdout" in c_index_data else "val"
    lm_dict = c_index_data.get(split_for_landmarks, {}).get("landmarks", {})
    lms = sorted([int(k) for k in lm_dict.keys()])
    cis = [lm_dict[str(k)]["c_index"] for k in lms]

    model_landmarks_data[model_name] = {
        "landmarks": lms,
        "c_indices": cis,
        "holdout_c_index": holdout_c_index,
    }

# 3. Construir DataFrame y ordenar por C-index en holdout descendente
df = pd.DataFrame(records)
df = df.sort_values(by="holdout", ascending=False).reset_index(drop=True)

# Guardar tabla CSV
df.to_csv(csv_output, index=False)
print("=== Comparación de Modelos (C-index medio por split) ===", flush=True)
print(df.to_string(index=False), flush=True)
print(f"\n-> Tabla guardada en {csv_output}", flush=True)

# 4. Generar figura de C-index por landmark
# Ordenar modelos por C-index en holdout para la leyenda y visualización
sorted_model_names = df["model"].tolist()

markers = ["o", "s", "^", "v", "D", "p", "*", "X"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

plt.figure(figsize=(10, 6))

for idx, m_name in enumerate(sorted_model_names):
    if m_name not in model_landmarks_data:
        continue
    m_info = model_landmarks_data[m_name]
    lms = m_info["landmarks"]
    cis = m_info["c_indices"]
    mean_holdout = m_info["holdout_c_index"]

    marker = markers[idx % len(markers)]
    linestyle = linestyles[idx % len(linestyles)]
    label = (
        f"{m_name} (Holdout: {mean_holdout:.4f})"
        if not np.isnan(mean_holdout)
        else m_name
    )

    plt.plot(
        lms,
        cis,
        marker=marker,
        linestyle=linestyle,
        linewidth=2,
        markersize=6,
        label=label,
    )

plt.title("Comparativa de C-index por Landmark (Holdout)", fontsize=14, pad=12)
plt.xlabel("Edad del Landmark", fontsize=12)
plt.ylabel("C-index", fontsize=12)
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend(
    title="Modelos (ordenados por C-index Holdout)",
    fontsize=10,
    title_fontsize=11,
    loc="best",
    framealpha=0.9,
)
plt.tight_layout()

# Guardar figura PNG
plt.savefig(png_output, dpi=300, bbox_inches="tight")
plt.close()
print(f"-> Figura guardada en {png_output}", flush=True)
