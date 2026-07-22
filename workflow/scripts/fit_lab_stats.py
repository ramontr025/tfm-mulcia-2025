"""Snakemake ``script:`` for fit_lab_stats (Stage 1). SKELETON, tu completas la funcion.

Ajusta la media y la desviacion de cada lab usando SOLO los pacientes de train, y
escribe un unico artefacto ``results/lab_stats.json`` que los tres ``build_sequences``
(train, val, holdout) comparten. Ese es el punto: la normalizacion se aprende en train
y val/holdout la REUTILIZAN, exactamente igual que un escalador se ajusta en train y se
aplica tal cual al resto. Al vivir en su propia regla, esa disciplina es un nodo del
DAG y no un recalculo escondido dentro de cada trabajo.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no se importa: el
bloque TYPE_CHECKING de abajo solo le da tipo al linter/IDE, nunca se ejecuta).

Reads:  snakemake.input.train_cohort    cohorte de train (Id, ...); define quien es train.
        snakemake.input.observations    observations crudo de discovery (PATIENT, CODE, VALUE).
Writes: snakemake.output[0]             results/lab_stats.json  {"lab_codes", "mean", "std"}.
"""

import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake


def fit_lab_stats(train_ids, observations, lab_codes):
    """Calcula media y desviacion por lab, SOLO sobre los pacientes de train.

    Args:
        train_ids: set de Id de pacientes de train.
        observations: DataFrame crudo de observations de discovery, SIN filtrar por
            split (columns: PATIENT, CODE, VALUE).
        lab_codes: codigos de lab a normalizar, en el orden de la config.

    Returns:
        (mean, std): dos arrays (D,) alineados con lab_codes. Si un lab no aparece
        nunca en train, usa mean=0.0, std=1.0 para no dividir por cero.
    """
    # TODO. Filtra observations a los pacientes de train_ids, quedate con las filas
    # cuyo CODE este en lab_codes, y saca por cada codigo la media y la desviacion de
    # VALUE (numerico). Devuelve los dos arrays en el orden de lab_codes.
    raise NotImplementedError("fit_lab_stats: pendiente (issue 1)")


# ── Cuerpo del script ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): cada regla deja su
# propio rastro en logs/, con los prints y cualquier traceback.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

train_cohort = pd.read_parquet(snakemake.input.train_cohort)
observations = pd.read_csv(
    snakemake.input.observations, usecols=["PATIENT", "CODE", "VALUE"]
)
lab_codes = list(snakemake.params.lab_codes)

mean, std = fit_lab_stats(set(train_cohort["Id"]), observations, lab_codes)

stats = {
    "lab_codes": lab_codes,
    "mean": np.asarray(mean).tolist(),
    "std": np.asarray(std).tolist(),
}
with open(snakemake.output[0], "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(
    f"[lab_stats] {len(lab_codes)} labs, ajustadas en {len(train_cohort)} pacientes de train",
    flush=True,
)
