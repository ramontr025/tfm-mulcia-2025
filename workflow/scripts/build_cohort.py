"""Snakemake ``script:`` for build_cohort (Stage 1).

Deriva la cohorte (Id, duration, event) a partir de patients/conditions/observations
crudos: BIRTHDATE fija la entrada a la cohorte (index_age), un codigo SNOMED de
cancer de pulmon en conditions fija el evento, y conditions+observations juntos
fijan el ultimo encuentro (para exigir seguimiento minimo y como fecha de censura
cuando no hay evento ni muerte registrada).

Corre una vez por pool (discovery y holdout) via el comodin {pool}: EL MISMO codigo
para los dos, asi se ve que reciben identico preprocesamiento. No reparte nada: el
reparto de discovery en train/val es un paso aparte (split_cohort.py).

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no se importa: el
bloque TYPE_CHECKING de abajo solo le da tipo al linter/IDE, nunca se ejecuta).

Reads:  snakemake.input.{patients,conditions,observations}   crudos del pool {pool}.
Writes: snakemake.output[0]                                  results/{pool}/cohort.parquet.
"""

import sys
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake


def build_cohort(
    patients_path, conditions_path, observations_path, index_age, lung_cancer_codes
):
    """Deriva Id/duration/event para todos los pacientes de un pool crudo."""
    lung_cancer_codes = set(lung_cancer_codes)

    patients = pd.read_csv(patients_path, usecols=["Id", "BIRTHDATE", "DEATHDATE"])
    conditions = pd.read_csv(conditions_path, usecols=["PATIENT", "START", "CODE"])
    observations = pd.read_csv(observations_path, usecols=["PATIENT", "DATE"])

    # observations.DATE trae sufijo "Z" (UTC, tz-aware); conditions.START no. Se homogeneizan a
    # naive con tz_localize(None), si no el concat de mas abajo mezcla dtypes y groupby().max() falla.
    patients["BIRTHDATE"] = pd.to_datetime(
        patients["BIRTHDATE"], errors="coerce"
    ).dt.tz_localize(None)
    patients["DEATHDATE"] = pd.to_datetime(
        patients["DEATHDATE"], errors="coerce"
    ).dt.tz_localize(None)
    patients["index_date"] = patients["BIRTHDATE"] + pd.DateOffset(years=int(index_age))

    cond_dates = conditions[["PATIENT", "START"]].rename(
        columns={"PATIENT": "Id", "START": "record_date"}
    )
    obs_dates = observations[["PATIENT", "DATE"]].rename(
        columns={"PATIENT": "Id", "DATE": "record_date"}
    )
    cond_dates["record_date"] = pd.to_datetime(
        cond_dates["record_date"], errors="coerce"
    ).dt.tz_localize(None)
    obs_dates["record_date"] = pd.to_datetime(
        obs_dates["record_date"], errors="coerce"
    ).dt.tz_localize(None)
    last_encounter = (
        pd.concat([cond_dates, obs_dates], ignore_index=True)
        .groupby("Id", as_index=False)["record_date"]
        .max()
        .rename(columns={"record_date": "last_encounter_date"})
    )

    conditions["CODE"] = conditions["CODE"].astype(str)
    lung_conds = conditions[conditions["CODE"].isin(lung_cancer_codes)].copy()
    lung_conds["START"] = pd.to_datetime(lung_conds["START"], errors="coerce")
    first_lung_cancer = (
        lung_conds.groupby("PATIENT", as_index=False)["START"]
        .min()
        .rename(columns={"PATIENT": "Id", "START": "lung_cancer_date"})
    )

    cohort = patients.merge(last_encounter, on="Id", how="left").merge(
        first_lung_cancer, on="Id", how="left"
    )

    # Criterios de exclusion, cada uno quita un grupo que no se puede analizar:
    cohort = cohort.dropna(subset=["BIRTHDATE", "index_date"])
    # 1. Vivo en index_age
    cohort = cohort[
        cohort["DEATHDATE"].isna() | (cohort["DEATHDATE"] > cohort["index_date"])
    ]
    # 2. Sin cancer de pulmon prevalente (ya diagnosticado antes de index_age)
    cohort = cohort[
        cohort["lung_cancer_date"].isna()
        | (cohort["lung_cancer_date"] > cohort["index_date"])
    ]
    # 3. Al menos un encuentro en o despues de index_date (seguimiento minimo)
    cohort = cohort[cohort["last_encounter_date"] >= cohort["index_date"]]

    cohort["event"] = cohort["lung_cancer_date"].notna().astype("int8")
    cohort["end_date"] = (
        cohort["lung_cancer_date"]
        .combine_first(cohort["DEATHDATE"])
        .combine_first(cohort["last_encounter_date"])
    )
    cohort["duration"] = (
        (cohort["end_date"] - cohort["index_date"]).dt.days / 365.25
    ).clip(lower=0)

    return cohort[["Id", "duration", "event"]].reset_index(drop=True)


# ── Cuerpo del script ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): asi cada regla deja
# su propio rastro en logs/, y los prints y cualquier traceback quedan ahi, no mezclados.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

this_cohort = build_cohort(
    snakemake.input.patients,
    snakemake.input.conditions,
    snakemake.input.observations,
    snakemake.params.index_age,
    snakemake.params.lung_cancer_codes,
)
this_cohort.to_parquet(snakemake.output[0], index=False)
print(
    f"[{snakemake.wildcards.pool}] {len(this_cohort)} pacientes  "
    f"event_rate={this_cohort['event'].mean():.3f}",
    flush=True,
)
