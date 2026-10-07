"""Snakemake ``script:`` for build_sequences (Stage 1). SKELETON, tu completas la funcion.

Construye las secuencias irregulares de un split y las escribe en un pickle con el
esquema exacto que ``load_sequences`` lee (notebook celda 5)::

    cache = {
        "sequences": {pid: {"times", "labs", "masks", "sex_male", "codes"}},
        "lab_codes": [...],                       # nombres de las D columnas de labs, en orden
        "cohort":    DataFrame[Id, duration, event],
    }

Las fuentes crudas son Synthea de verdad, no un formato ya recortado: patients trae
BIRTHDATE/GENDER (fechas, no edades), observations trae DATE/CODE(LOINC)/VALUE, y
conditions trae START/CODE(SNOMED CT). Tu conviertes fecha -> edad con BIRTHDATE.

La bolsa ``codes`` son capitulos ICD-10, no SNOMED: el cuerpo del script ya traduce
conditions.CODE (SNOMED) a capitulo ICD-10 con el mapa results/snomed_to_icd10.json, y
filtra lo que no sirve (codigos sin mapa, el desenlace C33/C34 -> anti-fuga, y capitulos
ruido O/Z/V/W/X/Y). A tu funcion ``build_patient_sequence`` le llega cond_pid ya
traducido y filtrado, con una columna ``icd10_parent``; tu solo la recoges por timestep.

La media/desviacion de los labs tampoco se calculan aqui: llegan ya ajustadas (en train)
desde ``results/lab_stats.json``, que produce la regla fit_lab_stats. Este script solo
las aplica. Tu unica funcion pendiente es ``build_patient_sequence``; el resto del
cuerpo (leer entradas, traducir codigos, recorrer pacientes, escribir el pickle) ya
esta escrito, para que veas como se ensambla la salida.

Honra ese esquema y ``RawSequenceDatasetWithICD`` (celda 8) construye sus muestras
sin tocarlas. El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no
se importa: el bloque TYPE_CHECKING de abajo solo le da tipo al linter/IDE).

Reads:  snakemake.input.cohort         cohort de ESTE split (Id, duration, event).
        snakemake.input.lab_stats       results/lab_stats.json (media/desviacion de train).
        snakemake.input.snomed_map      results/snomed_to_icd10.json (SNOMED -> capitulo ICD-10).
        snakemake.input.patients        patients crudo del pool (Id, BIRTHDATE, GENDER).
        snakemake.input.observations    observations crudo del pool (PATIENT, DATE, CODE, VALUE).
        snakemake.input.conditions      conditions crudo del pool (PATIENT, START, CODE).
Writes: snakemake.output[0]            pickle de secuencias de este split.
"""

import json
import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake


def load_sequences(pkl_path):
    """Load a sequence pickle, same signature as the notebook (cell 5).

    Returns ``(sequences, lab_codes, cohort)``. train.py and eval.py reuse this to read
    back what this rule writes, so it stays here as the shared contract.
    """
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    sequences = {pid: seq for pid, seq in cache["sequences"].items() if seq is not None}
    return sequences, list(cache["lab_codes"]), cache["cohort"]


def _to_naive_datetime(values):
    """Convert datetime-like values to naive datetimes to avoid tz mismatches."""
    dt = pd.to_datetime(values, errors="coerce")
    if isinstance(dt, pd.Series):
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(None)
        return dt
    if isinstance(dt, pd.Timestamp):
        if getattr(dt, "tzinfo", None) is not None:
            return dt.tz_convert(None)
        return dt
    return dt


def build_patient_sequence(
    pid, birthdate, gender, obs_pid, cond_pid, lab_codes, lab_mean, lab_std
):
    birthdate = _to_naive_datetime(birthdate)
    code_idx = {code: j for j, code in enumerate(lab_codes)}  # code -> column index

    obs = obs_pid.copy()
    obs["AGE"] = (_to_naive_datetime(obs["DATE"]) - birthdate).dt.days / 365.25
    obs["VAL"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["AGE", "VAL"])
    # TODO: keep only labs in lab_codes, and one value per (AGE, CODE)
    obs = obs[obs["CODE"].isin(lab_codes)]
    obs = (
        obs.groupby(["AGE", "CODE"]).first().reset_index()
    )  # Con esta línea agrupamos las filas que coinciden en (AGE,CODE) y nos quedamos con la primera.

    cond = cond_pid.copy()
    cond["AGE"] = (_to_naive_datetime(cond["START"]) - birthdate).dt.days / 365.25
    cond = cond.dropna(subset=["AGE"])

    if obs.empty and cond.empty:
        return None

    times = np.unique(np.concatenate([obs["AGE"].to_numpy(), cond["AGE"].to_numpy()]))
    age_idx = {
        age: i for i, age in enumerate(times)
    }  # age -> row index (times is already sorted) (np.unique ya ordena)

    labs = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    masks = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    if not obs.empty:
        ti = obs["AGE"].map(age_idx).to_numpy()  # row index for each obs row
        ci = obs["CODE"].map(code_idx).to_numpy()  # column index for each obs row
        # TODO: scatter the z-scored values into `labs`, and 1.0 into `masks`

        # Extraemos y convertimos los valores crudos a numérico
        raw_vals = pd.to_numeric(obs["VALUE"], errors="coerce").to_numpy(
            dtype=np.float32
        )

        # Calculamos la media y desviación estandar correspondientes a cada columna ci
        means = lab_mean[ci]
        stds = np.where(lab_std[ci] == 0.0, 1.0, lab_std[ci])

        # Z-score vectorizado
        z_vals = (raw_vals - means) / stds

        # Asignación vectorizada en las matrices labs y masks
        labs[ti, ci] = z_vals
        masks[ti, ci] = 1.0

    codes = [[] for _ in times]
    # TODO: for each age group in cond, put its icd10_parent list at codes[age_idx[age]]
    if not cond.empty:
        for age, group in cond.groupby("AGE"):
            idx = age_idx[age]
            codes[idx] = group["icd10_parent"].astype(str).tolist()

    return {
        "times": times.astype(np.float32),
        "labs": labs,
        "masks": masks,
        "sex_male": 1.0 if str(gender).upper() == "M" else 0.0,
        "codes": codes,
    }


# ── Cuerpo del script (escrito entero: solo build_patient_sequence queda pendiente) ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): cada regla deja su
# propio rastro en logs/, con los prints y cualquier traceback (util para tu TODO).
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

cohort = pd.read_parquet(snakemake.input.cohort)
lab_codes = list(snakemake.params.lab_codes)
excluded_icd_initials = tuple(snakemake.params.excluded_icd_initials)
outcome_icd_prefixes = tuple(snakemake.params.outcome_icd_prefixes)

# Estadisticas de normalizacion, ya ajustadas en train (regla fit_lab_stats). Este
# script solo las aplica; se reordenan al orden de lab_codes de este split por si acaso.
with open(snakemake.input.lab_stats, encoding="utf-8") as f:
    stats = json.load(f)
stat_idx = {code: i for i, code in enumerate(stats["lab_codes"])}
lab_mean = np.array([stats["mean"][stat_idx[c]] for c in lab_codes], dtype=np.float32)
lab_std = np.array([stats["std"][stat_idx[c]] for c in lab_codes], dtype=np.float32)

# Mapa SNOMED -> capitulo ICD-10 (regla build_icd10_map). Un unico artefacto compartido.
with open(snakemake.input.snomed_map, encoding="utf-8") as f:
    snomed_to_icd10 = json.load(f)

patients = pd.read_csv(snakemake.input.patients, usecols=["Id", "BIRTHDATE", "GENDER"])
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
patients = patients.set_index("Id")

observations = pd.read_csv(
    snakemake.input.observations, usecols=["PATIENT", "DATE", "CODE", "VALUE"]
)

# Traduce cada diagnostico a capitulo ICD-10 y filtra, todo vectorizado y una sola vez:
#   - sin mapa (icd10_parent == "")  -> fuera.
#   - desenlace outcome_icd_prefixes (C33/C34)  -> fuera: es el resultado a predecir,
#     dejarlo entrar como feature seria una fuga directa.
#   - capitulos ruido excluded_icd_initials (O/Z/V/W/X/Y: embarazo, factores, causas
#     externas)  -> fuera.
conditions = pd.read_csv(
    snakemake.input.conditions, usecols=["PATIENT", "START", "CODE"]
)
conditions["icd10_parent"] = (
    conditions["CODE"].astype(str).map(snomed_to_icd10).fillna("")
)
conditions = conditions[conditions["icd10_parent"] != ""]
conditions = conditions[
    ~conditions["icd10_parent"].str.startswith(outcome_icd_prefixes)
]
conditions = conditions[~conditions["icd10_parent"].str[0].isin(excluded_icd_initials)]

obs_by_pid = dict(tuple(observations.groupby("PATIENT")))
cond_by_pid = dict(tuple(conditions.groupby("PATIENT")))
empty_obs = observations.iloc[0:0]
empty_cond = conditions.iloc[0:0]

sequences = {}
for pid in cohort["Id"]:
    if pid not in patients.index:
        continue
    seq = build_patient_sequence(
        pid,
        patients.loc[pid, "BIRTHDATE"],
        patients.loc[pid, "GENDER"],
        obs_by_pid.get(pid, empty_obs),
        cond_by_pid.get(pid, empty_cond),
        lab_codes,
        lab_mean,
        lab_std,
    )
    if seq is not None:
        sequences[str(pid)] = seq

cache = {"sequences": sequences, "lab_codes": lab_codes, "cohort": cohort}
with open(snakemake.output[0], "wb") as f:
    pickle.dump(cache, f)
print(
    f"[{snakemake.wildcards.split}] {len(sequences)} secuencias -> {snakemake.output[0]}",
    flush=True,
)
