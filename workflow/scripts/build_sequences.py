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
    """Build the irregular sequence for one patient.

    Args:
        pid: patient id.
        birthdate: fecha de nacimiento del paciente (Timestamp), para pasar de
            fecha de observacion/diagnostico a edad.
        gender: "M" o "F", crudo de patients.
        obs_pid: this patient's lab rows (DATE, CODE, VALUE).
        cond_pid: this patient's diagnosis rows (START, icd10_parent), YA traducidas a
            capitulo ICD-10 y filtradas (sin ruido ni desenlace) por el cuerpo del script.
        lab_codes: ordered lab codes that become the D columns of ``labs``.
        lab_mean, lab_std: per-column statistics from fit_lab_stats (train only).

    Returns:
        A dict ``{"times", "labs", "masks", "sex_male", "codes"}``, or ``None`` if the
        patient has no observations.
    """
    # TODO. Construye la secuencia del paciente:
    #   - Convierte las fechas de obs_pid (DATE) y cond_pid (START) a edad:
    #     (fecha - birthdate).dt.days / 365.25.
    #   - Ordena las observaciones por edad. Agrupa por edad: cada edad unica es un
    #     timestep, y `times` (T,) son esas edades.
    #   - Monta `labs` (T, D) y `masks` (T, D), una columna por cada codigo de
    #     lab_codes. masks vale 1 donde el lab se midio y 0 donde no; labs lleva el
    #     valor z-scoreado (con lab_mean/lab_std) donde masks es 1 y 0 donde es 0.
    #   - `codes` es una lista de listas: por timestep, los capitulos ICD-10 (strings)
    #     de cond_pid["icd10_parent"] en esa edad (ya vienen traducidos y filtrados).
    #   - `sex_male` es 1.0 si gender == "M", 0.0 en otro caso.

    birthdate_naive = _to_naive_datetime(birthdate)
    obs_pid["AGE"] = (
        _to_naive_datetime(obs_pid["DATE"]) - birthdate_naive
    ).dt.days / 365.25
    cond_pid["AGE"] = (
        _to_naive_datetime(cond_pid["START"]) - birthdate_naive
    ).dt.days / 365.25

    obs_pid = obs_pid.dropna(subset=["AGE"])
    cond_pid = cond_pid.dropna(subset=["AGE"])

    if obs_pid.empty and cond_pid.empty:
        return None

    obs_by_age = {age: group for age, group in obs_pid.groupby("AGE", sort=True)}
    cond_by_age = {age: group for age, group in cond_pid.groupby("AGE", sort=True)}

    times = np.sort(
        np.unique(
            np.concatenate(
                [obs_pid["AGE"].to_numpy(), cond_pid["AGE"].to_numpy()]
            )
        )
    )

    labs = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    masks = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    codes = []

    for i, age in enumerate(times):
        age_key = float(age)
        obs_at_age = obs_by_age.get(age_key, pd.DataFrame(columns=obs_pid.columns))
        for j, code in enumerate(lab_codes):
            values = pd.to_numeric(
                obs_at_age.loc[obs_at_age["CODE"] == code, "VALUE"],
                errors="coerce",
            ).dropna()
            if not values.empty:
                std = float(lab_std[j])
                if std == 0.0:
                    std = 1.0
                labs[i, j] = (float(values.iloc[0]) - float(lab_mean[j])) / std
                masks[i, j] = 1.0

        cond_at_age = cond_by_age.get(age_key, pd.DataFrame(columns=cond_pid.columns))
        codes.append(cond_at_age["icd10_parent"].astype(str).tolist())

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
