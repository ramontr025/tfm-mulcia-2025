# Documentación del script `workflow/scripts/build_sequences.py`

Este documento explica, bloque a bloque, cómo funciona el script `build_sequences.py`, qué datos recibe y cómo construye las secuencias temporales irregulares para cada paciente en la cohorte.

## Propósito general

El script implementa una parte fundamental del preprocesado de la etapa 1. Su objetivo es transformar las fuentes clínicas crudas (observaciones de laboratorio y diagnósticos de condiciones) en secuencias temporales irregulares z-scoreadas por paciente.

La secuencia resultante para cada paciente se representa como un diccionario con el siguiente esquema:

```python
{
    "times": np.ndarray,     # Array 1D (T,) con la unión de edades (en años) de laboratorios y diagnósticos
    "labs": np.ndarray,      # Matriz 2D (T, D) con los valores de laboratorio z-scoreados
    "masks": np.ndarray,     # Matriz 2D (T, D) binaria (1.0 si se midió el lab, 0.0 si no)
    "sex_male": float,       # 1.0 si el género es masculino ("M"), 0.0 en otro caso
    "codes": list[list[str]] # Lista de listas (T,): capítulos ICD-10 presentes en cada edad
}
```

El script procesa todos los pacientes del split correspondiente y empaqueta el conjunto en un fichero `pickle` junto con la lista de códigos de laboratorio `lab_codes` y la tabla `cohort`.

## Qué recibe el script

El script se ejecuta como `snakemake script:`. El objeto `snakemake` no se importa manualmente: Snakemake lo inyecta en tiempo de ejecución.

### Entradas esperadas

- `snakemake.input.cohort`: Fichero Parquet con la cohorte del split (`Id`, `duration`, `event`).
- `snakemake.input.lab_stats`: Fichero JSON (`results/lab_stats.json`) generado en *train* con la media (`mean`) y desviación estándar (`std`) de cada prueba analítica.
- `snakemake.input.snomed_map`: Fichero JSON (`results/snomed_to_icd10.json`) con el mapa traducido `SNOMED -> capítulo ICD-10`.
- `snakemake.input.patients`: Fichero CSV crudo de pacientes (`Id`, `BIRTHDATE`, `GENDER`).
- `snakemake.input.observations`: Fichero CSV crudo de observaciones (`PATIENT`, `DATE`, `CODE`, `VALUE`).
- `snakemake.input.conditions`: Fichero CSV crudo de condiciones/diagnósticos (`PATIENT`, `START`, `CODE`).
- `snakemake.params.lab_codes`: Lista ordenada de códigos LOINC/laboratorio a incluir (define las $D$ columnas de `labs`).
- `snakemake.params.excluded_icd_initials`: Tupla de iniciales de capítulos ICD-10 a excluir por ruido (ej. `O`, `Z`, `V`, `W`, `X`, `Y`).
- `snakemake.params.outcome_icd_prefixes`: Tupla de prefijos de diagnóstico del desenlace a excluir (ej. `C33`, `C34`) para evitar fugas de información (*data leakage*).

### Salida esperada

- `snakemake.output[0]`: Fichero `pickle` que almacena un diccionario con el caché completo de secuencias del split.

## Importaciones y preparación inicial

```python
import json
import pickle
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
```

- `json` y `pickle` permiten leer los ficheros de configuración/estadísticas y escribir las secuencias finales.
- `sys` permite redirigir los logs de salida.
- `numpy` y `pandas` realizan toda la manipulación vectorial, temporal y matricial.

## Bloque `TYPE_CHECKING`

```python
if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake
```

Este bloque existe exclusivamente para el linter y el entorno de desarrollo (IDE). No se ejecuta en runtime.

## Función `load_sequences`

```python
def load_sequences(pkl_path):
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    sequences = {pid: seq for pid, seq in cache["sequences"].items() if seq is not None}
    return sequences, list(cache["lab_codes"]), cache["cohort"]
```

Esta función sirve como contrato compartido para leer los ficheros `pickle` generados. Es reutilizada posteriormente por los scripts de entrenamiento y evaluación (`train.py` y `eval.py`).

## Función `_to_naive_datetime`

```python
def _to_naive_datetime(values):
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
```

Normaliza cualquier objeto de fecha/hora convirtiéndolo a un formato *naive* (sin zona horaria) para evitar errores de tipo en restas temporales.

## Función principal `build_patient_sequence` (Vectorizada)

```python
def build_patient_sequence(pid, birthdate, gender, obs_pid, cond_pid, lab_codes, lab_mean, lab_std):
```

Esta función encapsula la construcción vectorizada de la secuencia irregular de un paciente individual.

### 1. Cálculo de edades y limpieza previa

```python
    birthdate = _to_naive_datetime(birthdate)
    code_idx = {code: j for j, code in enumerate(lab_codes)}

    obs = obs_pid.copy()
    obs["AGE"] = (_to_naive_datetime(obs["DATE"]) - birthdate).dt.days / 365.25
    obs["VAL"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["AGE", "VAL"])
    obs = obs[obs["CODE"].isin(lab_codes)]
    obs = obs.groupby(['AGE', 'CODE']).first().reset_index()
```

- Convierte la fecha de nacimiento a datetime naive.
- Calcula la edad exacta (en años) de cada observación de laboratorio restando `birthdate`.
- Filtra solo las analíticas contenidas en `lab_codes`.
- Aplica `groupby(['AGE', 'CODE']).first().reset_index()` para eliminar duplicados y conservar la primera medición registrada para una misma edad y tipo de laboratorio.

### 2. Procesado de diagnósticos

```python
    cond = cond_pid.copy()
    cond["AGE"] = (_to_naive_datetime(cond["START"]) - birthdate).dt.days / 365.25
    cond = cond.dropna(subset=["AGE"])

    if obs.empty and cond.empty:
        return None
```

- Calcula la edad en la que se diagnosticó cada condición.
- Si el paciente no posee ni observaciones ni diagnósticos válidos, se descarta devolviendo `None`.

### 3. Obtención de timesteps únicos y mapeo de índices

```python
    times = np.unique(np.concatenate([obs["AGE"].to_numpy(), cond["AGE"].to_numpy()]))
    age_idx = {age: i for i, age in enumerate(times)}
```

- `np.unique` combina todas las edades registradas tanto de observaciones analíticas (`obs["AGE"]`) como de diagnósticos (`cond["AGE"]`), elimina duplicados y **ordena de menor a mayor** la lista completa de edades. Por tanto, `times` contiene la unión de todos los instantes temporales en los que ocurrió al menos un evento (sea análisis o diagnóstico).
- `age_idx` crea un mapeo rápido de cada edad única a su índice de fila `i` en la matriz temporal ($0 \dots T-1$).

> [!NOTE]
> **Comportamiento por timestep según el evento ocurrido en esa edad (`AGE`):**
> - **Solo laboratorio:** `labs` y `masks` contendrán los valores z-scoreados y la máscara de las analíticas de esa edad; `codes` contendrá una lista vacía `[]`.
> - **Solo diagnóstico:** `labs` y `masks` serán ceros (`0.0`); `codes` contendrá la lista de capítulos ICD-10 diagnosticados en esa edad.
> - **Coincidencia (laboratorio + diagnóstico en la misma edad):** conviven en la misma fila $i$ del timestep (`labs`/`masks` con las mediciones y `codes` con los diagnósticos).

### 4. Construcción vectorizada de `labs` y `masks`

```python
    labs = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    masks = np.zeros((len(times), len(lab_codes)), dtype=np.float32)
    if not obs.empty:
        ti = obs["AGE"].map(age_idx).to_numpy()
        ci = obs["CODE"].map(code_idx).to_numpy()

        raw_vals = pd.to_numeric(obs["VALUE"], errors="coerce").to_numpy(dtype=np.float32)
        means = lab_mean[ci]
        stds = np.where(lab_std[ci] == 0.0, 1.0, lab_std[ci])
        z_vals = (raw_vals - means) / stds
        
        labs[ti, ci] = z_vals
        masks[ti, ci] = 1.0
```

- `ti` contiene los índices de fila (timesteps) de cada observación.
- `ci` contiene los índices de columna (tipo de laboratorio) de cada observación.
- `z_vals` estandariza los valores crudos restando la media `lab_mean` y dividiendo por la desviación estándar `lab_std` del laboratorio correspondiente.
- Con indexación avanzada `labs[ti, ci] = z_vals` y `masks[ti, ci] = 1.0`, se rellenan las matrices de golpe sin necesidad de bucles manuales.

### 5. Empaquetado de diagnósticos (`codes`) y formato final

```python
    codes = [[] for _ in times]
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
```

- `codes` se rellena asociando a cada índice de edad `age_idx[age]` la lista de capítulos ICD-10 diagnosticados en ese timestep.
- La función devuelve el diccionario formateado con tipos de datos de precisión simple (`float32`).

## Cuerpo ejecutable del script

```python
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")
```

Redirige todos los `print` y posibles mensajes de error al archivo de log de Snakemake correspondiente.

### Carga y alineación de estadísticas

```python
cohort = pd.read_parquet(snakemake.input.cohort)
lab_codes = list(snakemake.params.lab_codes)
excluded_icd_initials = tuple(snakemake.params.excluded_icd_initials)
outcome_icd_prefixes = tuple(snakemake.params.outcome_icd_prefixes)

with open(snakemake.input.lab_stats, encoding="utf-8") as f:
    stats = json.load(f)
stat_idx = {code: i for i, code in enumerate(stats["lab_codes"])}
lab_mean = np.array([stats["mean"][stat_idx[c]] for c in lab_codes], dtype=np.float32)
lab_std = np.array([stats["std"][stat_idx[c]] for c in lab_codes], dtype=np.float32)
```

Alinea los vectores de media y desviación estándar de entrenamiento al orden exacto de `lab_codes` configurado para este split.

### Filtrado y traducción global de condiciones

```python
with open(snakemake.input.snomed_map, encoding="utf-8") as f:
    snomed_to_icd10 = json.load(f)

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
```

De forma global y vectorizada (una sola vez para todo el fichero):
1. Mapea códigos SNOMED a capítulos ICD-10.
2. Descarta condiciones sin traducción.
3. Descarta diagnósticos del desenlace (`C33`/`C34`) para evitar fugas de información (*data leakage*).
4. Elimina capítulos ruidosos o no clínicamente relevantes (`O`, `Z`, `V`, `W`, `X`, `Y`).

### Bucle de empaquetado por paciente y exportación

```python
obs_by_pid = dict(tuple(observations.groupby("PATIENT")))
cond_by_pid = dict(tuple(conditions.groupby("PATIENT")))

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
```

Agrupa observaciones y condiciones por paciente, invoca `build_patient_sequence` para cada paciente de la cohorte y guarda el diccionario de caché en la ruta de salida `pickle`.

## Resumen del preproceso

En términos prácticos, este script:

1. Lee la cohorte, las observaciones, las condiciones y los parámetros de normalización.
2. Traduce y filtra globalmente todos los diagnósticos a capítulos ICD-10 limpios.
3. Transforma fechas en edades relativas al nacimiento.
4. Construye de forma vectorizada las matrices `labs` (z-scoreadas) y `masks` para cada paciente.
5. Agrupa los diagnósticos ICD-10 por timestep.
6. Exporta un fichero `pickle` estructurado listo para ser consumido directamente por los Datasets de PyTorch.
