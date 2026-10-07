# Documentación del script `workflow/scripts/build_cohort.py`

Este documento explica, bloque a bloque, cómo funciona el script `build_cohort.py`, qué forma tienen los datos que recibe y cómo construye la cohorte final con las variables de salida `Id`, `duration` y `event`.

## Propósito general

El script implementa el primer paso del preprocesado de la tubería. Su objetivo es transformar ficheros clínicos crudos de un pool concreto (`discovery` o `holdout`) en una cohorte analizable para supervivencia.

La cohorte se define así:

- La entrada temporal de cada paciente se fija en una edad índice (`index_age`) calculada a partir de su fecha de nacimiento.
- El evento de interés es el primer diagnóstico de cáncer de pulmón, identificado por códigos SNOMED.
- La censura o fin de seguimiento se define con la última fecha observada en condiciones u observaciones, o con la fecha de muerte si ocurre antes.

## Qué recibe el script

El script se ejecuta como `snakemake script:`. Eso significa que el objeto `snakemake` no se importa manualmente, sino que Snakemake lo inyecta en tiempo de ejecución.

### Entradas esperadas

El script consume tres tablas crudas y una lista de parámetros:

- `patients_path`: fichero con al menos `Id`, `BIRTHDATE` y `DEATHDATE`.
- `conditions_path`: fichero con al menos `PATIENT`, `START` y `CODE`.
- `observations_path`: fichero con al menos `PATIENT` y `DATE`.
- `index_age`: edad en años a la que se fija la entrada a cohorte.
- `lung_cancer_codes`: lista de códigos SNOMED que identifican cáncer de pulmón.

### Salida esperada

El script escribe un único parquet en `snakemake.output[0]` con estas columnas:

- `Id`
- `duration`
- `event`

## Importaciones y preparación inicial

```python
import sys
from typing import TYPE_CHECKING

import pandas as pd
```

- `sys` se usa para redirigir la salida estándar al log de Snakemake.
- `TYPE_CHECKING` permite declarar el tipo de `snakemake` sin ejecutarlo realmente.
- `pandas` realiza toda la manipulación tabular y temporal.

## Bloque `TYPE_CHECKING`

```python
if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake
```

Este bloque existe solo para el linter y el editor. En ejecución real no corre, porque `TYPE_CHECKING` vale `False`.

Su función es dar autocompletado y tipos al objeto `snakemake`, que Snakemake inyecta cuando lanza el script.

## Función principal `build_cohort`

```python
def build_cohort(
    patients_path, conditions_path, observations_path, index_age, lung_cancer_codes
):
```

Esta función encapsula toda la lógica de construcción de la cohorte. Recibe rutas a los ficheros de entrada y devuelve un `DataFrame` ya filtrado y listo para análisis.

### 1. Normalización de los códigos de cáncer de pulmón

```python
lung_cancer_codes = set(lung_cancer_codes)
```

La lista de códigos se convierte en `set` para acelerar la comprobación de pertenencia con `.isin(...)`.

### 2. Lectura de los ficheros crudos

```python
patients = pd.read_csv(patients_path, usecols=["Id", "BIRTHDATE", "DEATHDATE"])
conditions = pd.read_csv(conditions_path, usecols=["PATIENT", "START", "CODE"])
observations = pd.read_csv(observations_path, usecols=["PATIENT", "DATE"])
```

Solo se leen las columnas necesarias.

- `patients` aporta identidad, fecha de nacimiento y fecha de muerte.
- `conditions` aporta eventos clínicos con fecha de inicio y código diagnóstico.
- `observations` aporta fechas de observaciones clínicas para ampliar el seguimiento.

## Tratamiento de fechas en pacientes

```python
patients["BIRTHDATE"] = pd.to_datetime(
    patients["BIRTHDATE"], errors="coerce"
).dt.tz_localize(None)
patients["DEATHDATE"] = pd.to_datetime(
    patients["DEATHDATE"], errors="coerce"
).dt.tz_localize(None)
patients["index_date"] = patients["BIRTHDATE"] + pd.DateOffset(years=int(index_age))
```

Aquí se hace el primer preprocesado temporal:

- Se convierten las fechas a formato datetime.
- `errors="coerce"` transforma valores inválidos en `NaT`.
- `tz_localize(None)` elimina la zona horaria si existe, dejando fechas naive.
- `index_date` se calcula sumando `index_age` años a `BIRTHDATE`.

Esto fija el instante desde el que se empieza a observar a cada paciente.

## Construcción de la última fecha de encuentro

```python
cond_dates = conditions[["PATIENT", "START"]].rename(
    columns={"PATIENT": "Id", "START": "record_date"}
)
obs_dates = observations[["PATIENT", "DATE"]].rename(
    columns={"PATIENT": "Id", "DATE": "record_date"}
)
```

Se preparan dos tablas homogéneas:

- una con fechas procedentes de `conditions`
- otra con fechas procedentes de `observations`

Ambas se renombran al mismo esquema para poder combinarlas.

```python
cond_dates["record_date"] = pd.to_datetime(
    cond_dates["record_date"], errors="coerce"
).dt.tz_localize(None)
obs_dates["record_date"] = pd.to_datetime(
    obs_dates["record_date"], errors="coerce"
).dt.tz_localize(None)
```

Se repite la homogeneización temporal para evitar mezclar fechas con y sin zona horaria.

```python
last_encounter = (
    pd.concat([cond_dates, obs_dates], ignore_index=True)
    .groupby("Id", as_index=False)["record_date"]
    .max()
    .rename(columns={"record_date": "last_encounter_date"})
)
```

Aquí se calcula la última fecha observada de cada paciente, usando tanto condiciones como observaciones.

Este paso es importante porque la tubería usa esa fecha para:

- exigir un seguimiento mínimo posterior a la edad índice
- definir la censura si no hay evento ni muerte antes

## Identificación del evento de cáncer de pulmón

```python
conditions["CODE"] = conditions["CODE"].astype(str)
lung_conds = conditions[conditions["CODE"].isin(lung_cancer_codes)].copy()
lung_conds["START"] = pd.to_datetime(lung_conds["START"], errors="coerce")
```

Se asegura que `CODE` es texto y luego se filtran solo las condiciones cuyo código pertenece al conjunto de cáncer de pulmón.

```python
first_lung_cancer = (
    lung_conds.groupby("PATIENT", as_index=False)["START"]
    .min()
    .rename(columns={"PATIENT": "Id", "START": "lung_cancer_date"})
)
```

Para cada paciente, se conserva la primera fecha de diagnóstico de cáncer de pulmón.

Este valor será el momento del evento si existe y si ocurre después de la entrada a cohorte.

## Unión de todas las piezas

```python
cohort = patients.merge(last_encounter, on="Id", how="left").merge(
    first_lung_cancer, on="Id", how="left"
)
```

Se combinan en una sola tabla:

- datos demográficos básicos
- última fecha de contacto
- primera fecha de cáncer de pulmón

El `how="left"` mantiene a todos los pacientes cargados desde `patients`, aunque no tengan encuentros o no tengan evento.

## Criterios de exclusión

```python
cohort = cohort.dropna(subset=["BIRTHDATE", "index_date"])
```

Se eliminan pacientes sin fecha de nacimiento válida o sin `index_date` computable.

### 1. El paciente debe estar vivo en la edad índice

```python
cohort = cohort[
    cohort["DEATHDATE"].isna() | (cohort["DEATHDATE"] > cohort["index_date"])
]
```

Si el paciente ya había fallecido antes o en la fecha índice, no se puede incluir en la cohorte.

### 2. No debe haber cáncer de pulmón prevalente

```python
cohort = cohort[
    cohort["lung_cancer_date"].isna()
    | (cohort["lung_cancer_date"] > cohort["index_date"])
]
```

Si el cáncer de pulmón aparece antes o en la fecha índice, el caso es prevalente y se excluye.

### 3. Debe existir seguimiento mínimo tras la entrada

```python
cohort = cohort[cohort["last_encounter_date"] >= cohort["index_date"]]
```

Solo se conservan pacientes con al menos un encuentro en o después de la fecha índice.

Esto evita pacientes para los que no se puede establecer observación posterior a la entrada.

## Cálculo del evento y de la fecha final

```python
cohort["event"] = cohort["lung_cancer_date"].notna().astype("int8")
```

`event` vale `1` si existe fecha de cáncer de pulmón y `0` en caso contrario.

```python
cohort["end_date"] = (
    cohort["lung_cancer_date"]
    .combine_first(cohort["DEATHDATE"])
    .combine_first(cohort["last_encounter_date"])
)
```

Se calcula la fecha de finalización siguiendo esta prioridad:

1. fecha del cáncer de pulmón
2. fecha de muerte
3. última fecha de encuentro

En la práctica, esto define el tiempo observado hasta evento o censura.

## Cálculo de la duración

```python
cohort["duration"] = (
    (cohort["end_date"] - cohort["index_date"]).dt.days / 365.25
).clip(lower=0)
```

La duración se expresa en años desde `index_date` hasta `end_date`.

- Se calcula la diferencia en días.
- Se divide por `365.25` para pasar a años.
- `clip(lower=0)` evita duraciones negativas por inconsistencias de datos o empates temporales.

## Salida final de la función

```python
return cohort[["Id", "duration", "event"]].reset_index(drop=True)
```

La función devuelve solo las variables necesarias para el siguiente paso del flujo analítico:

- `Id`: identificador del paciente
- `duration`: tiempo de seguimiento en años
- `event`: indicador binario de aparición de cáncer de pulmón

## Cuerpo ejecutable del script

```python
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")
```

La salida estándar y de error se redirige al fichero de log de la regla en Snakemake. Así cada ejecución deja trazabilidad separada.

```python
this_cohort = build_cohort(
    snakemake.input.patients,
    snakemake.input.conditions,
    snakemake.input.observations,
    snakemake.params.index_age,
    snakemake.params.lung_cancer_codes,
)
```

Se invoca la función con las rutas y parámetros definidos por la regla Snakemake.

```python
this_cohort.to_parquet(snakemake.output[0], index=False)
```

La cohorte final se guarda en formato Parquet, sin índice.

```python
print(
    f"[{snakemake.wildcards.pool}] {len(this_cohort)} pacientes  "
    f"event_rate={this_cohort['event'].mean():.3f}",
    flush=True,
)
```

Se imprime un resumen mínimo por pool:

- nombre del pool (`discovery` o `holdout`)
- número de pacientes incluidos
- tasa de evento observada

## Resumen del preproceso

En términos prácticos, el script hace esto:

1. Lee pacientes, condiciones y observaciones crudas.
2. Normaliza y homogeneiza fechas.
3. Calcula la edad índice a partir de la fecha de nacimiento.
4. Recupera la última fecha de contacto usando condiciones y observaciones.
5. Detecta el primer cáncer de pulmón por paciente con una lista de códigos SNOMED.
6. Excluye pacientes no válidos para un análisis de supervivencia incidente.
7. Construye `duration` y `event`.
8. Exporta la cohorte final a Parquet.

El resultado es una tabla ya preparada para modelado de supervivencia o análisis longitudinal, con una definición clara de entrada, evento y censura.