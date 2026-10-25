# Documentación de la etapa 1.5: análisis exploratorio (EDA)

Este documento explica qué hace la regla `eda`, cómo construye sus vistas, qué mide en cada
bloque, qué ficheros produce y cómo se comprueba. Sustituye al EDA anterior (una prueba de
concepto de un solo fichero) y fija el contrato del artefacto que consume el capítulo 4 de la
memoria.

Las conclusiones del análisis, con sus cifras, están en `MemoriaTFM/docs/03_metodos.md` §7.
Aquí se documenta la implementación.

## Propósito

La regla describe dos objetos antes de que ningún modelo los use:

- la **cohorte** (§4.1 de la memoria): quién entra, cuánto se sigue, cuánto se pierde por el
  camino y cuál es la incidencia acumulada;
- las **secuencias** (§4.2): cuántas visitas tiene cada paciente, cómo se espacian, qué se mide
  en cada visita, qué analíticas aparecen juntas y si la frecuencia de medición depende del
  desenlace.

Trabaja sobre los mismos artefactos que consumen los modelos (`results/{split}/sequences.pkl`,
`results/{split}/cohort.parquet`) y no reimplementa el truncado por landmark ni el
`max_events`: la vista del modelo sale de `RawSequenceDatasetWithICD`.

## Por qué se sustituyó el EDA anterior

1. `sparsity_visits.png` dibujaba dos veces el mismo Δt (años y días); con el eje lineal, la
   distribución quedaba aplastada y el modo de 371 días aparecía como una barra suelta sin
   contexto.
2. `trajectories.png` tenía 115 ejes de 2,4 pulgadas con el código LOINC de título;
   `global_sparsity_heatmap.png` tenía 9 917 filas y funcionaba como código de barras.
3. El contraste de muestreo informativo se hacía sobre pares de visitas (208 333), no sobre
   pacientes, y el p-valor resultante no era interpretable.
4. El EDA llamaba «visita» a un timestep con analítica, pero en la secuencia un timestep puede
   ser solo un diagnóstico (25,7 % en train).
5. No había ninguna cifra sobre la muestra paciente-landmark, que es lo que ve el modelo.

## Entradas

| Entrada | Fichero | Uso |
|---|---|---|
| `sequences` | `results/{train,val,holdout}/sequences.pkl` | Historia completa; el sexo de cada split sale de aquí |
| `cohorts` | `results/{discovery,train,val,holdout}/cohort.parquet` | Cohortes (`Id`, `duration`, `event`) |
| `lab_stats` | `results/lab_stats.json` | Medias y desviaciones crudas de train, para el contraste del bloque 5 |
| `labels` | `resources/loinc_labels.csv` | Código, nombre corto, nombre completo y grupo de las 23 analíticas |
| `dataset` | `workflow/scripts/dataset.py` | Se declara como entrada para invalidar la salida si cambia el dataset |
| `modules` | `workflow/scripts/eda/*.py` | Lista explícita: un módulo nuevo se añade aquí |

`data/` **no** se declara como entrada: no está en el árbol de trabajo y añadirlo rompería el
dryrun en cualquier checkout sin los crudos. `appendix/data/` tampoco se lee, porque no es un
artefacto del DAG.

## Parámetros

Todos salen de `config/config.yaml`, bloque `eda:`.

| Parámetro | Valor | Qué fija |
|---|---|---|
| `seed` | 0 | Semilla única (jitter de las figuras) |
| `dpi` | 300 | Resolución del PNG |
| `figure_width_cm` | 12.8 | Ancho de diseño (80 % de una caja de texto de 16 cm) |
| `week_modulus_days` | 7 | Módulo de la rejilla semanal |
| `day_multiple_tolerance` | 0.1 | Tolerancia (días) para considerar un Δt múltiplo de 7 |
| `annual_mode_center_days` | 371 | Centro del modo anual (53 semanas) |
| `jaccard_flag_threshold` | 0.95 | Umbral para marcar parejas redundantes |
| `value_sigma_limit` | 5.0 | Límite de \|z\| para contar valores extremos |
| `labstats_z_mean_tol` | 0.01 | Tolerancia de la media z frente a 0 |
| `labstats_z_std_tol` | 0.01 | Tolerancia de la desviación z frente a 1 |
| `age_bands` | 18-39, 40-49, 50-59, 60-69, 70+ | Bandas de la intensidad por edad |
| `max_seq_len` | 200 | `max_events` del dataset (mismos valores que `train`) |
| `age_weight_gamma` | 1.0 | Peso por edad del dataset (mismo valor que `train`) |

`LANDMARKS` (45 a 70) y `index_age` (18) vienen de `cohort` y de la configuración global.
`horizon` (5 años) también.

## Las tres vistas

| Vista | Origen | Uso |
|---|---|---|
| **V1, historia completa** | `load_sequences()` | Bloques 2, 3, 4 y 5: cada paciente entra una vez |
| **V2, historia en el corte** | `RawSequenceDatasetWithICD.samples[i]` | Timesteps disponibles antes de cada landmark |
| **V3, muestra del modelo** | las mismas muestras más `codes_idx`, `landmark_age`, `pid`, `duration` | Bloque 1 y cifras de muestra del bloque C |

La regla anti-pseudo-réplica es explícita: una visita aparece hasta 26 veces en V3, así que
todo lo que razona sobre espaciado, densidad, redundancia y magnitudes se calcula sobre V1. V3
se usa solo donde la pregunta es literalmente «qué ve el modelo».

`views.build_model_view()` es la única puerta a `RawSequenceDatasetWithICD`. El vocabulario
`code_to_idx` se replica de `train.py` (sorted → índice + 1), documentado en
`views.build_code_to_idx()`.

## Los seis bloques

### C. Cohorte y supervivencia (§4.1)

- Tabla por split (`discovery`, `train`, `val`, `holdout`): pacientes, eventos, tasa de evento,
  seguimiento mediano, edad mediana al final (`index_age + duration`) y porcentaje de mujeres
  (desde `sex_male` del pickle).
- Embudo: cohorte de discovery → train y val → secuencias de train → pacientes con al menos un
  par válido → pares paciente-landmark. La etapa cruda del pool se escribe como `null`: `data/`
  no está en el árbol.
- Kaplan-Meier por split (`lifelines.KaplanMeierFitter`) con intervalo al 95 %. Se ajusta sobre
  la **edad al final del seguimiento** (`duration + index_age`), no sobre el tiempo desde la fecha
  índice, y se dibuja en una rejilla de 18 a 90 años. La figura recorta el eje cinco años antes
  del primer evento y cinco después del último, y marca esas dos edades con línea discontinua.
  El número en riesgo en los 26 landmarks se calcula y se guarda en el JSON
  (`blocks.C.km[split].at_risk`), pero no se dibuja: la figura es solo la curva. Discovery no
  lleva curva: es la unión de train y val y duplicaría la figura.
- `discovery` solo aparece en el embudo y en la tabla; las curvas son train, val y holdout.

### 1. Lo que ve el modelo en cada corte

Por landmark (26 filas):

- `n_samples`, `n_without_labs`, `pct_without_labs`;
- mediana y cuartiles de analíticas observadas por muestra (incluida la mediana **condicional**
  a tener alguna);
- mediana y p90 de códigos ICD por muestra;
- mediana y cuartiles de timesteps por muestra.

La población es la muestra del modelo: 202 672 pares y 9 917 pacientes en train.

### 2. Estructura temporal y muestreo informativo

- Visitas con analítica y timesteps por paciente (mediana, cuartiles, p99, máximo) sobre V1.
- Δt en días entre **timesteps consecutivos** de la historia completa. Es la unidad que fija la
  identidad `318 780 − 12 069 = 306 711`: 318 780 timesteps de train menos una diferencia por
  secuencia. No son pares de visitas con analítica (eso daría del orden de 208 000 pares, la
  cifra del EDA anterior).
- Estadísticos del Δt: mediana, cuartiles, p99, máximo, proporción a menos de 0,1 día de un
  múltiplo de 7, moda en días redondeados, recuento del modo anual y proporción por encima del
  horizonte de 5 años.
- **Muestreo informativo por paciente**: `visitas con analítica / duration`, con `duration` de la
  cohorte y sobre los 9 917 pacientes del modelo. Mann-Whitney sobre las tasas y δ de Cliff
  derivado del mismo estadístico U (`δ = 2U₁/(n₀n₁) − 1`), sin comparaciones O(n²). El convenio
  de signo se guarda en el JSON: positivo significa que el grupo con evento se mide más.

### 3. Densidad de observación

- Analíticas por visita (mediana, cuartiles, mínimo, máximo).
- Por analítica: porcentaje de pacientes que la tienen al menos una vez y porcentaje de visitas
  en que aparece. Son dos columnas distintas con unidades distintas en el nombre, porque el EDA
  anterior las confundía.
- Intensidad por banda de edad: observaciones por paciente-año, con los años de exposición
  calculados como solapamiento entre la ventana de cada paciente y la banda.

### 4. Redundancia

- Matriz de Jaccard por visita sobre la matriz aplanada `(visitas × analíticas)`. Nunca una
  matriz de pacientes.
- Reordenamiento jerárquico con `scipy.cluster.hierarchy` (linkage promedio sobre `1 − Jaccard`);
  el orden se guarda en el JSON y la matriz reordenada también.
- Listado de las 253 parejas con su Jaccard y la marca `flagged_ge_threshold`.

### 5. Magnitudes y normalización

- Por analítica: mediana, media, desviación, p1, p99, mínimo, máximo y porcentaje de valores con
  \|z\| > 5, sobre los valores observados (máscara > 0).
- Contraste en dos pasos con `results/lab_stats.json`: (a) el conjunto y el orden de `lab_codes`
  coinciden con los del pipeline; (b) la media de los z observados está a ±0,01 de 0 y la
  desviación a ±0,01 de 1.
- Las analíticas que se salen se **marcan, no se corrigen**. En la ejecución de referencia salen
  cuatro, todas por debajo del 1,5 % de desviación.

## Salidas

```
results/eda/
├── eda_summary.json              schema_version 2.0, provenance y bloques C, 1-5
├── figures/
│   ├── fig_cohort.{png,pdf}
│   ├── fig_sample_coverage.{png,pdf}
│   ├── fig_time_structure.{png,pdf}
│   ├── fig_observation_density.{png,pdf}
│   ├── fig_lab_redundancy.{png,pdf}
│   └── fig_value_scale.{png,pdf}
└── tables/
    ├── cohort_summary.csv
    ├── sample_flow.csv
    ├── coverage_by_landmark.csv
    ├── visits_timing.csv
    ├── lab_coverage.csv
    ├── lab_redundancy_pairs.csv
    └── lab_value_stats.csv
```

El JSON lleva `provenance` con fecha, commit, parámetros usados, dimensiones de entrada,
`runtime_seconds`, versiones de las bibliotecas y `input_hashes` (SHA-256 de cada entrada
declarada). Las cifras del texto de la memoria deben salir de aquí, no de la prosa.

Los CSV van en formato largo: cada fila es una métrica con su unidad, su población (o su split)
y su denominador. Los nombres de campo incluyen la unidad (`median_delta_days_between_consecutive_timesteps`,
`median_lab_visits_per_year_of_followup`).

## Cómo se ejecuta

```powershell
# Reconstrucción limpia (results/eda es una salida directory(): sin borrar, Snakemake no rehace)
Remove-Item -Recurse -Force results\eda
pixi run snakemake results/eda -c1 --rerun-triggers mtime

# Verificación completa
pixi run python workflow/scripts/checks/check_eda.py --only all

# Copia de las figuras a la memoria (PNG y PDF)
pixi run figs
```

`--rerun-triggers mtime` evita que un cambio en `config/config.yaml` reejecute por procedencia
las reglas de cohorte y de secuencias. El EDA no necesita regenerar sus entradas.

En la ejecución de referencia la regla tarda unos 26 s en un equipo local sin GPU.

## Verificación

`workflow/scripts/checks/check_eda.py` no es un framework de tests: es un script plano que lee
el artefacto y comprueba contrato, inventario y cifras. Agrupa las comprobaciones y crece con las
tareas:

| Grupo | Qué comprueba |
|---|---|
| `views` | `schema_version`, bloques exactos, procedencia, hashes, embudo y seis figuras declaradas |
| `cohort` | Tabla frente a la publicada, tres curvas, número en riesgo = muestras por corte, KM frente a un producto-límite recalculado en numpy (1e-9) |
| `coverage` | Muestras, pacientes, proporción sin analítica y medianas de timesteps |
| `timing` | Identidad 318 780 − 12 069 = 306 711, modos, horizonte y contraste por paciente |
| `coverage_labs` | 23 analíticas, separación de los dos porcentajes, filas de visita sin código |
| `redundancy` | 253 parejas, matriz 23×23 simétrica con diagonal 1, forma declarada |
| `values` | 23 analíticas, medias crudas de referencia, discrepancias declaradas |
| `figures:<nombre>` | PNG y PDF, 300 dpi por chunk `pHYs`, ancho 5,04 in, paneles declarados sin repetir dato con unidades distintas |
| `inventory` | Árbol exacto, sin NaN/Inf, UTF-8 estricto, docstrings, sin holdout en los CSV de §4.2 |
| `hash` | Manifiesto SHA-256 de `results/*` (sin `results/eda`), con `--out` y `--compare` |

La comprobación de que el EDA no toca nada fuera de su carpeta no se hace con `git status`:
`.gitignore` excluye `results/`, así que git es ciego a los artefactos que hay que proteger. Se
usa el manifiesto de hashes antes y después.

## Decisiones de diseño

- **Kaplan-Meier con `lifelines`, no implementada a mano.** Ya es dependencia del proyecto y se
  usa en tres scripts. El riesgo de caja negra se acota con un oráculo numpy independiente en el
  verificador, que recalcula el producto-límite y exige coincidencia a 1e-9.
- **Paquete `eda/` con orquestador `EDA.py`.** Un fichero único con seis bloques y seis figuras
  no se puede revisar por tareas; los módulos permiten cortes pequeños y `ruff` los formatea.
- **`modules` se declara como lista explícita, no con `glob`.** Un `glob` no invalidaría la
  salida al añadir un módulo; la lista sí. Declarar un módulo que aún no existe rompe el dryrun,
  así que se añade en la misma tarea que lo crea.
- **Jaccard sobre visitas aplanadas.** Una matriz de pacientes sería O(N²) y no responde a la
  pregunta (qué se mide junto).
- **El paciente es la unidad de inferencia.** Los contrastes se hacen sobre tasas por paciente y
  reportan tamaño del efecto.
- **Figuras a 12,8 cm con `tight_layout`, sin `bbox_inches="tight"`.** Con recorte, la imagen
  guardada cambia de ancho y la fuente de 8 pt deja de ser de 8 pt al insertarla al 80 %.
- **`resources/loinc_labels.csv` con excepción en `.gitignore`.** `/resources/` y `*.csv` están
  ignorados por tamaño (el mapa SNOMED pesa ~100 MB), pero las etiquetas son fuente única de los
  nombres de las figuras y deben viajar con el código.

## Limitaciones y avisos

- Solo describe **train**, salvo la tabla de cohorte y la Kaplan-Meier, que son descriptivas y no
  intervienen en ninguna selección de modelo.
- No mide asociación entre variables ni valor predictivo: eso es el capítulo 4.
- La etapa cruda del embudo queda como `null` mientras `data/` no esté disponible.
- Las siete figuras antiguas del EDA siguen en `MemoriaTFM/images/` hasta que se reescriba §4.2
  de la memoria; `pixi run figs` copia las nuevas y no borra nada.
- Cuatro analíticas aparecen marcadas en el bloque 5. Es un hallazgo, no un error del código: el
  z-score se ajustó en train sobre las observaciones antes del filtrado por edad y código, y la
  desviación observada se separa menos del 1,5 % de 1.

## Cifras de referencia (contrato del artefacto)

| Cifra | Valor | Campo |
|---|---|---|
| Pacientes con par válido | 9 917 | `blocks.C.flow` |
| Pares paciente-landmark | 202 672 | `blocks.C.flow` |
| Muestras sin analítica previa | 160 233 (79,06 %) | `blocks."1"` |
| Timesteps sin analítica | 81 881 de 318 780 | `blocks."2"` |
| Pares de Δt | 306 711 | `blocks."2"` |
| Δt en múltiplos de 7 días | 43,15 % | `blocks."2"` |
| Moda anual | 371 días, 45 106 pares | `blocks."2"` |
| Visitas por año, y=0 / y=1 | 0,260 / 0,699 | `blocks."2".informative_sampling` |
| δ de Cliff | 0,479 | `blocks."2".informative_sampling` |
| Parejas con Jaccard ≥ 0,95 | 48 de 253 | `blocks."4"` |
| Duración de la regla | ~26 s | `provenance.runtime_seconds` |
| Primer evento observado (train) | 45,3 años | `blocks.C.km.train.event_age_min` |

Si una cifra no cae en la tolerancia de `check_eda.py`, no se ajusta el número: se documenta la
discrepancia y se para.

## Mantenimiento

- **Añadir un bloque**: función en `eda/blocks.py`, fila en el CSV, figura en `eda/plots.py` y
  grupo en `check_eda.py`. El contrato del JSON exige una clave por bloque.
- **Cambiar el ancho de las figuras**: `eda.figure_width_cm` en la configuración; no hay que
  tocar código.
- **Cambiar una etiqueta**: `resources/loinc_labels.csv`. Las figuras y las tablas la leen de
  ahí.
- **Auditar cifras**: `results/eda/eda_summary.json` y `results/eda/tables/*.csv` son la fuente;
  `MemoriaTFM/docs/03_metodos.md` §7 recoge las conclusiones.
