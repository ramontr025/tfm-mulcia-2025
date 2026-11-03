# Cribado de cáncer de pulmón con modelos secuenciales sobre series clínicas irregulares

Trabajo Fin de Máster del **Máster en Lógica, Computación e Inteligencia Artificial** (Universidad de Sevilla).

El repositorio contiene un pipeline reproducible (Snakemake + Pixi) que construye una cohorte sintética
a partir de Synthea, la representa como secuencias clínicas irregulares, entrena cinco modelos de riesgo
—Cox con última observación arrastrada, LSTM, dos GRU-D (con y sin códigos CIE-10) y una adaptación de
STraTS— y los evalúa con métricas de supervivencia sobre un pool de holdout. Incluye soporte para
múltiples réplicas de entrenamiento y particionado, así como comparaciones emparejadas rigurosas mediante
bootstrap de dos niveles. La memoria del TFM se compila aparte con Quarto/Typst a partir de los
artefactos que produce el pipeline.

> Este README explica cómo reconstruir el entorno, ejecutar el pipeline y compilar la memoria por primera vez.

---

## Estructura del repositorio

```
.
├── workflow/                  Pipeline Snakemake
│   ├── Snakefile              Config, mapas derivados, réplicas y endpoints (`all`, `seeds`, `paired`)
│   ├── rules/
│   │   ├── ingest.smk         build_cohort · split_cohort · fit_lab_stats ·
│   │   │                      build_icd10_map · build_sequences (+ réplicas de partición)
│   │   ├── eda.smk            eda (análisis exploratorio y tablas clínicas)
│   │   └── models.smk         train · eval · compare · seeds · paired
│   └── scripts/               Módulos ejecutables y utilidades del pipeline
│       ├── metrics.py         Métricas compartidas (AUROC, IPCW C-index, IPCW Brier, IPA, estimador agrupado)
│       ├── train.py           Entrenamiento de modelos con control de semilla y concurrencia de hilos
│       ├── eval.py            Evaluación con estimandos agrupados y exportación de predicciones
│       ├── compare.py         Comparativa global y figuras maestras
│       ├── seeds.py           Varianza entre réplicas (runs.csv, summary.csv)
│       ├── paired.py          Comparación emparejada con bootstrap anidado de dos niveles
│       ├── eda/               Módulos del análisis exploratorio (io, views, blocks, plots)
│       └── sync_figs.py       Copia figuras y tablas a MemoriaTFM/images/ (`pixi run figs`)
├── config/
│   └── config.yaml            Cohortes, landmarks 45–70, horizonte 5 años, 5 modelos, bootstrap, réplicas y recursos
├── MemoriaTFM/                Memoria del TFM (Quarto → PDF Typst y HTML)
│   ├── MemoriaTFM.qmd         Documento maestro (incluye los capítulos)
│   ├── 01_introduccion.qmd … 07_conclusiones.qmd
│   ├── apendice_estudio_estatico.qmd   Reproducción del estudio estático de referencia y análisis de fugas
│   ├── teoria.qmd             Material de partida de §3.4 (no entra en el render)
│   ├── referencias.bib
│   ├── logo.png
│   ├── images/                Figuras (PNG/PDF/SVG) y tablas del EDA que consume el render
│   ├── _extensions/calote/memoriatfetypst/   Extensión Typst de plantilla
│   ├── MemoriaTFM.pdf         Memoria compilada
│   └── MemoriaTFM.html        Memoria compilada (HTML autocontenido)
├── appendix/                  Material de fases anteriores y apéndices
│   ├── modelos_entrenados/    Modelos predictivos del Apéndice A (pkl tuneados + preprocesador)
│   ├── notebooks/             Notebooks del modelo estático (01_replica_paper_baseline.ipynb, …)
│   ├── Variables_codif/       Codificaciones SNOMED, LOINC y RxNorm + reveladoras de cáncer
│   ├── despliegue_Docker/     Despliegue local del modelo (docker-compose + guía)
│   └── 2DA FASE/              Repositorio de la segunda fase del proyecto
├── figs/                      dag.png y rulegraph.png (grafos de Snakemake actualizados)
├── resources/
│   └── loinc_labels.csv       Etiquetas LOINC de las analíticas
├── sandbox/                   Notebooks de exploración (transformer, GRU-D)
├── _extensions/               Copia de la extensión Typst en la raíz (el render usa la de MemoriaTFM/)
├── pixi.toml · pixi.lock      Entorno y tareas reproducibles
└── readme.md
```

### Etapas del pipeline

| Etapa | Reglas | Qué produce |
|---|---|---|
| Ingesta principal | `build_cohort`, `split_cohort`, `fit_lab_stats`, `build_icd10_map`, `build_sequences` | Cohortes por pool, partición train/val de discovery, normalización ajustada en train y secuencias temporales |
| Réplicas de ingesta | `split_cohort_replica`, `fit_lab_stats_replica`, `build_sequences_replica` | Secuencias bajo `results/replicas/s{seed}/` con particiones y normalizaciones independientes por semilla |
| EDA (1.5) | `eda` | 12 figuras, 7 tablas y `eda_summary.json` |
| Modelado principal | `train`, `eval`, `compare` | Un `.pkl` por modelo, métricas `.json` (estimandos agrupados y cluster bootstrap), predicciones `.pkl` y figuras comparativas |
| Análisis de réplicas | `seeds` | Varianza entre réplicas por modelo (`results/seeds/runs.csv` y `summary.csv`) evaluando estabilidad de entrenamiento y partición |
| Comparación emparejada | `paired` | Contrastes pareados dentro de cada réplica sobre los mismos pacientes de holdout mediante bootstrap de dos niveles (`results/paired/differences.csv`) |

---

## Requisitos previos

| Requisito | Versión / nota |
|---|---|
| **Pixi** | Última versión. Gestiona Python 3.12, Snakemake 9, PyTorch (CPU), pandas, lifelines, scikit-survival… |
| **Quarto** | 1.8.x (probado con 1.8.26). **No** está en `pixi.toml`: se instala aparte. Incluye Typst para el PDF |
| **Git** | Para clonar el repositorio |
| **Plataformas** | El manifiesto declara `win-64` y `linux-64`; no hay soporte declarado para macOS |
| **Datos crudos** | CSV de Synthea (`patients`, `conditions`, `observations`, `.csv.gz`) y el mapa SNOMED→CIE-10 |

El pipeline corre en CPU: no necesita GPU.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio> tfm-mulcia-2025
cd tfm-mulcia-2025
```

### 2. Instalar Pixi

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://pixi.sh/install.ps1 | iex"
```

**Linux:**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 3. Crear el entorno con Pixi

```bash
pixi install
```

Esto materializa `.pixi/` a partir de `pixi.lock` (Python 3.12, Snakemake 9, PyTorch CPU y el resto de
dependencias). Snakemake **no se instala aparte**: vive dentro del entorno.

Comprobación rápida:

```bash
pixi run snakemake --version
pixi run python --version
```

### 4. Colocar los datos de entrada

El pipeline espera los CSV crudos en `data/` y el mapa terminológico en `resources/`.

```bash
mkdir -p data/discovery data/holdout
```

Coloca los CSV de Synthea (`patients`, `conditions`, `observations`) en cada pool:

```
data/discovery/{patients,conditions,observations}.csv.gz
data/holdout/{patients,conditions,observations}.csv.gz
resources/snomed_icd10_map.csv
resources/loinc_labels.csv
```

El mapa SNOMED CT → CIE-10 (~100 MB) se descarga desde
<https://thetuvaproject.com/terminology/snomed-to-icd10-map> y se guarda como
`resources/snomed_icd10_map.csv`. El fichero `resources/loinc_labels.csv` ya está en el repositorio.

Las rutas se configuran en `config/config.yaml`; si tus ficheros están en otro sitio, edita ese fichero.

### 5. Instalar Quarto

Descarga el instalador desde <https://quarto.org/docs/get-started/> y comprueba que está en el `PATH`:

```bash
quarto --version
```

La extensión de plantilla (`memoriatfetypst`) ya está incluida en `MemoriaTFM/_extensions/`, así que no
hay que instalarla. Typst viene con Quarto.

---

## Ejecutar el pipeline

Todas las tareas están declaradas en `pixi.toml`:

| Comando | Qué hace |
|---|---|
| `pixi run dryrun` | `snakemake -n`: resuelve el DAG sin ejecutar nada |
| `pixi run pipeline` | `snakemake -c4`: ejecución completa (raw → secuencias → modelos → evaluación) |
| `pixi run ok` | Formatea (`smkfmt`, `pyfmt`), resuelve el DAG y redibuja los grafos (`dag`, `rdag`) |
| `pixi run figs` | Copia figuras y tablas del pipeline a `MemoriaTFM/images/` |
| `pixi run dag` / `pixi run rdag` | Regenera `figs/dag.png` (ejecución principal) y `figs/rulegraph.png` (con `seeds` y `paired`) |
| `pixi run smkfmt` / `pixi run pyfmt` | Formatea reglas Snakemake / scripts Python |
| `pixi run cluster` | Ejecuta con el executor de SLURM (requiere descomentar el plugin en `pixi.toml`) |

### Comandos de análisis empírico y réplicas

```bash
# Ejecución completa de la varianza entre réplicas (10 réplicas x 5 modelos = 50 entrenamientos)
pixi run snakemake -c32 seeds

# Comparación emparejada con bootstrap de dos niveles
pixi run snakemake -c32 paired

# Ejecución conjunta masiva en máquina de cómputo
pixi run snakemake -c60 seeds paired
```

Notas:

- Para reconstruir un solo objetivo: `pixi run snakemake -c1 results/train/sequences.pkl`.
- En **Windows**, `pixi run ok`, `dag` y `rdag` usan `mkdir -p` y fallan en PowerShell. Ejecuta esas tareas
  desde Git Bash/WSL, o lanza por separado `pixi run dryrun`, `pixi run smkfmt` y `pixi run pyfmt`.
- `pixi run figs` no borra nada y avisa (sin fallar) de los orígenes que aún no existan. Sobrescribe
  `MemoriaTFM/images/`, que está versionada: revisa `git diff` antes de commitear.

---

## Compilar la memoria

```bash
cd MemoriaTFM
quarto render MemoriaTFM.qmd              # PDF (Typst)
quarto render MemoriaTFM.qmd --to html    # HTML autocontenido
```

Salidas: `MemoriaTFM/MemoriaTFM.pdf` y `MemoriaTFM/MemoriaTFM.html`.

Estructura de la memoria:
- **Capítulos principales (`01` a `07`)**: Introducción, Materiales, Métodos, Resultados, Discusión, Limitaciones y Conclusiones.
- **Apéndice metodológico unificado (`apendice_estudio_estatico.qmd`)**: Reúne en un único documento autocontenido la reproducción del modelo estático previo (Chen y Chen), la auditoría de fuga de información (*data leakage*), la ingeniería ontológica (Snowstorm y ATC nivel 2) y la demostración formal de que la aparente predictibilidad transversal respondía a un artefacto del generador de Synthea (*null experiment*), justificando el viraje al modelado longitudinal y landmarking del cuerpo del trabajo.

Ten en cuenta:

- El documento tiene `execute: eval: false`: **no reejecuta** los chunks de Python, solo maqueta.
- Las figuras viven en `MemoriaTFM/images/` y están versionadas: la memoria compila en un clon limpio
  sin ejecutar el pipeline. Si reejecutas el pipeline, `pixi run figs` refresca esa carpeta.
- `teoria.qmd` (fuera del render) referencia los esquemas de celdas RNN/LSTM/GRU/GRU-D, por lo que Quarto
  avisa de los cruces `@fig-rnn`, `@fig-lstm`, `@fig-gru` y `@fig-grud` sin resolver. Son avisos, no errores.
- El render debe lanzarse **desde `MemoriaTFM/`**: ahí viven `referencias.bib`, `logo.png` y la extensión.

---

## Salidas y artefactos

| Ruta | Contenido |
|---|---|
| `results/{discovery,holdout}/cohort.parquet` | Cohorte por pool (Id, duración, evento) |
| `results/{train,val}/cohort.parquet` | Reparto de discovery a nivel de paciente, estratificado por evento |
| `results/{train,val,holdout}/sequences.pkl` | Secuencias clínicas irregulares que consumen los modelos |
| `results/replicas/s{seed}/{train,val,holdout}/sequences.pkl` | Secuencias de las réplicas con normalización ajustada en su propio train |
| `results/lab_stats.json` | Media y desviación por analítica, ajustadas solo en train |
| `results/snomed_to_icd10.json` | Mapa de códigos SNOMED a capítulos CIE-10 |
| `results/models/{name}.pkl` | Pesos de `grud`, `grud_labs`, `cox`, `lstm` y `strats` (y variantes `_s{seed}`) |
| `results/metrics/{name}.json` | C-index IPCW, AUROC, Brier IPCW e IPA con estimandos agrupados e IC bootstrap |
| `results/preds/{name}.pkl` | Predicciones de holdout y referencias de entrenamiento para comparaciones rápidas |
| `results/comparison.{csv,png}` | Tabla y figura comparativa de los cinco modelos |
| `results/{auroc,brier,cindex_ipcw,ipa}.png` | Curvas y métricas por landmark |
| `results/figure_main.{png,pdf}` | Figura principal de la memoria |
| `results/seeds/runs.csv` | Estimandos agrupados individuales de todas las réplicas evaluadas |
| `results/seeds/summary.csv` | Media, desviación estándar, mínimo y máximo por modelo a través de las réplicas |
| `results/paired/by_replica.csv` | Comparación emparejada detallada por réplica |
| `results/paired/differences.csv` | Diferencias entre modelos e IC 95 % obtenidos mediante bootstrap de dos niveles |
| `results/eda/figures`, `results/eda/tables` | 12 figuras (PNG + PDF) y 7 tablas del EDA |
| `results/eda/eda_summary.json` | Resumen numérico del EDA |
| `figs/dag.png`, `figs/rulegraph.png` | Grafos del pipeline (ejecución base y grafo completo de reglas) |

`results/` se reconstruye con `pixi run pipeline` (o `seeds` / `paired` para los análisis de robustez).

---

## Autoría

Ramón Trinidad Acedo — Máster Universitario en Lógica, Computación e Inteligencia Artificial,
Universidad de Sevilla. Tutor: Carlos Loucera Muñecas.
