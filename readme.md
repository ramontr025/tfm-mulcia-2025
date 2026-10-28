# Cribado de cáncer de pulmón con modelos secuenciales sobre series clínicas irregulares

Trabajo Fin de Máster del **Máster en Lógica, Computación e Inteligencia Artificial** (Universidad de Sevilla).

El repositorio contiene un pipeline reproducible (Snakemake + Pixi) que construye una cohorte sintética
a partir de Synthea, la representa como secuencias clínicas irregulares, entrena cinco modelos de riesgo
—Cox con última observación arrastrada, LSTM, dos GRU-D (con y sin códigos CIE-10) y una adaptación de
STraTS— y los evalúa con métricas de supervivencia sobre un pool de holdout. La memoria del TFM se
compila aparte con Quarto/Typst a partir de los artefactos que produce el pipeline.

> Este README explica cómo reconstruir el entorno, ejecutar el pipeline y compilar la memoria por primera vez.

---

## Estructura del repositorio

```
.
├── workflow/                  Pipeline Snakemake
│   ├── Snakefile              Config, mapas derivados y `rule all`
│   ├── rules/
│   │   ├── ingest.smk         build_cohort · split_cohort · fit_lab_stats ·
│   │   │                      build_icd10_map · build_sequences
│   │   ├── eda.smk            eda
│   │   └── models.smk         train · eval · compare
│   └── scripts/               Un script por regla (Snakemake los inyecta como `script:`)
│       ├── eda/               Módulos del análisis exploratorio (io, views, blocks, plots)
│       └── sync_figs.py       Copia figuras y tablas a MemoriaTFM/images/ (`pixi run figs`)
├── config/
│   └── config.yaml            Cohortes, landmarks 45–70, horizonte 5 años, 5 modelos, bootstrap
├── MemoriaTFM/                Memoria del TFM (Quarto → PDF Typst y HTML)
│   ├── MemoriaTFM.qmd         Documento maestro (incluye los capítulos)
│   ├── 01_introduccion.qmd … 07_conclusiones.qmd
│   ├── apendice_A_modelo_estatico.qmd
│   ├── apendice_B_resultados_estatico.qmd
│   ├── apendice_C_reveladoras.qmd
│   ├── teoria.qmd             Material de partida de §3.4 (no entra en el render)
│   ├── referencias.bib
│   ├── logo.png
│   ├── _extensions/calote/memoriatfetypst/   Extensión Typst de plantilla
│   ├── images/                [local] Figuras que consume el render
│   ├── docs/                  [local] Dossieres de investigación de la redacción
│   ├── MemoriaTFM.pdf         Memoria compilada
│   └── MemoriaTFM.html        Memoria compilada (HTML autocontenido)
├── appendix/                  Material de fases anteriores y apéndices
│   ├── modelos_entrenados/    Modelos predictivos del Apéndice A (pkl tuneados + preprocesador)
│   ├── notebooks/             Notebooks del modelo estático (01_replica_paper_baseline.ipynb, …)
│   ├── Variables_codif/       Codificaciones SNOMED, LOINC y RxNorm + reveladoras de cáncer
│   ├── despliegue_Docker/     Despliegue local del modelo (docker-compose + guía)
│   ├── 2DA FASE/              Repositorio de la segunda fase del proyecto
│   └── data/                  [local] CSV crudos de Synthea (discovery y holdout)
├── results/                   [local] Artefactos generados por el pipeline
│   ├── {discovery,holdout}/cohort.parquet
│   ├── {train,val,holdout}/cohort.parquet · sequences.pkl
│   ├── models/{name}.pkl
│   ├── metrics/{name}.json
│   ├── comparison.csv · comparison.png · figure_main.*
│   ├── eda/figures · eda/tables · eda_summary.json
│   └── lab_stats.json · snomed_to_icd10.json
├── figs/                      dag.png y rulegraph.png (grafos de Snakemake)
├── logs/                      [local] Un log por regla (`logs/<regla>/…`)
├── resources/                 [local] snomed_icd10_map.csv (~100 MB, fuera de git) + loinc_labels.csv (versionado)
├── sandbox/                   Notebooks de exploración (transformer, GRU-D)
├── docs/                      [local] Documentación técnica del pipeline (no versionada)
├── spec/                      [local] Especificaciones y agentes de trabajo
├── _extensions/               Copia de la extensión Typst en la raíz (el render usa la de MemoriaTFM/)
├── pixi.toml · pixi.lock      Entorno y tareas reproducibles
└── readme.md
```

`[local]` = no viaja en git (ver `.gitignore`): un clon nuevo **no** lo trae y hay que reconstruirlo.

### Etapas del pipeline

| Etapa | Reglas | Qué produce |
|---|---|---|
| Ingesta | `build_cohort`, `split_cohort`, `fit_lab_stats`, `build_icd10_map`, `build_sequences` | Cohortes por pool, reparto train/val, normalización de analíticas y secuencias |
| EDA (1.5) | `eda` | 12 figuras, 7 tablas y `eda_summary.json` |
| Modelado | `train`, `eval`, `compare` | Un `.pkl` por modelo, métricas con bootstrap agrupado y comparativa |

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

### 4. Colocar los datos que no viajan en git

El pipeline espera los CSV crudos en `data/` y el mapa terminológico en `resources/`. Ni `data/` ni
`appendix/data/` viajan en git; si conservas el material local de `appendix/data/`, cópialo:

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item appendix\data\discovery data\ -Recurse
Copy-Item appendix\data\holdout   data\ -Recurse
```

**Linux:**

```bash
mkdir -p data
cp -r appendix/data/discovery data/
cp -r appendix/data/holdout   data/
```

Si partes de cero, genera o descarga los CSV de Synthea (`patients`, `conditions`, `observations`) y
colócalos con esa misma estructura de dos pools.

Después hay que descargar el mapa SNOMED CT → CIE-10 desde
<https://thetuvaproject.com/terminology/snomed-to-icd10-map> y guardarlo como
`resources/snomed_icd10_map.csv` (~100 MB). El fichero `resources/loinc_labels.csv` sí está versionado.

Árbol esperado:

```
data/discovery/{patients,conditions,observations}.csv.gz
data/holdout/{patients,conditions,observations}.csv.gz
resources/snomed_icd10_map.csv
resources/loinc_labels.csv
```

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
| `pixi run ok` | Formatea (`smkfmt`, `pyfmt`), resuelve el DAG y redibuja los grafos |
| `pixi run figs` | Copia figuras y tablas del pipeline a `MemoriaTFM/images/` |
| `pixi run dag` / `pixi run rdag` | Regenera `figs/dag.png` y `figs/rulegraph.png` |
| `pixi run smkfmt` / `pixi run pyfmt` | Formatea reglas Snakemake / scripts Python |
| `pixi run cluster` | Ejecuta con el executor de SLURM (requiere descomentar el plugin en `pixi.toml`) |

Primer arranque recomendado:

```bash
pixi run dryrun      # comprueba que el DAG se cablea bien
pixi run pipeline    # ejecución completa; el entrenamiento de los 5 modelos es la parte lenta
pixi run figs        # publica las figuras en MemoriaTFM/images/
```

Notas:

- Para reconstruir un solo objetivo: `pixi run snakemake -c1 results/train/sequences.pkl`.
- En **Windows**, `pixi run ok`, `dag` y `rdag` usan `mkdir -p` y fallan en PowerShell. Ejecuta esas tareas
  desde Git Bash/WSL, o lanza por separado `pixi run dryrun`, `pixi run smkfmt` y `pixi run pyfmt`.
- `pixi run figs` no borra nada y avisa (sin fallar) de los orígenes que aún no existan.

---

## Compilar la memoria por primera vez

```bash
cd MemoriaTFM
quarto render MemoriaTFM.qmd              # PDF (Typst)
quarto render MemoriaTFM.qmd --to html    # HTML autocontenido
```

Salidas: `MemoriaTFM/MemoriaTFM.pdf` y `MemoriaTFM/MemoriaTFM.html`.

Ten en cuenta:

- El documento tiene `execute: eval: false`: **no reejecuta** los chunks de Python, solo maqueta.
- Las figuras se leen de `MemoriaTFM/images/`, que no está versionada. Sin `pixi run figs` (o sin copiar
  los PNG/PDF a mano) la compilación termina con avisos de imagen ausente.
- Nueve figuras de los apéndices (`fig-rnn.png`, `fig-grud.png`, `pr_roc_modelo_cie.png`, …) no las genera
  el pipeline y deben añadirse manualmente si se quieren sin avisos.
- El render debe lanzarse **desde `MemoriaTFM/`**: ahí viven `referencias.bib`, `logo.png` y la extensión.

---

## Salidas y artefactos

| Ruta | Contenido |
|---|---|
| `results/{discovery,holdout}/cohort.parquet` | Cohorte por pool (Id, duración, evento) |
| `results/{train,val}/cohort.parquet` | Reparto de discovery a nivel de paciente, estratificado por evento |
| `results/{train,val,holdout}/sequences.pkl` | Secuencias clínicas irregulares que consumen los modelos |
| `results/lab_stats.json` | Media y desviación por analítica, ajustadas solo en train |
| `results/snomed_to_icd10.json` | Mapa de códigos SNOMED a capítulos CIE-10 |
| `results/models/{name}.pkl` | Pesos de `grud`, `grud_labs`, `cox`, `lstm` y `strats` |
| `results/metrics/{name}.json` | C-index IPCW, AUROC, Brier IPCW e IPA con intervalos bootstrap |
| `results/comparison.{csv,png}` | Tabla y figura comparativa de los cinco modelos |
| `results/{auroc,brier,cindex_ipcw,ipa}.png` | Curvas y métricas por landmark |
| `results/figure_main.{png,pdf}` | Figura principal de la memoria |
| `results/eda/figures`, `results/eda/tables` | 12 figuras (PNG + PDF) y 7 tablas del EDA |
| `results/eda/eda_summary.json` | Resumen numérico del EDA |
| `logs/<regla>/…` | Un log por ejecución de regla |
| `figs/dag.png`, `figs/rulegraph.png` | Grafos del pipeline |

`results/` está ignorado por git: se reconstruye con `pixi run pipeline`.

---

## Documentación local (no versionada)

Estos materiales se conservan en disco pero **no se distribuyen con el repositorio**:

- `docs/` — documentación técnica del pipeline:
  `bootstrap.md`, `build_cohort.md`, `build_icd10_map.md`, `build_sequences.md`, `competing_risks.md`,
  `cox_calibration.md`, `eda.md`, `eval.md`, `strats_implementation.md`, `survival_metrics.md`.
- `MemoriaTFM/docs/` — dossieres de investigación que alimentan los capítulos de la memoria.
- `appendix/2DA FASE/us-mulcia-2026-ramon/docs/` — documentación de la segunda fase del proyecto.
- `spec/` — especificaciones y configuración de los agentes de trabajo.

---

## Autoría

Ramón Trinidad Acedo — Máster Universitario en Lógica, Computación e Inteligencia Artificial,
Universidad de Sevilla. Tutor: Carlos Loucera Muñecas.
