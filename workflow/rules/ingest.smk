# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Ingest. De las fuentes crudas a los datos que consumen los modelos.
#
# Cinco pasos, uno por concepto. Cada regla lee y escribe lo justo, para que el DAG
# se lea como el pipeline conceptual y ninguna entrada sea un canal escondido:
#
#   1. build_cohort     raw de UN pool          -> results/{pool}/cohort.parquet
#                       Id/duration/event. Corre una vez por pool (discovery, holdout),
#                       EL MISMO codigo: asi se ve que reciben identico preprocesamiento.
#   2. split_cohort     results/discovery/cohort -> results/{train,val}/cohort.parquet
#                       Reparte SOLO discovery en train/val. holdout no pasa por aqui.
#   3. fit_lab_stats    results/train/cohort +   -> results/lab_stats.json
#                       data/discovery/observations
#                       Media y desviacion de cada lab, ajustadas SOLO en train. Un
#                       unico artefacto, que los tres build_sequences comparten.
#   4. build_icd10_map  resources/snomed_icd10_map.csv -> results/snomed_to_icd10.json
#                       Reduce el mapa SNOMED->ICD-10 (~100 MB) a un diccionario compacto,
#                       una vez. Otro artefacto compartido, como fit_lab_stats.
#   5. build_sequences  cohort del split + raw del pool -> results/{split}/sequences.pkl
#                       + lab_stats.json + snomed_to_icd10.json
#                       Una regla por split, gracias al comodin {split}.
#
# discovery y holdout nunca se mezclan: discovery es de donde sale todo lo que se usa
# para AJUSTAR el modelo (se reparte en train/val, y las estadisticas de normalizacion
# salen solo de train); holdout es el pool de prueba final, aparte desde el origen.
# Como el pool holdout se llama igual que su split, results/holdout/cohort.parquet ya
# es la cohorte de holdout: no se reparte, y build_sequences la lee por la misma ruta
# results/{split}/cohort.parquet que train y val.
#
# El paso 3 es la clave metodologica de este issue hecha nodo del DAG: la normalizacion
# se ajusta en train y val/holdout la REUTILIZAN. Al ser un artefacto propio
# (results/lab_stats.json) que build_sequences recibe como entrada, esa regla es una
# arista visible en el grafo, no un recalculo escondido dentro de cada trabajo. Asi
# build_sequences(holdout) depende de lab_stats.json (unas estadisticas), no de los
# datos crudos de train: no hay ninguna fuga, y el grafo lo deja claro.
#
# Cada regla usa un `script:`, un archivo .py que recibe el objeto `snakemake` inyectado
# con sus input, output y params. Nunca leas rutas fijas dentro del script.
#
# Cada regla declara ademas un `log:` propio (con el mismo comodin que la salida, para
# que cada trabajo tenga su archivo): logs/<regla>/<wildcard>.log. Snakemake NO redirige
# solo la salida de un `script:` ahi; el script lo hace en su primera linea con
# `sys.stdout = sys.stderr = open(snakemake.log[0], "w")`. Asi cada trabajo deja su
# propio rastro (prints y cualquier traceback) en logs/, separado del resto.
# ─────────────────────────────────────────────────────────────────────────────


# 1. build_cohort deriva la cohorte de un pool (discovery o holdout) con el MISMO
# codigo, gracias al comodin {pool}. No reparte nada. Primera regla escrita entera,
# usala de plantilla.
rule build_cohort:
    input:
        patients=lambda wc: config["raw"][wc.pool]["patients"],
        conditions=lambda wc: config["raw"][wc.pool]["conditions"],
        observations=lambda wc: config["raw"][wc.pool]["observations"],
    output:
        "results/{pool}/cohort.parquet",
    log:
        "logs/build_cohort/{pool}.log",
    params:
        index_age=config["cohort"]["index_age"],
        lung_cancer_codes=config["cohort"]["lung_cancer_snomed_codes"],
    script:
        "../scripts/build_cohort.py"


# 2. split_cohort reparte la cohorte de discovery en train/val (estratificado, con
# semilla). Es el unico paso especifico de discovery; holdout no pasa por aqui.
rule split_cohort:
    input:
        discovery="results/discovery/cohort.parquet",
    output:
        train="results/train/cohort.parquet",
        val="results/val/cohort.parquet",
    log:
        "logs/split_cohort.log",
    params:
        train_frac=config["splits"]["train_frac"],
        seed=config["splits"]["seed"],
    script:
        "../scripts/split_cohort.py"


# 3. fit_lab_stats ajusta la media y la desviacion de cada lab usando SOLO los
# pacientes de train (por eso su entrada es el cohort de train, no el del split). El
# resultado es un unico artefacto que comparten los tres build_sequences.
rule fit_lab_stats:
    input:
        train_cohort="results/train/cohort.parquet",
        observations=config["raw"]["discovery"]["observations"],
    output:
        "results/lab_stats.json",
    log:
        "logs/fit_lab_stats.log",
    params:
        lab_codes=config["features"]["lab_codes_to_keep"],
    script:
        "../scripts/fit_lab_stats.py"


# 4. build_icd10_map reduce el mapa SNOMED->ICD-10 (~100 MB) a un diccionario compacto
# {snomed: capitulo ICD-10}, una vez, para que los tres build_sequences lo compartan (el
# mismo patron de artefacto compartido que fit_lab_stats). No depende de ningun split.
rule build_icd10_map:
    input:
        snomed_map=config["resources"]["snomed_icd10_map"],
    output:
        "results/snomed_to_icd10.json",
    log:
        "logs/build_icd10_map.log",
    script:
        "../scripts/build_icd10_map.py"


def raw_pool_for(wc):
    # train/val vienen del pool discovery; holdout es su propio pool crudo.
    pool = "holdout" if wc.split == "holdout" else "discovery"
    return config["raw"][pool]


# 5. build_sequences corre una vez por split. Toma el cohort de ESE split, las fuentes
# crudas del pool que toque, las estadisticas de normalizacion (ajustadas en train) y el
# mapa SNOMED->ICD-10. Traduce los diagnosticos a capitulo ICD-10, filtra el ruido y el
# desenlace, y escribe el pickle. Ninguna entrada es un canal escondido: cada una tiene
# un papel claro en el DAG.
rule build_sequences:
    input:
        cohort="results/{split}/cohort.parquet",
        lab_stats="results/lab_stats.json",
        snomed_map="results/snomed_to_icd10.json",
        patients=lambda wc: raw_pool_for(wc)["patients"],
        observations=lambda wc: raw_pool_for(wc)["observations"],
        conditions=lambda wc: raw_pool_for(wc)["conditions"],
    output:
        "results/{split}/sequences.pkl",
    log:
        "logs/build_sequences/{split}.log",
    params:
        lab_codes=config["features"]["lab_codes_to_keep"],
        excluded_icd_initials=config["features"]["excluded_icd_initials"],
        outcome_icd_prefixes=config["features"]["outcome_icd_prefixes"],
    script:
        "../scripts/build_sequences.py"
