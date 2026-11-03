# ─────────────────────────────────────────────────────────────────────────────
# Stage 1.5: EDA de la cohorte (§4.1) y del tensor clinico longitudinal (§4.2).
#
# Una regla que lee los pickles de secuencias de los tres splits, las cuatro
# cohortes, las estadisticas de normalizacion y las etiquetas de las analiticas, y
# produce results/eda/ con: eda_summary.json (esquema 2.0 + procedencia), siete
# tablas CSV y seis figuras en PNG 300 dpi + PDF.
#
# El protocolo (bloques, unidades y poblaciones) esta declarado en
# MemoriaTFM/docs/03_metodos.md. Resumen:
#   C  cohorte y Kaplan-Meier (train, val, holdout; discovery solo en el embudo)
#   1  lo que ve el modelo en cada corte (muestra paciente-landmark)
#   2  espaciado entre timesteps y muestreo informativo por paciente
#   3  densidad de observacion por visita y por analitica
#   4  redundancia (Jaccard por visita, orden jerarquico)
#   5  magnitudes y contraste de la normalizacion con lab_stats.json
#
# La vista de modelo sale de RawSequenceDatasetWithICD: el truncado por landmark y
# max_events no se reimplementan (dataset.py). El EDA no escribe fuera de
# results/eda/.
#
# Salida: directory("results/eda") con eda_summary.json, tables/*.csv y
# figures/fig_{cohort,sample_coverage,time_structure,observation_density,
# lab_redundancy,value_scale}.{png,pdf}
# ─────────────────────────────────────────────────────────────────────────────


rule eda:
    input:
        sequences=expand("results/{split}/sequences.pkl", split=SPLITS),
        cohorts=expand(
            "results/{split}/cohort.parquet",
            split=["discovery", "train", "val", "holdout"],
        ),
        lab_stats="results/lab_stats.json",
        labels="resources/loinc_labels.csv",
        dataset="workflow/scripts/dataset.py",
        modules=[
            "workflow/scripts/eda/__init__.py",
            "workflow/scripts/eda/io.py",
            "workflow/scripts/eda/views.py",
            "workflow/scripts/eda/blocks.py",
            "workflow/scripts/eda/plots.py",
        ],
    output:
        directory("results/eda"),
    benchmark:
        "benchmarks/eda.tsv"
    threads: config["compute"]["ingest_threads"]
    resources:
        mem_mb=config["compute"]["ingest_mem_mb"],
    log:
        "logs/eda.log",
    params:
        landmarks=LANDMARKS,
        index_age=config["cohort"]["index_age"],
        horizon=config["cohort"]["horizon"],
        max_events=config["eda"].get("max_seq_len", 200),
        age_weight_gamma=config["eda"].get("age_weight_gamma", 1.0),
        seed=config["eda"]["seed"],
        dpi=config["eda"]["dpi"],
        figure_width_cm=config["eda"]["figure_width_cm"],
        week_modulus_days=config["eda"]["week_modulus_days"],
        day_multiple_tolerance=config["eda"]["day_multiple_tolerance"],
        annual_mode_center_days=config["eda"]["annual_mode_center_days"],
        jaccard_flag_threshold=config["eda"]["jaccard_flag_threshold"],
        value_sigma_limit=config["eda"]["value_sigma_limit"],
        labstats_z_mean_tol=config["eda"]["labstats_z_mean_tol"],
        labstats_z_std_tol=config["eda"]["labstats_z_std_tol"],
        age_bands=config["eda"]["age_bands"],
    script:
        "../scripts/EDA.py"
