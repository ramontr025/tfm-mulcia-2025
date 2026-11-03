rule train:
    input:
        train=lambda wc: sequences_for(wc.name, "train"),
        val=lambda wc: sequences_for(wc.name, "val"),
        dataset="workflow/scripts/dataset.py",
        model="workflow/scripts/model.py",
        engine="workflow/scripts/engine.py",
        loss="workflow/scripts/loss.py",
        cox="workflow/scripts/cox.py",
    output:
        "results/models/{name}.pkl",
    benchmark:
        "benchmarks/train/{name}.tsv"
    threads: lambda wc: MODELS[wc.name].get("threads", config["compute"]["train_threads"])
    priority: lambda wc: MODELS[wc.name].get("priority", 0)
    resources:
        mem_mb=config["compute"]["train_mem_mb"],
    log:
        "logs/train/{name}.log",
    params:
        arch=lambda wc: MODELS[wc.name]["arch"],
        icd=lambda wc: MODELS[wc.name]["icd"],
        hidden_dim=lambda wc: MODELS[wc.name].get("hidden_dim"),  # None para cox
        num_heads=lambda wc: MODELS[wc.name].get("num_heads"),  # None para cox y secuenciales
        num_layers=lambda wc: MODELS[wc.name].get("num_layers"),  # None para cox y secuenciales
        dim_feedforward=lambda wc: MODELS[wc.name].get("dim_feedforward"),  # None para cox y secuenciales
        dropout=lambda wc: MODELS[wc.name].get("dropout"),
        demo_dim=lambda wc: MODELS[wc.name].get("demo_dim"),  # None para cox y secuenciales
        batch_size=lambda wc: MODELS[wc.name]["batch_size"],
        learning_rate=lambda wc: MODELS[wc.name]["learning_rate"],
        weight_decay=lambda wc: MODELS[wc.name]["weight_decay"],
        max_epochs=lambda wc: MODELS[wc.name]["max_epochs"],
        patience=lambda wc: MODELS[wc.name]["patience"],
        event_oversample=lambda wc: MODELS[wc.name]["event_oversample"],
        max_events=lambda wc: MODELS[wc.name]["max_seq_len"],
        age_weight_gamma=lambda wc: MODELS[wc.name]["age_weight_gamma"],
        seed=lambda wc: MODELS[wc.name]["seed"],  # splits.seed en la principal, s en las replicas
        index_age=config["cohort"]["index_age"],
        landmark_min=config["cohort"]["landmark_min"],
        landmark_max=config["cohort"]["landmark_max"],
    script:
        "../scripts/train.py"


rule eval:
    input:
        model_path="results/models/{name}.pkl",
        train=lambda wc: sequences_for(wc.name, "train"),
        val=lambda wc: sequences_for(wc.name, "val"),
        holdout=lambda wc: sequences_for(wc.name, "holdout"),
        dataset="workflow/scripts/dataset.py",
        model="workflow/scripts/model.py",
        engine="workflow/scripts/engine.py",
        loss="workflow/scripts/loss.py",
        cox="workflow/scripts/cox.py",
        metrics="workflow/scripts/metrics.py",
    output:
        metrics="results/metrics/{name}.json",
        preds="results/preds/{name}.pkl",
    benchmark:
        "benchmarks/eval/{name}.tsv"
    threads: config["compute"]["eval_threads"]
    resources:
        mem_mb=config["compute"]["eval_mem_mb"],
    log:
        "logs/eval/{name}.log",
    params:
        horizon=config["cohort"]["horizon"],
        landmark_min=config["cohort"]["landmark_min"],
        landmark_max=config["cohort"]["landmark_max"],
        index_age=config["cohort"]["index_age"],
        boot_n=config["bootstrap"]["n_boot"],
        boot_n_pooled=config["bootstrap"]["n_boot_pooled"],
        boot_seed=config["bootstrap"]["seed"],
        boot_landmarks=config["bootstrap"]["landmarks"],
    script:
        "../scripts/eval.py"


rule compare:
    input:
        metrics=expand("results/metrics/{name}.json", name=BASE_MODELS),
        compare="workflow/scripts/compare.py",
    output:
        csv="results/comparison.csv",
        png="results/comparison.png",
        # figura central para paper/memoria (los PNG por metrica se escriben como
        # subproductos en el mismo directorio, sin declarar)
        main_png="results/figure_main.png",
        main_pdf="results/figure_main.pdf",
    benchmark:
        "benchmarks/compare.tsv"
    log:
        "logs/compare.log",
    script:
        "../scripts/compare.py"


# Varianza entre replicas: la ejecucion principal de cada modelo mas sus replicas
# {name}_s{seed} (replica_seeds en la config), cada una con su particion y su semilla.
# Escribe una fila por ejecucion y el resumen por modelo (media, desviacion, minimo y
# maximo) de los agrupados en holdout.
#   pixi run snakemake -c60 seeds paired
rule seeds:
    input:
        metrics=expand(
            "results/metrics/{name}_s{seed}.json", name=BASE_MODELS, seed=SEEDS
        ),
        base=expand("results/metrics/{name}.json", name=BASE_MODELS),
    output:
        runs="results/seeds/runs.csv",
        summary="results/seeds/summary.csv",
    benchmark:
        "benchmarks/seeds.tsv"
    log:
        "logs/seeds.log",
    params:
        main_seed=MAIN_SEED,
    script:
        "../scripts/seeds.py"


# Comparacion emparejada: dentro de cada replica los modelos comparten particion, asi
# que se comparan sobre los mismos pacientes de holdout. Bootstrap de dos niveles
# (replica al azar y cluster bootstrap de pacientes) para el IC de cada diferencia.
rule paired:
    input:
        preds=expand("results/preds/{name}.pkl", name=BASE_MODELS)
        + expand("results/preds/{name}_s{seed}.pkl", name=BASE_MODELS, seed=SEEDS),
        metrics="workflow/scripts/metrics.py",
    output:
        by_replica="results/paired/by_replica.csv",
        differences="results/paired/differences.csv",
    benchmark:
        "benchmarks/paired.tsv"
    threads: config["paired"]["threads"]
    resources:
        mem_mb=config["paired"]["mem_mb"],
    log:
        "logs/paired.log",
    params:
        base_models=BASE_MODELS,
        main_seed=MAIN_SEED,
        n_boot=config["paired"]["n_boot"],
        seed=config["paired"]["seed"],
    script:
        "../scripts/paired.py"
