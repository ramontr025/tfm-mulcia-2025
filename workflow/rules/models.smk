rule train:
    input:
        train="results/train/sequences.pkl",
        val="results/val/sequences.pkl",
        dataset="workflow/scripts/dataset.py",
        model="workflow/scripts/model.py",
        engine="workflow/scripts/engine.py",
        loss="workflow/scripts/loss.py",
        cox="workflow/scripts/cox.py",
    output:
        "results/models/{name}.pkl",
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
        index_age=config["cohort"]["index_age"],
        landmark_min=config["cohort"]["landmark_min"],
        landmark_max=config["cohort"]["landmark_max"],
    script:
        "../scripts/train.py"


rule eval:
    input:
        model_path="results/models/{name}.pkl",
        train="results/train/sequences.pkl",
        val="results/val/sequences.pkl",
        holdout="results/holdout/sequences.pkl",
        dataset="workflow/scripts/dataset.py",
        model="workflow/scripts/model.py",
        engine="workflow/scripts/engine.py",
        loss="workflow/scripts/loss.py",
        cox="workflow/scripts/cox.py",
    output:
        "results/metrics/{name}.json",
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
        metrics=expand("results/metrics/{name}.json", name=list(MODELS)),
        compare="workflow/scripts/compare.py",
    output:
        csv="results/comparison.csv",
        png="results/comparison.png",
        # figura central para paper/memoria (los PNG por metrica se escriben como
        # subproductos en el mismo directorio, sin declarar)
        main_png="results/figure_main.png",
        main_pdf="results/figure_main.pdf",
    log:
        "logs/compare.log",
    script:
        "../scripts/compare.py"
