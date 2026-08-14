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
        hidden_dim=config["training"]["hidden_dim"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        event_oversample=config["training"]["event_oversample"],
        max_events=config["training"]["max_seq_len"],
        age_weight_gamma=config["training"]["age_weight_gamma"],
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
    script:
        "../scripts/eval.py"
