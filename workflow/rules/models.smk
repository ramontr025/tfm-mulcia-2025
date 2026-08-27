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
    script:
        "../scripts/eval.py"


rule compare:
    input:
        metrics=expand("results/metrics/{name}.json", name=list(MODELS)),
    output:
        csv="results/comparison.csv",
        png="results/comparison.png",
    log:
        "logs/compare.log",
    script:
        "../scripts/compare.py"
