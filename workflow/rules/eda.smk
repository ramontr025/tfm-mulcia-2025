# ─────────────────────────────────────────────────────────────────────────────
# Stage 1.5: EDA del tensor clinico longitudinal.
#
# Una regla sola que lee el pickle de secuencias de train y produce un
# directorio results/eda/ con figuras + un resumen JSON. El input es
# results/train/sequences.pkl (el que consumen los modelos), asi el EDA
# describe justo lo que entra al pipeline, no los crudos.
#
# Filtro de modelo: el script pasa las secuencias por RawSequenceDatasetWithICD
# (landmark truncation + max_events) y se queda con los pacientes que generan
# al menos una muestra valida. Asi el EDA no cuenta secuencias que el modelo
# descarta. Los hiperparametros de ese filtro (index_age, landmark_min/max,
# max_events, age_weight_gamma) llegan de las mismas claves de config que usa
# `train`, para que el EDA y el entrenamiento esten siempre sincronizados.
#
# Salida: directory("results/eda") con:
#   trajectories.png, sparsity_visits.png, observability_load.png,
#   cooccurrence.png, global_sparsity_heatmap.png, informative_sampling.png,
#   tsne_landmarks.png, eda_summary.json
# ─────────────────────────────────────────────────────────────────────────────


rule eda:
    input:
        sequences="results/train/sequences.pkl",
        dataset="workflow/scripts/dataset.py",
    output:
        directory("results/eda"),
    log:
        "logs/eda.log",
    params:
        landmark_age_min=config["eda"]["landmark_age_min"],
        landmark_age_max=config["eda"]["landmark_age_max"],
        window_half_width=config["eda"]["window_half_width"],
        batch_size=config["eda"]["batch_size"],
        max_gap_to_connect=config["eda"]["max_gap_to_connect"],
        random_state=config["eda"]["random_state"],
        tsne_perplexity=config["eda"]["tsne_perplexity"],
        tsne_n_iter=config["eda"]["tsne_n_iter"],
        knn_n_neighbors=config["eda"]["knn_n_neighbors"],
        pca_threshold_dim=config["eda"]["pca_threshold_dim"],
        index_age=config["cohort"]["index_age"],
        landmark_min=config["cohort"]["landmark_min"],
        landmark_max=config["cohort"]["landmark_max"],
        max_events=config.get("eda", {}).get(
            "max_seq_len", config.get("training", {}).get("max_seq_len", 200)
        ),
        age_weight_gamma=config.get("eda", {}).get(
            "age_weight_gamma", config.get("training", {}).get("age_weight_gamma", 1.0)
        ),
    script:
        "../scripts/EDA.py"
