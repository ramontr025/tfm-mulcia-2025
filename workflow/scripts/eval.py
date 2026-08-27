import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader

# Permitir importar módulos locales desde workflow/scripts
sys.path.append(str(Path(__file__).parent))

from dataset import RawSequenceDatasetWithICD, raw_pad_collate_with_icd, load_sequences
from engine import collect_raw_outputs, compute_baseline_hazard
from model import TimeAwareGRUD, SimpleLSTM

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake

# ── Redirección de logs de Snakemake ─────────────────────────────────────────
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

# 1. Cargar artefacto del modelo generado en train.py
model_artifact = snakemake.input.model_path
with open(model_artifact, "rb") as f:
    artifact = pickle.load(f)

arch_name = artifact.get("arch", "grud")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Parámetros del experimento desde Snakemake
HORIZON = float(snakemake.params.horizon)
LANDMARK_MIN = int(snakemake.params.landmark_min)
LANDMARK_MAX = int(snakemake.params.landmark_max)
INDEX_AGE = float(snakemake.params.index_age)

# 3. Cargar secuencias de los tres splits (train, val y holdout)
sequences_train, _, cohort_train = load_sequences(snakemake.input.train)
sequences_val, _, cohort_val = load_sequences(snakemake.input.val)
sequences_holdout, _, cohort_holdout = load_sequences(snakemake.input.holdout)

# ── Rama de evaluación para Cox Baseline (tabular / LOCF) ──
if arch_name == "cox":
    from cox import evaluate_cox

    model_name = getattr(snakemake.wildcards, "name", "cox")
    results_json = evaluate_cox(
        artifact=artifact,
        sequences_train=sequences_train,
        cohort_train=cohort_train,
        sequences_val=sequences_val,
        cohort_val=cohort_val,
        sequences_holdout=sequences_holdout,
        cohort_holdout=cohort_holdout,
        landmark_min=LANDMARK_MIN,
        landmark_max=LANDMARK_MAX,
        index_age=INDEX_AGE,
        horizon=HORIZON,
        model_name=model_name,
        device=device,
    )

    with open(snakemake.output[0], "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(
        f"[{model_name}] Evaluación Cox completada. Val C-Index medio: {results_json['c_index']['val']['mean_c_index']:.4f}",
        flush=True,
    )
    sys.exit(0)

# ── Rama de evaluación para Modelos Secuenciales (GRU-D) ──
lab_dim = artifact["lab_dim"]
hidden_dim = artifact["hidden_dim"]
vocab_size = artifact["vocab_size"]
embed_dim = artifact["embed_dim"]
mean_x = artifact["mean_x"]
code_to_idx = artifact["code_to_idx"]
state_dict = artifact["state_dict"]


def build_model(
    arch, input_dim, vocab_size, mean_x, embed_dim, hidden_dim, dropout=0.2
):
    if arch == "grud":
        return TimeAwareGRUD(
            input_dim=input_dim,
            vocab_size=vocab_size,
            mean_x=mean_x,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif arch == "lstm":
        return SimpleLSTM(
            input_dim=input_dim,
            vocab_size=vocab_size,
            mean_x=mean_x,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")


model = build_model(
    arch=arch_name,
    input_dim=lab_dim,
    vocab_size=vocab_size,
    mean_x=mean_x,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    dropout=0.2,
).to(device)

model.load_state_dict(state_dict)
model.eval()

# 5. Evaluar C-index a través de todos los landmarks (45 a 70) para train, val y holdout
landmark_times = list(range(LANDMARK_MIN, LANDMARK_MAX + 1))
metrics_by_split = {}

for split_name, seqs, cohort in [
    ("train", sequences_train, cohort_train),
    ("val", sequences_val, cohort_val),
    ("holdout", sequences_holdout, cohort_holdout),
]:
    ds_landmarks = RawSequenceDatasetWithICD(
        seqs,
        cohort,
        landmark_ages=landmark_times,
        index_age=INDEX_AGE,
        code_to_idx=code_to_idx,
        max_events=200,
        age_weight_gamma=1.0,
    )
    loader = DataLoader(
        ds_landmarks,
        batch_size=128,
        shuffle=False,
        collate_fn=raw_pad_collate_with_icd,
    )
    risks, durs, evs, lms = collect_raw_outputs(loader, model, device)

    c_indices_by_lm = {}
    for lm in landmark_times:
        idx = lms == lm
        if not np.any(idx):
            continue
        n_samples = int(np.sum(idx))
        n_events = int(np.sum(evs[idx]))
        if n_events > 0:
            ci_lm = concordance_index(durs[idx], -risks[idx], evs[idx].astype(bool))
        else:
            ci_lm = 0.5
        c_indices_by_lm[str(lm)] = {
            "n_samples": n_samples,
            "n_events": n_events,
            "c_index": float(ci_lm),
        }
    mean_ci = (
        float(np.mean([item["c_index"] for item in c_indices_by_lm.values()]))
        if c_indices_by_lm
        else 0.0
    )

    metrics_by_split[split_name] = {
        "mean_c_index": mean_ci,
        "landmarks": c_indices_by_lm,
    }

# 6. Estimar el riesgo base de Breslow H0 en train para el landmark 55
ds_train_55 = RawSequenceDatasetWithICD(
    sequences_train,
    cohort_train,
    landmark_ages=[55],
    index_age=INDEX_AGE,
    code_to_idx=code_to_idx,
    max_events=200,
    age_weight_gamma=1.0,
)
train_loader_55 = DataLoader(
    ds_train_55,
    batch_size=128,
    shuffle=False,
    collate_fn=raw_pad_collate_with_icd,
)
train_risks, train_durs, train_evs, _ = collect_raw_outputs(
    train_loader_55, model, device
)

H0, max_r = compute_baseline_hazard(train_risks, train_durs, train_evs, HORIZON)

# 7. Evaluar calibración en el landmark 55 para val y holdout
calibration_results = {}
for split_name, seqs, cohort in [
    ("val", sequences_val, cohort_val),
    ("holdout", sequences_holdout, cohort_holdout),
]:
    ds_eval_55 = RawSequenceDatasetWithICD(
        seqs,
        cohort,
        landmark_ages=[55],
        index_age=INDEX_AGE,
        code_to_idx=code_to_idx,
        max_events=200,
        age_weight_gamma=1.0,
    )
    eval_loader_55 = DataLoader(
        ds_eval_55,
        batch_size=128,
        shuffle=False,
        collate_fn=raw_pad_collate_with_icd,
    )
    eval_risks, eval_durs, eval_evs, _ = collect_raw_outputs(
        eval_loader_55, model, device
    )

    shifted_eval = eval_risks - max_r
    pred_prob = 1.0 - np.exp(-H0 * np.exp(shifted_eval))
    obs_event_horizon = ((eval_durs <= HORIZON) & (eval_evs == 1)).astype(float)
    mean_pred = float(np.mean(pred_prob))
    mean_obs = float(np.mean(obs_event_horizon))

    calibration_results[split_name] = {
        "mean_pred_prob": mean_pred,
        "mean_obs_rate": mean_obs,
    }

# 8. Exportar JSON de métricas en snakemake.output[0]
model_name = getattr(snakemake.wildcards, "name", "model")
results_json = {
    "model_name": model_name,
    "c_index": metrics_by_split,
    "landmark_55_calibration": {
        "H0_train": float(H0),
        "max_r_train": float(max_r),
        "splits": calibration_results,
    },
}

with open(snakemake.output[0], "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

print(
    f"[{model_name}] Evaluación completada. Val C-Index medio: {metrics_by_split['val']['mean_c_index']:.4f}",
    flush=True,
)
