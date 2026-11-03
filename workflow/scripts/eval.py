import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from sksurv.util import Surv
from torch.utils.data import DataLoader

# Permitir importar módulos locales desde workflow/scripts
sys.path.append(str(Path(__file__).parent))

from dataset import RawSequenceDatasetWithICD, raw_pad_collate_with_icd, load_sequences
from engine import collect_raw_outputs, compute_baseline_hazard
from model import TimeAwareGRUD, SimpleLSTM
from metrics import (
    SCALAR_METRIC_KEYS,
    _compute_metrics,
    _patient_positions,
    _percentile_ci,
    _pool_landmark_metrics,
)

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake

# ── Redirección de logs de Snakemake ─────────────────────────────────────────
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

SPLIT_NAMES = ["train", "val", "holdout"]


def _bootstrap_ci(
    risks_sub,
    durs_sub,
    evs_sub,
    ids_sub,
    y_train,
    h0,
    maxr,
    h0_null,
    horizon,
    n_boot,
    seed,
):
    """
    IC 95% (percentil) por bootstrap A NIVEL DE PACIENTE, para UN landmark.
    Dentro de un landmark cada fila ya es un paciente distinto, pero se agrupa por
    pid de forma explícita para que la función siga siendo correcta si eso cambia.
    La referencia de censura (y_train) y el Breslow (h0/maxr/h0_null) son los de
    train y NUNCA se remuestrean.
    """
    rng = np.random.default_rng(seed)
    pos_by_id = _patient_positions(
        ids_sub
    )  # Para cada paciente obtenemos las posiciones que ocupa en el subconjunto
    n_ids = len(pos_by_id)
    boots = {k: [] for k in SCALAR_METRIC_KEYS}
    for _ in range(n_boot):
        chosen = rng.integers(0, n_ids, size=n_ids)  # pacientes, no filas
        pos = np.concatenate([pos_by_id[c] for c in chosen])
        m = _compute_metrics(
            risks_sub[pos],
            durs_sub[pos],
            evs_sub[pos],
            y_train,
            h0,
            maxr,
            h0_null,
            horizon,
            warn=lambda _msg: None,  # NaN esperables al remuestrear; no inundar el log
        )
        for k in SCALAR_METRIC_KEYS:
            boots[k].append(
                m[k]
            )  # Entran en cada lista de metrica, tantos resultados como n_boot
    return {
        k: _percentile_ci(v) for k, v in boots.items()
    }  # se devuelve los percentiles correspondientes a cada metrica


def _cluster_bootstrap_pooled(
    risks,
    durs,
    evs,
    lms_arr,
    ids,
    y_train_by_lm,
    h0_by_lm,
    maxr_by_lm,
    h0_null_by_lm,
    horizon,
    landmark_times,
    n_boot,
    seed,
):
    """
    IC 95% (percentil) del estimando AGRUPADO, por CLUSTER bootstrap a nivel de
    paciente. Los landmarks comparten pacientes (están anidados: quien es elegible
    en el landmark 70 ya lo era en el 45), así que promediar las 26 métricas como
    si fueran estudios independientes sobreestima la información. En cada réplica
    se remuestrea el paciente UNA sola vez y se arrastran todas sus filas, de todos
    los landmarks, juntas; remuestrear cada landmark por separado rompería la
    correlación intra-paciente y daría un IC demasiado estrecho.
    """
    rng = np.random.default_rng(seed)
    pos_by_id = _patient_positions(ids)
    n_ids = len(pos_by_id)
    lm_masks = {lm: (lms_arr == lm) for lm in landmark_times}
    boots = {k: [] for k in SCALAR_METRIC_KEYS}
    for b in range(n_boot):
        chosen = rng.integers(0, n_ids, size=n_ids)  # pacientes, no filas
        pos = np.concatenate([pos_by_id[c] for c in chosen])
        landmark_metrics = {}
        for lm in landmark_times:
            if lm not in y_train_by_lm:
                continue
            sel = pos[lm_masks[lm][pos]]
            if len(sel) == 0:
                continue
            landmark_metrics[lm] = _compute_metrics(
                risks[sel],
                durs[sel],
                evs[sel],
                y_train_by_lm[lm],
                h0_by_lm[lm],
                maxr_by_lm[lm],
                h0_null_by_lm[lm],
                horizon,
                warn=lambda _msg: None,
            )
        pooled = _pool_landmark_metrics(landmark_metrics)
        for k in SCALAR_METRIC_KEYS:
            boots[k].append(
                pooled[k]
            )  # Para cada boot, almacenas cada metrica agregada
        if (b + 1) % 10 == 0:
            print(f"    cluster bootstrap {b + 1}/{n_boot}", flush=True)
    return {
        k: _percentile_ci(v) for k, v in boots.items()
    }  # Se devuelve los percentiles de los boot


def _safe(x):
    """float finito para JSON; NaN/inf -> null."""
    return float(x) if x is not None and np.isfinite(x) else None


# ── 1. Cargar artefacto del modelo generado en train.py ──────────────────────
model_artifact = snakemake.input.model_path
with open(model_artifact, "rb") as f:
    artifact = pickle.load(f)

arch_name = artifact.get("arch", "grud")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = getattr(snakemake.wildcards, "name", "model")

# Hilos de este trabajo: los que reserva Snakemake (threads:). Sin limite, torch y
# BLAS abren uno por nucleo y los trabajos en paralelo se pisan la CPU.
torch.set_num_threads(snakemake.threads)
threadpool_limits(snakemake.threads)

# ── 2. Parámetros del experimento desde Snakemake ────────────────────────────
HORIZON = float(snakemake.params.horizon)
LANDMARK_MIN = int(snakemake.params.landmark_min)
LANDMARK_MAX = int(snakemake.params.landmark_max)
INDEX_AGE = float(snakemake.params.index_age)
BOOT_N = int(snakemake.params.boot_n)
BOOT_N_POOLED = int(snakemake.params.boot_n_pooled)
BOOT_SEED = int(snakemake.params.boot_seed)
BOOT_LANDMARKS = snakemake.params.boot_landmarks

# ── 3. Cargar secuencias de los tres splits (train, val y holdout) ───────────
sequences = {}
cohorts = {}
for split_name, input_key in [
    ("train", "train"),
    ("val", "val"),
    ("holdout", "holdout"),
]:
    seqs, _, cohort = load_sequences(getattr(snakemake.input, input_key))
    sequences[split_name] = seqs
    cohorts[split_name] = cohort

landmark_times = list(range(LANDMARK_MIN, LANDMARK_MAX + 1))

# ── 4. Despachador único de predicción: Cox y los secuenciales devuelven lo ──
# ── mismo: (risks, durs, evs, lms, ids), como engine.collect_raw_outputs. ────
if arch_name == "cox":
    from cox import predict_cox_landmarks

    def predict(seqs, cohort):
        return predict_cox_landmarks(
            artifact, seqs, cohort, landmark_times, INDEX_AGE, device
        )
elif arch_name == "strats":
    from engine import predict_STraTS

    def predict(seqs, cohort):
        return predict_STraTS(artifact, seqs, cohort, landmark_times, INDEX_AGE, device)

else:
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

    def predict(seqs, cohort):
        ds = RawSequenceDatasetWithICD(
            seqs,
            cohort,
            landmark_ages=landmark_times,
            index_age=INDEX_AGE,
            code_to_idx=code_to_idx,
            max_events=200,
            age_weight_gamma=1.0,
        )
        loader = DataLoader(
            ds,
            batch_size=128,
            shuffle=False,
            collate_fn=raw_pad_collate_with_icd,
        )
        return collect_raw_outputs(loader, model, device)


# ── 5. Predicciones crudas por split sobre TODOS los landmarks ───────────────
preds = {}
for split_name in SPLIT_NAMES:
    print(f"[{model_name}] Predicciones en {split_name}...", flush=True)
    risks, durs, evs, lms, ids = predict(sequences[split_name], cohorts[split_name])
    preds[split_name] = (
        np.asarray(risks, dtype=np.float64),
        np.asarray(durs, dtype=np.float64),
        np.asarray(evs, dtype=np.float64),
        np.asarray(lms),
        np.asarray(ids, dtype=object),
    )
    print(
        f"[{model_name}] {split_name}: {len(risks)} filas-landmark, "
        f"{len(np.unique(ids))} pacientes únicos",
        flush=True,
    )

# ── 6. Referencias fijadas en TRAIN por landmark: distribución de censura ────
# ── (IPCW) y hazard base de Breslow (modelo y nulo). NUNCA se recalculan ─────
# ── sobre val/holdout (fuga de datos). ───────────────────────────────────────
risks_tr, durs_tr, evs_tr, lms_tr, _ = preds["train"]
y_train_by_lm, h0_by_lm, maxr_by_lm, h0_null_by_lm = {}, {}, {}, {}
for lm in landmark_times:
    idx = lms_tr == lm
    if not np.any(idx):
        continue
    y_train_by_lm[lm] = Surv.from_arrays(
        event=evs_tr[idx].astype(bool), time=durs_tr[idx]
    )
    h0, maxr = compute_baseline_hazard(
        risks_tr[idx], durs_tr[idx], evs_tr[idx], HORIZON
    )
    h0_by_lm[lm], maxr_by_lm[lm] = h0, maxr
    # Modelo nulo sin covariables: riesgos fijados a cero -> incidencia basal de train
    h0_null, _ = compute_baseline_hazard(
        np.zeros(int(np.sum(idx))), durs_tr[idx], evs_tr[idx], HORIZON
    )
    h0_null_by_lm[lm] = h0_null

# ── 7. Métricas por landmark y estimando AGRUPADO (pooled) por split ─────────
landmark_metrics_by_split = {}
pooled_by_split = {}
for split_name in SPLIT_NAMES:
    risks, durs, evs, lms, _ids = preds[split_name]
    lm_metrics = {}
    for lm in landmark_times:
        idx = lms == lm
        if not np.any(idx) or lm not in y_train_by_lm:
            continue
        lm_metrics[lm] = _compute_metrics(
            risks[idx],
            durs[idx],
            evs[idx],
            y_train_by_lm[lm],
            h0_by_lm[lm],
            maxr_by_lm[lm],
            h0_null_by_lm[lm],
            HORIZON,
        )
    landmark_metrics_by_split[split_name] = lm_metrics
    pooled_by_split[split_name] = _pool_landmark_metrics(lm_metrics)
    print(
        f"[{model_name}] {split_name}: pooled "
        + ", ".join(
            f"{k}={_safe(v):.4f}" if _safe(v) is not None else f"{k}=NaN"
            for k, v in pooled_by_split[split_name].items()
        ),
        flush=True,
    )

# ── 8. Intervalos de confianza 95% (solo val y holdout: en train el conjunto ─
# ── de test coincide con la referencia de censura y el IC no significaría ────
# ── nada). Bootstrap por landmark y cluster bootstrap del estimando ──────────
# ── agrupado, ambos a nivel de paciente. ─────────────────────────────────────
if isinstance(BOOT_LANDMARKS, str) and BOOT_LANDMARKS == "all":
    boot_landmark_times = list(landmark_times)
else:
    boot_landmark_times = [int(x) for x in BOOT_LANDMARKS]

bootstrap_ci = {split_name: {} for split_name in SPLIT_NAMES}
pooled_ci = {split_name: {} for split_name in SPLIT_NAMES}

for split_name in ["val", "holdout"]:
    risks, durs, evs, lms, ids = preds[split_name]
    print(
        f"[{model_name}] Bootstrap por landmark en {split_name} "
        f"({BOOT_N} réplicas x {len(boot_landmark_times)} landmarks)...",
        flush=True,
    )
    for lm in boot_landmark_times:
        idx = lms == lm
        if not np.any(idx) or lm not in y_train_by_lm:
            continue
        bootstrap_ci[split_name][str(lm)] = _bootstrap_ci(
            risks[idx],
            durs[idx],
            evs[idx],
            ids[idx],
            y_train_by_lm[lm],
            h0_by_lm[lm],
            maxr_by_lm[lm],
            h0_null_by_lm[lm],
            HORIZON,
            BOOT_N,
            BOOT_SEED + int(lm),
        )
    print(
        f"[{model_name}] Cluster bootstrap agrupado en {split_name} "
        f"({BOOT_N_POOLED} réplicas x {len(landmark_times)} landmarks)...",
        flush=True,
    )
    pooled_ci[split_name] = _cluster_bootstrap_pooled(
        risks,
        durs,
        evs,
        lms,
        ids,
        y_train_by_lm,
        h0_by_lm,
        maxr_by_lm,
        h0_null_by_lm,
        HORIZON,
        landmark_times,
        BOOT_N_POOLED,
        BOOT_SEED + 1000,
    )

# ── 9. Calibración en el landmark 55 (Breslow fijado en train) ───────────────
calibration_results = {}
if 55 in h0_by_lm:
    H0_55, max_r_55 = h0_by_lm[55], maxr_by_lm[55]
    for split_name in ["val", "holdout"]:
        risks, durs, evs, lms, _ids = preds[split_name]
        idx = lms == 55
        if not np.any(idx):
            continue
        pred_prob = 1.0 - np.exp(-H0_55 * np.exp(risks[idx] - max_r_55))
        obs_event_horizon = ((durs[idx] <= HORIZON) & (evs[idx] == 1)).astype(float)
        calibration_results[split_name] = {
            "mean_pred_prob": float(np.mean(pred_prob)),
            "mean_obs_rate": float(np.mean(obs_event_horizon)),
        }
else:
    H0_55, max_r_55 = 0.0, 0.0

# ── 10. Exportar JSON de métricas en snakemake.output.metrics ────────────────
# Ojo con el significado de mean_*: ya no es la media por landmark, es el
# estimando AGRUPADO. El nombre se mantiene por compatibilidad con compare.py
# y con los JSON anteriores.
results_json = {"model_name": model_name}
for metric_key in SCALAR_METRIC_KEYS:
    results_json[metric_key] = {
        split_name: {
            f"mean_{metric_key}": _safe(pooled_by_split[split_name][metric_key]),
            "landmarks": {
                str(lm): {
                    "n_samples": m["n"],
                    "n_events": m["n_events"],
                    "n_events_horizon": m["n_events_horizon"],
                    metric_key: _safe(m[metric_key]),
                }
                for lm, m in landmark_metrics_by_split[split_name].items()
            },
        }
        for split_name in SPLIT_NAMES
    }
results_json["bootstrap_ci"] = bootstrap_ci
results_json["pooled_ci"] = pooled_ci
results_json["landmark_55_calibration"] = {
    "H0_train": float(H0_55),
    "max_r_train": float(max_r_55),
    "splits": calibration_results,
}

with open(snakemake.output.metrics, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

# ── 11. Predicciones de holdout por paciente y landmark, con las referencias ──
# ── de train de cada landmark. paired.py las usa para comparar modelos sobre ──
# ── los mismos pacientes, sin volver a cargar ningún modelo. ─────────────────
risks_ho, durs_ho, evs_ho, lms_ho, ids_ho = preds["holdout"]
train_refs = {
    lm: {
        "durs": durs_tr[lms_tr == lm],
        "evs": evs_tr[lms_tr == lm],
        "h0": float(h0_by_lm[lm]),
        "maxr": float(maxr_by_lm[lm]),
        "h0_null": float(h0_null_by_lm[lm]),
    }
    for lm in y_train_by_lm
}
with open(snakemake.output.preds, "wb") as f:
    pickle.dump(
        {
            "model_name": model_name,
            "horizon": HORIZON,
            "holdout": {
                "risks": risks_ho,
                "durs": durs_ho,
                "evs": evs_ho,
                "lms": lms_ho,
                "ids": ids_ho,
            },
            "train_refs": train_refs,
        },
        f,
    )

print(
    f"[{model_name}] Evaluación completada. Holdout pooled: "
    + ", ".join(
        f"{k}={_safe(v):.4f}" if _safe(v) is not None else f"{k}=NaN"
        for k, v in pooled_by_split["holdout"].items()
    ),
    flush=True,
)
