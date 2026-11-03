"""Snakemake ``script:`` de la regla paired: comparación emparejada entre modelos.

Cada réplica r tiene su propia partición de discovery y su propia semilla de
entrenamiento, y los cinco modelos de una misma réplica comparten esa partición. Por
eso dos modelos se comparan dentro de cada réplica, sobre los mismos pacientes de
holdout. El intervalo de cada diferencia sale de un bootstrap de dos niveles: en cada
iteración se sortea una réplica y se remuestrean los pacientes de holdout (cluster
bootstrap, con todas sus filas de todos los landmarks), y se calculan los estimandos
agrupados de todos los modelos de esa réplica sobre la misma pseudopoblación. Así un
único intervalo recoge a la vez la variación de partición y entrenamiento y la de la
muestra de evaluación.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecución.

Reads:  snakemake.input.preds          results/preds/{name}.pkl y {name}_s{seed}.pkl
Writes: snakemake.output.by_replica    results/paired/by_replica.csv
        snakemake.output.differences   results/paired/differences.csv
"""

import itertools
import multiprocessing as mp
import pickle
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sksurv.util import Surv
from threadpoolctl import threadpool_limits

sys.path.append(str(Path(__file__).parent))

from metrics import (
    SCALAR_METRIC_KEYS,
    _compute_metrics,
    _patient_positions,
    _percentile_ci,
    _pool_landmark_metrics,
)

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

BASE_MODELS = list(snakemake.params.base_models)
MAIN_SEED = int(snakemake.params.main_seed)
N_BOOT = int(snakemake.params.n_boot)
BOOT_SEED = int(snakemake.params.seed)
LOWER_IS_BETTER = {"brier"}

# ── 1. Cargar las predicciones de cada ejecución: (modelo, réplica) -> datos ──
runs = {}
for fpath in snakemake.input.preds:
    stem = Path(fpath).stem
    match = re.fullmatch(r"(.+)_s(\d+)", stem)
    model, rep = (match.group(1), int(match.group(2))) if match else (stem, MAIN_SEED)
    with open(fpath, "rb") as f:
        runs[(model, rep)] = pickle.load(f)
REPLICAS = sorted({rep for _, rep in runs})
HORIZON = float(next(iter(runs.values()))["horizon"])
print(f"{len(runs)} ejecuciones, réplicas {REPLICAS}", flush=True)

# ── 2. Alinear las filas de holdout por (paciente, landmark) ─────────────────
# Cada modelo recorre los landmarks en su propio orden; tras ordenar por la clave,
# todas las ejecuciones comparten filas y posiciones, y un mismo remuestreo de
# pacientes vale para todas.
reference = None
risks = {}
for key, data in runs.items():
    ho = data["holdout"]
    row_keys = np.array([f"{i}|{int(lm)}" for i, lm in zip(ho["ids"], ho["lms"])])
    order = np.argsort(row_keys, kind="stable")
    aligned = {k: np.asarray(ho[k])[order] for k in ("durs", "evs", "lms", "ids")}
    if reference is None:
        reference = {**aligned, "row_keys": row_keys[order]}
    else:
        # mismas filas, mismo desenlace: si no, la comparación no estaría emparejada
        assert np.array_equal(row_keys[order], reference["row_keys"]), key
        assert np.array_equal(aligned["evs"], reference["evs"]), key
        assert np.allclose(aligned["durs"], reference["durs"]), key
    risks[key] = np.asarray(ho["risks"], dtype=np.float64)[order]

DURS, EVS, LMS = reference["durs"], reference["evs"], reference["lms"]
POS_BY_ID = _patient_positions(reference["ids"])
LANDMARKS = sorted({int(lm) for lm in LMS})
LM_MASKS = {lm: LMS == lm for lm in LANDMARKS}

# Referencias de train de cada ejecución y landmark (censura y Breslow), fijas
REFS = {
    key: {
        lm: (
            Surv.from_arrays(event=ref["evs"].astype(bool), time=ref["durs"]),
            ref["h0"],
            ref["maxr"],
            ref["h0_null"],
        )
        for lm, ref in data["train_refs"].items()
    }
    for key, data in runs.items()
}
del runs


def pooled_metrics(key, pos):
    """Estimando agrupado de una ejecución sobre las filas `pos` de holdout."""
    landmark_metrics = {}
    for lm in LANDMARKS:
        if lm not in REFS[key]:
            continue
        sel = pos[LM_MASKS[lm][pos]]
        if len(sel) == 0:
            continue
        y_train, h0, maxr, h0_null = REFS[key][lm]
        landmark_metrics[lm] = _compute_metrics(
            risks[key][sel],
            DURS[sel],
            EVS[sel],
            y_train,
            h0,
            maxr,
            h0_null,
            HORIZON,
            warn=lambda _msg: None,
        )
    return _pool_landmark_metrics(landmark_metrics)


def one_draw(b):
    """Una iteración del bootstrap de dos niveles: réplica al azar y pacientes."""
    rng = np.random.default_rng(BOOT_SEED + b)
    rep = REPLICAS[rng.integers(0, len(REPLICAS))]
    chosen = rng.integers(0, len(POS_BY_ID), size=len(POS_BY_ID))
    pos = np.concatenate([POS_BY_ID[c] for c in chosen])
    with threadpool_limits(1):
        return rep, {m: pooled_metrics((m, rep), pos) for m in BASE_MODELS}


def diff(a, b, metric):
    return a[metric] - b[metric]


def a_better(d, metric):
    return d < 0 if metric in LOWER_IS_BETTER else d > 0


PAIRS = list(itertools.combinations(BASE_MODELS, 2))

# ── 3. Estimación puntual por réplica: todos los pacientes de holdout ────────
all_rows = np.arange(len(DURS))
point = {
    rep: {m: pooled_metrics((m, rep), all_rows) for m in BASE_MODELS}
    for rep in REPLICAS
}
by_replica = pd.DataFrame(
    [
        {
            "model_a": a,
            "model_b": b,
            "replica": rep,
            "metric": metric,
            "diff": diff(point[rep][a], point[rep][b], metric),
        }
        for a, b in PAIRS
        for rep in REPLICAS
        for metric in SCALAR_METRIC_KEYS
    ]
)
by_replica.to_csv(snakemake.output.by_replica, index=False)
print("Estimaciones puntuales por réplica listas", flush=True)

# ── 4. Bootstrap de dos niveles, repartido entre los hilos del trabajo ───────
# Procesos con fork: heredan los datos ya cargados y no hay que serializar ni las
# predicciones ni la función, solo los resultados de cada iteración.
def run_chunk(draw_ids, queue):
    queue.put([one_draw(b) for b in draw_ids])


ctx = mp.get_context("fork")
queue = ctx.Queue()
chunks = [c for c in np.array_split(np.arange(N_BOOT), snakemake.threads) if len(c)]
procs = [ctx.Process(target=run_chunk, args=(c.tolist(), queue)) for c in chunks]
for proc in procs:
    proc.start()
draws = []
for i in range(len(procs)):
    draws.extend(queue.get())  # antes de join: si no, la cola llena bloquea
    print(f"    bootstrap: {i + 1}/{len(procs)} bloques", flush=True)
for proc in procs:
    proc.join()

rows = []
for a, b in PAIRS:
    for metric in SCALAR_METRIC_KEYS:
        reps = by_replica[
            (by_replica.model_a == a)
            & (by_replica.model_b == b)
            & (by_replica.metric == metric)
        ]["diff"].to_numpy()
        boots = np.array([diff(d[a], d[b], metric) for _, d in draws])
        finite = boots[np.isfinite(boots)]
        lo, hi = _percentile_ci(boots)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "mean_diff": float(np.nanmean(reps)),
                "sd_diff_replicas": float(np.nanstd(reps, ddof=1)) if len(reps) > 1 else np.nan,
                "frac_replicas_a_better": float(np.mean(a_better(reps, metric))),
                "ci_lo": lo,
                "ci_hi": hi,
                "p_a_better": float(np.mean(a_better(finite, metric))) if len(finite) else np.nan,
                "n_replicas": len(reps),
                "n_boot": len(finite),
            }
        )
differences = pd.DataFrame(rows)
differences.to_csv(snakemake.output.differences, index=False)
print(differences.to_string(index=False), flush=True)
