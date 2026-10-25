"""Las tres vistas de la historia (C2).

- **V1, historia completa**: las secuencias del pickle, tal cual. Es la vista de los
  bloques 2 a 5: cada paciente entra una vez y no hay pseudo-replica.
- **V2, historia en el corte**: la secuencia truncada al landmark por
  ``RawSequenceDatasetWithICD``. No se reimplementa el truncado.
- **V3, muestra del modelo**: las mismas muestras de V2 mas el vocabulario de codigos,
  el landmark y la duracion. Es la vista del bloque 1 y de las cifras de muestra.

La unica logica que se replica del modelado es el vocabulario ``code_to_idx`` de
``train.py`` (sorted -> i+1), que se necesita para contar codigos por muestra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset import RawSequenceDatasetWithICD, load_sequences  # noqa: E402


def load_history(pkl_path: Path) -> tuple[dict, list[str], pd.DataFrame]:
    """V1 desde el pickle de secuencias: ``(sequences, lab_codes, cohort)``."""
    return load_sequences(Path(pkl_path))


def build_code_to_idx(sequences: dict) -> dict[str, int]:
    """Vocabulario ICD del entrenamiento: ``{code: i+1}`` sobre el codigo ordenado.

    Es la replica exacta de ``train.py`` (sorted -> i+1). Se documenta aqui porque
    es el unico punto donde el EDA depende del vocabulario del modelo.
    """
    codes: set[str] = set()
    for seq in sequences.values():
        for step in seq["codes"]:
            codes.update(step)
    return {code: i + 1 for i, code in enumerate(sorted(codes))}


def build_model_view(
    sequences: dict,
    cohort: pd.DataFrame,
    landmarks: list[int],
    index_age: float,
    max_events: int,
    age_weight_gamma: float,
) -> RawSequenceDatasetWithICD:
    """V2 + V3: el dataset real que consume el entrenamiento."""
    return RawSequenceDatasetWithICD(
        sequences=sequences,
        cohort_df=cohort,
        landmark_ages=landmarks,
        index_age=index_age,
        code_to_idx=build_code_to_idx(sequences),
        max_events=max_events,
        age_weight_gamma=age_weight_gamma,
    )


def sample_table(dataset: RawSequenceDatasetWithICD) -> pd.DataFrame:
    """Una fila por muestra del modelo con lo que de verdad ve la red.

    Columnas:
        pid, landmark_age, event, n_timesteps, n_timesteps_with_labs,
        n_labs_observed, n_icd_codes, duration
    """
    rows = []
    for sample in dataset.samples:
        masks = np.asarray(sample["masks"], dtype=np.float32)
        per_step = masks.sum(axis=1)
        rows.append(
            {
                "pid": sample["pid"],
                "landmark_age": float(sample["landmark_age"]),
                "event": int(sample["event"]),
                "n_timesteps": int(len(sample["times"])),
                "n_timesteps_with_labs": int((per_step > 0).sum()),
                "n_labs_observed": int(per_step.sum()),
                "n_icd_codes": int(sum(len(step) for step in sample["codes_idx"])),
                "duration": float(sample["duration"]),
            }
        )
    return pd.DataFrame(rows)


def history_table(sequences: dict) -> pd.DataFrame:
    """Una fila por paciente de V1: visitas con analitica, timesteps y Δt.

    ``n_lab_visits`` son los timesteps con al menos una analitica observada;
    ``n_timesteps`` incluye los que solo llevan diagnostico (25,7 % en train).
    """
    rows = []
    for pid, seq in sequences.items():
        times = np.asarray(seq["times"], dtype=float)
        masks = np.asarray(seq["masks"], dtype=np.float32)
        per_step = masks.sum(axis=1)
        with_labs = per_step > 0
        deltas = np.diff(times) if len(times) >= 2 else np.zeros(0)
        rows.append(
            {
                "pid": pid,
                "n_timesteps": int(len(times)),
                "n_lab_visits": int(with_labs.sum()),
                "n_labs_observed": int(per_step.sum()),
                "first_age": float(times.min()) if len(times) else np.nan,
                "last_age": float(times.max()) if len(times) else np.nan,
                "delta_days": deltas * 365.25,
            }
        )
    return pd.DataFrame(rows)


def funnel(
    cohorts: dict[str, pd.DataFrame],
    sequences: dict,
    dataset: RawSequenceDatasetWithICD,
) -> list[dict]:
    """Embudo de la muestra, con la etapa cruda declarada como no disponible.

    ``data/`` no esta en el arbol de trabajo, asi que el recuento de pacientes del
    pool crudo no se calcula: se escribe ``None`` con el motivo. No se lee
    ``appendix/data`` porque no es un artefacto del DAG y haria irreproducible la
    regla en otro checkout.
    """
    n_discovery = len(cohorts.get("discovery", []))
    n_train = len(cohorts.get("train", []))
    n_val = len(cohorts.get("val", []))
    n_holdout = len(cohorts.get("holdout", []))
    pids = {sample["pid"] for sample in dataset.samples}
    stages = [
        (
            0,
            "pool_crudo",
            "paciente",
            None,
            "raw no disponible: data/ fuera del arbol; el recuento publicado esta en 2.2",
        ),
        (1, "cohorte_discovery", "paciente", n_discovery, "cohort.parquet"),
        (2, "cohorte_train", "paciente", n_train, "split de discovery"),
        (3, "cohorte_val", "paciente", n_val, "split de discovery"),
        (
            4,
            "secuencias_train",
            "paciente",
            len(sequences),
            "results/train/sequences.pkl",
        ),
        (5, "pacientes_con_par_valido", "paciente", len(pids), "filtro de landmarks"),
        (
            6,
            "pares_paciente_landmark",
            "par paciente-landmark",
            len(dataset.samples),
            "dataset.samples",
        ),
    ]
    rows = []
    previous = None
    for order, stage, unit, n, note in stages:
        pct = None
        if n is not None and previous:
            pct = 100.0 * n / previous
        rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "unit": unit,
                "n": n,
                "pct_of_previous": pct,
                "note": note,
                "population": "discovery",
            }
        )
        if n:
            previous = n
    rows.append(
        {
            "stage_order": 7,
            "stage": "cohorte_holdout",
            "unit": "paciente",
            "n": n_holdout,
            "pct_of_previous": None,
            "note": "pool aislado; no pasa por el filtro de landmarks",
            "population": "holdout",
        }
    )
    return rows
