"""Estimandos de los seis bloques del protocolo.

Cada funcion devuelve estructuras puras (dict y ``DataFrame``), sin tocar ficheros:
la escritura la hace ``EDA.py`` a traves de ``io``. Las unidades y la poblacion van
en el nombre de cada campo o en la columna ``population``.

El orden de columnas de las analiticas lo fija siempre ``lab_codes`` (el del
pickle), nunca el orden del CSV de etiquetas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

DAYS_PER_YEAR = 365.25

#: Techo de la rejilla de Kaplan-Meier, en años de edad. Los eventos observados
#: terminan hacia los 85; mas alla solo queda censura, asi que no se dibuja la cola.
KM_GRID_MAX_AGE = 90.0


# ── Bloque C: cohorte y supervivencia ───────────────────────────────────────


def block_cohort(
    cohorts: dict[str, pd.DataFrame],
    sequences_by_split: dict[str, dict],
    index_age: float,
    landmarks: list[int],
) -> tuple[list[dict], dict]:
    """Tabla de cohorte por split y Kaplan-Meier con numero en riesgo.

    Returns:
        (filas de ``cohort_summary.csv``, bloque ``C`` del JSON)
    """
    from lifelines import KaplanMeierFitter

    rows: list[dict] = []
    km: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for split in ("discovery", "train", "val", "holdout"):
        frame = cohorts.get(split)
        if frame is None or not len(frame):
            continue
        duration = frame["duration"].to_numpy(dtype=float)
        event = frame["event"].to_numpy(dtype=int)
        n = int(len(frame))
        n_events = int(event.sum())
        rate = 100.0 * n_events / n if n else None
        med_follow = float(np.median(duration)) if n else None
        med_age = float(np.median(duration + index_age)) if n else None
        women = None
        seqs = sequences_by_split.get(split)
        if seqs:
            sex = np.array([float(s["sex_male"]) for s in seqs.values()])
            if sex.size:
                women = 100.0 * float(np.mean(sex == 0))
        for metric, value, unit in (
            ("n_patients", n, "paciente"),
            ("n_events", n_events, "evento"),
            ("event_rate_pct", rate, "porcentaje"),
            ("median_followup_years", med_follow, "años"),
            ("median_age_at_end_years", med_age, "años"),
            ("pct_women", women, "porcentaje"),
        ):
            rows.append(
                {
                    "split": split,
                    "unit": unit,
                    "metric": metric,
                    "value": value,
                    "n_denominator": n,
                }
            )

        summary[split] = {
            "n_patients": n,
            "n_events": n_events,
            "event_rate_pct": rate,
            "median_followup_years": med_follow,
        }
        if split == "discovery":
            continue

        kmf = KaplanMeierFitter()
        age = duration + index_age
        kmf.fit(age, event_observed=event, label=split)
        grid = np.arange(index_age, KM_GRID_MAX_AGE + 1.0, 1.0)
        surv = np.asarray(kmf.survival_function_at_times(grid).to_numpy(), dtype=float)
        ci = kmf.confidence_interval_
        ci_lo = np.interp(
            grid, ci.index.to_numpy(dtype=float), ci.iloc[:, 0].to_numpy()
        )
        ci_hi = np.interp(
            grid, ci.index.to_numpy(dtype=float), ci.iloc[:, 1].to_numpy()
        )
        at_risk = {
            str(lm): int(np.sum(duration > (lm - index_age))) for lm in landmarks
        }
        event_age = age[event == 1]
        km[split] = {
            "grid_age_years": [float(x) for x in grid],
            "km_by": "age_at_end_of_followup",
            "survival": [float(x) for x in surv],
            "ci_lower": [float(x) for x in ci_lo],
            "ci_upper": [float(x) for x in ci_hi],
            "at_risk": at_risk,
            "event_age_min": float(event_age.min()) if event_age.size else None,
            "event_age_max": float(event_age.max()) if event_age.size else None,
        }
    return rows, {"summary": summary, "km": km}


# ── Bloque 1: lo que ve el modelo en cada corte ─────────────────────────────


def block_model_view(
    samples: pd.DataFrame, landmarks: list[int]
) -> tuple[list[dict], dict]:
    """Cobertura del historial en cada landmark, sobre las muestras reales (V3)."""
    rows: list[dict] = []
    for lm in landmarks:
        subset = samples[samples["landmark_age"] == lm]
        n = int(len(subset))
        if not n:
            continue
        without = int((subset["n_labs_observed"] == 0).sum())
        with_any = subset[subset["n_labs_observed"] > 0]["n_labs_observed"]
        rows.append(
            {
                "landmark_age": lm,
                "unit": "muestra",
                "n_samples": n,
                "n_without_labs": without,
                "pct_without_labs": 100.0 * without / n,
                "median_labs_observed": float(np.median(subset["n_labs_observed"])),
                "p25_labs_observed": float(
                    np.percentile(subset["n_labs_observed"], 25)
                ),
                "p75_labs_observed": float(
                    np.percentile(subset["n_labs_observed"], 75)
                ),
                "p90_labs_observed": float(
                    np.percentile(subset["n_labs_observed"], 90)
                ),
                "median_labs_if_any": float(np.median(with_any))
                if len(with_any)
                else None,
                "p25_labs_if_any": float(np.percentile(with_any, 25))
                if len(with_any)
                else None,
                "p75_labs_if_any": float(np.percentile(with_any, 75))
                if len(with_any)
                else None,
                "median_icd_codes": float(np.median(subset["n_icd_codes"])),
                "p90_icd_codes": float(np.percentile(subset["n_icd_codes"], 90)),
                "median_timesteps": float(np.median(subset["n_timesteps"])),
                "p25_timesteps": float(np.percentile(subset["n_timesteps"], 25)),
                "p75_timesteps": float(np.percentile(subset["n_timesteps"], 75)),
            }
        )
    n_total = int(len(samples))
    without_total = int((samples["n_labs_observed"] == 0).sum()) if n_total else 0
    block = {
        "n_samples": n_total,
        "n_patients_with_valid_pair": int(samples["pid"].nunique()) if n_total else 0,
        "n_samples_without_labs": without_total,
        "pct_samples_without_labs": 100.0 * without_total / n_total
        if n_total
        else None,
        "median_labs_per_sample": float(np.median(samples["n_labs_observed"]))
        if n_total
        else None,
        "median_icd_codes_per_sample": float(np.median(samples["n_icd_codes"]))
        if n_total
        else None,
        "p90_icd_codes_per_sample": float(np.percentile(samples["n_icd_codes"], 90))
        if n_total
        else None,
    }
    return rows, block


# ── Bloque 2: estructura temporal y muestreo informativo ────────────────────


def block_timing(
    history: pd.DataFrame,
    samples: pd.DataFrame,
    cohort_train: pd.DataFrame,
    index_age: float,
    horizon: float,
    week_days: float,
    week_tolerance_days: float,
    annual_center_days: float,
) -> tuple[list[dict], dict, np.ndarray, dict[int, np.ndarray]]:
    """Δt entre timesteps, visitas por paciente y muestreo informativo por paciente.

    Returns:
        (filas de ``visits_timing.csv``, bloque ``2``, Δt en días, tasas por grupo)
    """
    rows: list[dict] = []
    n_patients = int(len(history))
    n_timesteps = int(history["n_timesteps"].sum()) if n_patients else 0
    n_visits = int(history["n_lab_visits"].sum()) if n_patients else 0
    deltas = (
        np.concatenate(history["delta_days"].to_numpy()) if n_patients else np.zeros(0)
    )
    n_pairs = int(deltas.size)

    def add(
        population: str, subset: str, unit: str, metric: str, value, n: int
    ) -> None:
        rows.append(
            {
                "population": population,
                "subset": subset,
                "unit": unit,
                "metric": metric,
                "value": value,
                "n": n,
            }
        )

    for metric, values in (
        ("lab_visits_per_patient", history["n_lab_visits"].to_numpy(dtype=float)),
        ("timesteps_per_patient", history["n_timesteps"].to_numpy(dtype=float)),
    ):
        if not values.size:
            continue
        add(
            "train_12069",
            "all",
            "paciente",
            f"{metric}_median",
            float(np.median(values)),
            n_patients,
        )
        add(
            "train_12069",
            "all",
            "paciente",
            f"{metric}_q25",
            float(np.percentile(values, 25)),
            n_patients,
        )
        add(
            "train_12069",
            "all",
            "paciente",
            f"{metric}_q75",
            float(np.percentile(values, 75)),
            n_patients,
        )
        add(
            "train_12069",
            "all",
            "paciente",
            f"{metric}_p99",
            float(np.percentile(values, 99)),
            n_patients,
        )
        add(
            "train_12069",
            "all",
            "paciente",
            f"{metric}_max",
            float(values.max()),
            n_patients,
        )

    pct_week = pct_horizon = modal_days = modal_count = annual_count = None
    if n_pairs:
        rounded = np.rint(deltas).astype(int)
        values, counts = np.unique(rounded, return_counts=True)
        modal_days = float(values[int(np.argmax(counts))])
        modal_count = int(counts.max())
        residual = np.abs(deltas - week_days * np.rint(deltas / week_days))
        pct_week = 100.0 * float(np.mean(residual <= week_tolerance_days))
        pct_horizon = 100.0 * float(np.mean(deltas > horizon * DAYS_PER_YEAR))
        annual_count = int(np.sum(np.abs(rounded - annual_center_days) <= 1))
        for metric, value in (
            (
                "delta_days_between_consecutive_timesteps_median",
                float(np.median(deltas)),
            ),
            (
                "delta_days_between_consecutive_timesteps_q25",
                float(np.percentile(deltas, 25)),
            ),
            (
                "delta_days_between_consecutive_timesteps_q75",
                float(np.percentile(deltas, 75)),
            ),
            (
                "delta_days_between_consecutive_timesteps_p99",
                float(np.percentile(deltas, 99)),
            ),
            ("delta_days_between_consecutive_timesteps_max", float(deltas.max())),
            ("pct_delta_near_multiple_of_7", pct_week),
            ("pct_delta_over_horizon", pct_horizon),
            ("modal_delta_days", modal_days),
            ("modal_delta_count", float(modal_count)),
            ("annual_mode_count", float(annual_count)),
        ):
            add("train_12069", "all", "par de timesteps", metric, value, n_pairs)

    # Muestreo informativo: tasa por paciente sobre la vista de modelo. El numerador
    # cuenta solo las visitas dentro de la ventana del denominador, de la edad índice
    # al final del seguimiento: la secuencia no se corta en el diagnóstico, y sin
    # acotarla las analíticas del tratamiento inflarían la tasa de los casos.
    duration = dict(
        zip(cohort_train["Id"].astype(str), cohort_train["duration"].astype(float))
    )
    event = dict(zip(cohort_train["Id"].astype(str), cohort_train["event"].astype(int)))
    lab_ages = dict(zip(history["pid"].astype(str), history["lab_visit_ages"]))
    rates: dict[int, list[float]] = {0: [], 1: []}
    for pid in sorted(samples["pid"].unique()):
        if pid not in duration or pid not in lab_ages:
            continue
        end_age = index_age + float(duration[pid])
        in_window = (lab_ages[pid] >= index_age) & (lab_ages[pid] <= end_age)
        denom = max(float(duration[pid]), 1e-6)
        rates[int(event.get(pid, 0))].append(int(in_window.sum()) / denom)
    r0 = np.array(rates[0], dtype=float)
    r1 = np.array(rates[1], dtype=float)
    delta = p_value = None
    if r0.size and r1.size:
        test = mannwhitneyu(r1, r0, alternative="two-sided")
        delta = 2.0 * float(test.statistic) / (r0.size * r1.size) - 1.0
        p_value = float(test.pvalue)
    for subset, values in (("y=0", r0), ("y=1", r1)):
        for name, fn in (
            ("median", np.median),
            ("q25", lambda v: np.percentile(v, 25)),
            ("q75", lambda v: np.percentile(v, 75)),
        ):
            add(
                "model_9917",
                subset,
                "paciente",
                f"lab_visits_per_year_of_followup_{name}",
                float(fn(values)) if values.size else None,
                int(values.size),
            )

    block = {
        "n_patients": n_patients,
        "timesteps_total": n_timesteps,
        "lab_visits_total": n_visits,
        "timesteps_without_labs": int(n_timesteps - n_visits),
        "pct_timesteps_without_labs": 100.0 * (n_timesteps - n_visits) / n_timesteps
        if n_timesteps
        else None,
        "n_consecutive_timestep_pairs": n_pairs,
        "pct_delta_near_multiple_of_7": pct_week,
        "modal_delta_days": modal_days,
        "modal_delta_count": modal_count,
        "annual_mode_count": annual_count,
        "pct_delta_over_horizon": pct_horizon,
        "informative_sampling": {
            "n_event0": int(r0.size),
            "n_event1": int(r1.size),
            "median_rate_event0": float(np.median(r0)) if r0.size else None,
            "median_rate_event1": float(np.median(r1)) if r1.size else None,
            "mannwhitney_p": p_value,
            "cliffs_delta": delta,
            "sign_convention": "positivo = el grupo con evento se mide mas (P(X1>X0)-P(X1<X0))",
        },
    }
    return rows, block, deltas, {0: r0, 1: r1}


# ── Bloque 3: densidad de observación ───────────────────────────────────────


def block_density(
    sequences: dict,
    lab_codes: list[str],
    labels: dict[str, dict[str, str]],
    bands: list[tuple[str, float, float]],
) -> tuple[list[dict], dict]:
    """Analíticas por visita, cobertura por analítica e intensidad por banda de edad."""
    rows: list[dict] = []
    lab_dim = len(lab_codes)

    labs_per_visit: list[np.ndarray] = []
    flat_masks: list[np.ndarray] = []
    ever: list[np.ndarray] = []
    years = np.zeros(len(bands), dtype=float)
    obs_band = np.zeros((len(bands), lab_dim), dtype=float)
    for seq in sequences.values():
        masks = np.asarray(seq["masks"], dtype=np.float32)
        times = np.asarray(seq["times"], dtype=float)
        per_step = masks.sum(axis=1)
        visit = per_step > 0
        if visit.any():
            labs_per_visit.append(per_step[visit])
            flat_masks.append(masks[visit] > 0)
        ever.append((masks > 0).any(axis=0))
        if len(times) >= 2:
            lo, hi = float(times.min()), float(times.max())
            for b, (_, a, z) in enumerate(bands):
                years[b] += max(0.0, min(hi, z) - max(lo, a))
                inside = (times >= a) & (times < z)
                if inside.any():
                    obs_band[b] += masks[inside].sum(axis=0)

    lp = np.concatenate(labs_per_visit) if labs_per_visit else np.zeros(0)
    for metric, value in (
        ("labs_per_visit_median", float(np.median(lp)) if lp.size else None),
        ("labs_per_visit_q25", float(np.percentile(lp, 25)) if lp.size else None),
        ("labs_per_visit_q75", float(np.percentile(lp, 75)) if lp.size else None),
        ("labs_per_visit_min", float(lp.min()) if lp.size else None),
        ("labs_per_visit_max", float(lp.max()) if lp.size else None),
    ):
        rows.append(
            {
                "lab_code": "",
                "lab_name_es": "",
                "group": "",
                "age_band": "all",
                "scope": "visit",
                "unit": "analítica por visita",
                "metric": metric,
                "value": value,
                "n": int(lp.size),
            }
        )

    masks_all = (
        np.concatenate(flat_masks) if flat_masks else np.zeros((0, lab_dim), bool)
    )
    n_visits = int(masks_all.shape[0])
    ever_all = np.vstack(ever) if ever else np.zeros((0, lab_dim), bool)
    n_patient = int(ever_all.shape[0])
    for idx, code in enumerate(lab_codes):
        meta = labels.get(code, {})
        pct_patients = 100.0 * float(ever_all[:, idx].mean()) if n_patient else None
        pct_visits = 100.0 * float(masks_all[:, idx].mean()) if n_visits else None
        for metric, value, unit, n in (
            ("pct_patients_ever_measured", pct_patients, "porcentaje", n_patient),
            ("pct_visits_with_lab", pct_visits, "porcentaje", n_visits),
        ):
            rows.append(
                {
                    "lab_code": code,
                    "lab_name_es": meta.get("lab_name_es", ""),
                    "group": meta.get("group", ""),
                    "age_band": "all",
                    "scope": "lab",
                    "unit": unit,
                    "metric": metric,
                    "value": value,
                    "n": n,
                }
            )
        for b, (band_name, _, _) in enumerate(bands):
            rows.append(
                {
                    "lab_code": code,
                    "lab_name_es": meta.get("lab_name_es", ""),
                    "group": meta.get("group", ""),
                    "age_band": band_name,
                    "scope": "lab",
                    "unit": "observaciones por paciente-año",
                    "metric": "observations_per_patient_year",
                    "value": obs_band[b, idx] / years[b] if years[b] > 0 else None,
                    "n": int(obs_band[b, idx]),
                }
            )
    block = {
        "lab_dim": lab_dim,
        "n_visits": n_visits,
        "n_patients": n_patient,
        "labs_per_visit": {
            "median": float(np.median(lp)) if lp.size else None,
            "mean": float(lp.mean()) if lp.size else None,
            "min": float(lp.min()) if lp.size else None,
            "max": float(lp.max()) if lp.size else None,
        },
        "bands": [name for name, _, _ in bands],
    }
    return rows, block


# ── Bloque 4: redundancia ───────────────────────────────────────────────────


def block_redundancy(
    sequences: dict,
    lab_codes: list[str],
    labels: dict[str, dict[str, str]],
    threshold: float,
) -> tuple[list[dict], dict, np.ndarray]:
    """Jaccard por visita entre analíticas, con orden jerárquico.

    Returns:
        (filas de ``lab_redundancy_pairs.csv``, bloque ``4``, matriz reordenada)
    """
    pieces = []
    for seq in sequences.values():
        masks = np.asarray(seq["masks"], dtype=np.float32)
        visit = masks.sum(axis=1) > 0
        if visit.any():
            pieces.append(masks[visit] > 0)
    flat = np.concatenate(pieces) if pieces else np.zeros((0, len(lab_codes)), bool)
    counts = flat.T.astype(float) @ flat.astype(float)
    diag = np.diag(counts).astype(float)
    denom = diag[:, None] + diag[None, :] - counts
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(denom > 0, counts / denom, 0.0)
    np.fill_diagonal(jacc, 1.0)
    dim = jacc.shape[0]

    if dim >= 3:
        distance = squareform(1.0 - jacc, checks=False)
        order = leaves_list(linkage(distance, method="average")).tolist()
    else:
        order = list(range(dim))

    rows: list[dict] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            rows.append(
                {
                    "lab_a": lab_codes[i],
                    "lab_b": lab_codes[j],
                    "jaccard": float(jacc[i, j]),
                    "flagged_ge_threshold": bool(jacc[i, j] >= threshold),
                }
            )
    block = {
        "n_visits_flat": int(flat.shape[0]),
        "matrix_shape": [int(dim), int(dim)],
        "order": [lab_codes[k] for k in order],
        "n_pairs": len(rows),
        "n_flagged": int(sum(bool(row["flagged_ge_threshold"]) for row in rows)),
        "top_pairs": sorted(rows, key=lambda r: -r["jaccard"])[:5],
    }
    reordered = jacc[np.ix_(order, order)]
    block["matrix"] = [[float(x) for x in row] for row in reordered]
    return rows, block, reordered


# ── Bloque 5: magnitudes y normalización ────────────────────────────────────


def block_values(
    sequences: dict,
    lab_stats: dict,
    lab_codes: list[str],
    labels: dict[str, dict[str, str]],
    sigma_limit: float,
    mean_tol: float,
    std_tol: float,
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Distribución de z-scores observados y contraste con ``lab_stats.json``.

    Returns:
        (filas de ``lab_value_stats.csv``, bloque ``5``, valores observados por analítica)
    """
    dim = len(lab_codes)
    buckets: list[list[np.ndarray]] = [[] for _ in range(dim)]
    for seq in sequences.values():
        masks = np.asarray(seq["masks"], dtype=np.float32) > 0
        labs = np.asarray(seq["labs"], dtype=np.float32)
        if not masks.any():
            continue
        for column_index in range(min(dim, masks.shape[1])):
            observed = masks[:, column_index]
            if observed.any():
                buckets[column_index].append(labs[observed, column_index])
    columns = [
        np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        for parts in buckets
    ]
    n_values = int(sum(column.size for column in columns))

    stat_codes = [str(c) for c in lab_stats.get("lab_codes", [])]
    raw_mean = np.asarray(lab_stats.get("mean", []), dtype=float)
    raw_std = np.asarray(lab_stats.get("std", []), dtype=float)
    order_ok = stat_codes == list(lab_codes)

    rows: list[dict] = []
    discrepancies: list[dict] = []
    for idx, code in enumerate(lab_codes):
        meta = labels.get(code, {})
        column = columns[idx] if idx < len(columns) else np.zeros(0)
        mean = float(np.mean(column)) if column.size else None
        std = float(np.std(column)) if column.size else None
        flags = []
        if mean is not None and abs(mean) > mean_tol:
            flags.append("mean_off_zero")
        if std is not None and abs(std - 1.0) > std_tol:
            flags.append("std_off_one")
        flag = ";".join(flags)
        if flags:
            discrepancies.append(
                {"lab_code": code, "flags": flags, "mean_z": mean, "std_z": std}
            )
        for metric, value, unit in (
            ("median_z", float(np.median(column)) if column.size else None, "z"),
            ("mean_z", mean, "z"),
            ("std_z", std, "z"),
            ("p1_z", float(np.percentile(column, 1)) if column.size else None, "z"),
            ("p99_z", float(np.percentile(column, 99)) if column.size else None, "z"),
            ("min_z", float(column.min()) if column.size else None, "z"),
            ("max_z", float(column.max()) if column.size else None, "z"),
            (
                "pct_abs_z_over_limit",
                100.0 * float(np.mean(np.abs(column) > sigma_limit))
                if column.size
                else None,
                "porcentaje",
            ),
        ):
            rows.append(
                {
                    "lab_code": code,
                    "lab_name_es": meta.get("lab_name_es", ""),
                    "group": meta.get("group", ""),
                    "unit": unit,
                    "metric": metric,
                    "value": value,
                    "reference_value": None,
                    "flag": flag,
                }
            )
        ref_idx = stat_codes.index(code) if code in stat_codes else None
        for metric, ref in (
            (
                "mean_raw_labstats",
                float(raw_mean[ref_idx]) if ref_idx is not None else None,
            ),
            (
                "std_raw_labstats",
                float(raw_std[ref_idx]) if ref_idx is not None else None,
            ),
        ):
            rows.append(
                {
                    "lab_code": code,
                    "lab_name_es": meta.get("lab_name_es", ""),
                    "group": meta.get("group", ""),
                    "unit": "valor crudo",
                    "metric": metric,
                    "value": None,
                    "reference_value": ref,
                    "flag": flag,
                }
            )
    block = {
        "n_values": n_values,
        "lab_codes_match_config": bool(order_ok),
        "sigma_limit": sigma_limit,
        "mean_tolerance": mean_tol,
        "std_tolerance": std_tol,
        "discrepancies": discrepancies,
    }
    values_by_lab = {code: columns[idx] for idx, code in enumerate(lab_codes)}
    return rows, block, values_by_lab
