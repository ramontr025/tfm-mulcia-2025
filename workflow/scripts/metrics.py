"""Métricas compartidas por eval.py y paired.py.

Una sola implementación de las métricas por landmark y del estimando agrupado, para
que la evaluación de cada modelo y la comparación emparejada entre modelos calculen
exactamente lo mismo.
"""

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv

SCALAR_METRIC_KEYS = ["c_index", "auroc", "c_index_ipcw", "brier", "ipa"]


def _compute_metrics(
    risks_sub, durs_sub, evs_sub, y_train, h0, maxr, h0_null, horizon, warn=None
):
    """
    Calcula (c_index, auroc, c_index_ipcw, brier, ipa) sobre un subconjunto de test.
    `y_train` (censura de referencia, IPCW) y `h0`/`maxr`/`h0_null` (Breslow, fijados
    en train) NUNCA se remuestrean: solo varía el subconjunto de test, así que esta
    misma función sirve tanto para el punto estimado por landmark como para cada
    réplica de cualquiera de los dos bootstrap (por landmark o agrupado/cluster).

    Convenio de signo: TODAS las funciones de sksurv esperan que a mayor estimate,
    mayor riesgo (justo lo que devuelven los encoders), así que `risks_sub` se pasa
    SIN negar (al contrario que lifelines.concordance_index).

    Los arrays estructurados se construyen siempre con Surv.from_arrays y solo se
    captura ValueError (con aviso al log): un except genérico devolvería un 0.5/NaN
    silencioso en todos los landmarks.
    """
    if warn is None:
        warn = print
    risks_sub = np.asarray(risks_sub, dtype=np.float64)
    durs_sub = np.asarray(durs_sub, dtype=np.float64)
    evs_sub = np.asarray(evs_sub, dtype=np.float64)
    n = int(len(risks_sub))
    n_events = int(np.sum(evs_sub))  # seguimiento completo
    n_events_horizon = int(
        np.sum((durs_sub <= horizon) & (evs_sub == 1))
    )  # los del horizonte
    # Inicializamos un diccionario para guardar los resultados
    out = {
        "n": n,
        "n_events": n_events,
        "n_events_horizon": n_events_horizon,
        "c_index": np.nan,
        "auroc": np.nan,
        "c_index_ipcw": np.nan,
        "brier": np.nan,
        "brier_null": np.nan,
        "ipa": np.nan,
        # Conteos para el agrupado: Harrell exacto (conc/disc/tied) y peso de
        # pares comparables para la media ponderada del C-index IPCW.
        "concordant": 0,
        "discordant": 0,
        "tied_risk": 0,
        "n_comparable": 0,
    }
    if n == 0:
        return out
    # Construimos el array estructurado de supervivencia
    y_test = Surv.from_arrays(event=evs_sub.astype(bool), time=durs_sub)

    # C-index de Harrell, guardando los conteos para el bootstrap de C-index IPCW
    if n_events > 0:
        ci, conc, disc, tied_risk, _tied_time = concordance_index_censored(
            evs_sub.astype(bool), durs_sub, risks_sub
        )
        out["c_index"] = float(ci)
        out["concordant"] = int(conc)
        out["discordant"] = int(disc)
        out["tied_risk"] = int(tied_risk)
        out["n_comparable"] = int(conc + disc + tied_risk)
    else:
        out["c_index"] = (
            0.5  # En caso de que no haya eventos, el c-index convenimos 0.5 de C-Index (aleatorio)
        )

    # AUROC(t)
    try:
        auc_t = cumulative_dynamic_auc(y_train, y_test, risks_sub, times=[horizon])[0]
        out["auroc"] = float(
            auc_t[0]
        )  # cumulative_dynamic_auc devuelve un par (auroc(t), mean_auroc), por ello seleccionamos del primer elemento del par el primer elemento temporal (solo hemos pedido un horizonte temporal)
    except ValueError as e:
        warn(f"[warn] AUROC no definido en este subconjunto (n={n}): {e}")

    # C-index de Uno (IPCW): corrige el sesgo por censura con la referencia de train
    try:
        out["c_index_ipcw"] = float(
            concordance_index_ipcw(y_train, y_test, risks_sub)[0]
        )
    except ValueError as e:
        warn(f"[warn] C-index IPCW no definido en este subconjunto (n={n}): {e}")

    # Brier IPCW en el horizon + IPA
    # Probabilidad de supervivencia desde el Breslow fijado en train:
    try:
        surv_prob_model = np.exp(-h0 * np.exp(risks_sub - maxr))
        surv_prob_null = np.full(n, np.exp(-h0_null))

        brier_model = brier_score(y_train, y_test, surv_prob_model[:, None], [horizon])[
            1
        ][0]
        brier_null = brier_score(y_train, y_test, surv_prob_null[:, None], [horizon])[
            1
        ][0]
        out["brier"] = float(brier_model)
        out["brier_null"] = float(brier_null)
        out["ipa"] = float(1.0 - brier_model / brier_null) if brier_null > 0 else np.nan
    except ValueError as e:
        warn(f"[warn] Brier/IPA no definido en este subconjunto (n={n}): {e}")

    return out


# En este punto cada landmark posee un diccionario con las metricas y resulatdos. Ahora sacaremos las metricas agrupadas:


def _pool_landmark_metrics(landmark_metrics):
    """
    Combina las métricas de varios landmarks en un único estimando AGRUPADO (pooled /
    stratified): nunca compara pacientes de landmarks distintos entre sí.

    - c_index (Harrell): exacto. Suma los pares concordantes, discordantes y
      empatados de cada landmark y divide una sola vez al final.
    - brier: exacto como media ponderada por n, porque cada Brier por landmark ya es
      la media de sus propias filas.
    - ipa: derivado del cociente de los Brier agrupados (1 - brier_agrupado / brier_null_agrupado)
    - auroc y c_index_ipcw: media de los valores por landmark ponderada por el
      tamaño de su estrato (n y pares comparables respectivamente). Es una
      APROXIMACIÓN, no el estimador estratificado exacto: los conteos que devuelve
      concordance_index_ipcw están SIN ponderar (idénticos a los de la versión
      censurada), así que agrupar por conteos anularía la corrección IPCW.
    """
    keys = [
        lm for lm, m in landmark_metrics.items() if m["n"] > 0
    ]  # escogemos unicmaente aquellos que tengan pacientes

    conc = sum(landmark_metrics[lm]["concordant"] for lm in keys)
    disc = sum(landmark_metrics[lm]["discordant"] for lm in keys)
    tied = sum(landmark_metrics[lm]["tied_risk"] for lm in keys)
    denom = conc + disc + tied
    pooled_cindex = (
        (conc + 0.5 * tied) / denom if denom > 0 else np.nan
    )  # Sumamos 0.5 por par empatado analogamente a como se hace en sksurv

    def _wmean(metric_key, weight_key):
        """Calcula la media ponderada de una métrica según el peso dado."""
        num, den = 0.0, 0.0
        for lm in keys:
            m = landmark_metrics[lm]
            value, weight = m[metric_key], m[weight_key]
            if np.isfinite(value) and weight > 0:
                num += weight * value
                den += weight
        return (
            num / den if den > 0 else np.nan
        )  # De esta manera se pondera w_1*v_1+w_2*v_2+... / w_1+w_2+...

    brier = _wmean("brier", "n")
    brier_null = _wmean("brier_null", "n")
    # El IPA agregado NO es la media de los IPA por landmark: es un cociente, y
    # promediar cocientes no da el cociente de los promedios
    # Brier = sum(ni*B_i)/sum(ni)
    # IPA = 1 - Brier/Brier_null = 1 - sum(ni*B_i)/sum(ni)*B_null) que es justo lo que queriamos
    ipa = 1 - brier / brier_null if brier_null > 0 else np.nan

    return {
        "c_index": pooled_cindex,
        "auroc": _wmean("auroc", "n"),
        "c_index_ipcw": _wmean("c_index_ipcw", "n_comparable"),
        "brier": brier,
        "ipa": ipa,
    }


def _patient_positions(ids):
    """Mapa pid -> posiciones de fila, para remuestrear pacientes y no filas."""
    ids_arr = np.asarray(ids, dtype=object)
    uniq, inv = np.unique(ids_arr, return_inverse=True)
    pos_by_id = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    return pos_by_id


def _percentile_ci(values):
    """IC 95% por percentiles sobre las réplicas bootstrap (NaN -> excluidos)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return [None, None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
