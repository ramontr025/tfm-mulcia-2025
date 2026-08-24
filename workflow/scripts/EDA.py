"""Snakemake ``script:`` for eda (Stage 1.5).

Analisis Exploratorio de Datos (EDA) sobre el tensor clinico longitudinal de
pruebas de laboratorio. Lee ``results/train/sequences.pkl`` y produce, en
``results/eda/``, un conjunto de figuras PNG + un resumen JSON.

Filtro de modelo: las secuencias se pasan por ``RawSequenceDatasetWithICD``
(landmark truncation + ``max_events``) para analizar SOLO los pacientes que
entraran en el modelo, no todos los crudos. Esto evita que el EDA describa
secuencias que el pipeline descarta.

Tensors de trabajo (construidos a nivel PACIENTE, no por muestra landmark):
    X_val   (N, T_max, D)   valores z-score de las analiticas
    X_mask  (N, T_max, D)   1 = observado, 0 = ausencia/padding
    X_time  (N, T_max)      edad del paciente (anos) en cada timestep
    y_event (N,)            etiqueta clinica (1 = evento, 0 = no evento)
    lab_names (D,)          codigos LOINC de cada columna de labs

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no se
importa: el bloque TYPE_CHECKING solo le da tipo al linter/IDE).

Reads:  snakemake.input.sequences   results/train/sequences.pkl
        snakemake.input.dataset     workflow/scripts/dataset.py (dependencia)
Writes: snakemake.output[0]         directorio results/eda/ con:
            trajectories.png, sparsity_visits.png, observability_load.png,
            cooccurrence.png, global_sparsity_heatmap.png,
            informative_sampling.png, tsne_landmarks.png, eda_summary.json
"""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib

matplotlib.use("Agg")  # backend no interactivo para correr headless en snakemake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE

if TYPE_CHECKING:
    # Solo para el linter/IDE. En runtime TYPE_CHECKING es False y `snakemake`
    # es el objeto inyectado por Snakemake.
    from snakemake.iocontainers import snakemake


# ── Helpers de construccion del tensor a nivel paciente ───────────────────────


def build_patient_tensors(
    sequences: dict, cohort: pd.DataFrame, lab_codes: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Apila las secuencias irregulares en tensores padded (N, T_max, D).

    Una fila por paciente (no por landmark): conserva TODAS las visitas del
    paciente. El padding rellena con 0 y su mascara asociada tambien es 0, asi
    que un timestep padding tiene ``sum(X_mask[i, t, :]) == 0`` y no cuenta
    como visita.

    Returns:
        (X_val, X_mask, X_time, y_event, pids)
    """
    event_map = {str(r.Id): float(r.event) for r in cohort.itertuples(index=False)}
    pids = [str(pid) for pid in cohort["Id"] if str(pid) in sequences]
    if not pids:
        return (
            np.zeros((0, 0, len(lab_codes)), dtype=np.float32),
            np.zeros((0, 0, len(lab_codes)), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            [],
        )
    d = len(lab_codes)
    t_max = max(len(sequences[pid]["times"]) for pid in pids)
    n = len(pids)
    x_val = np.zeros((n, t_max, d), dtype=np.float32)
    x_mask = np.zeros((n, t_max, d), dtype=np.float32)
    x_time = np.zeros((n, t_max), dtype=np.float32)
    y_event = np.zeros((n,), dtype=np.int32)
    for i, pid in enumerate(pids):
        seq = sequences[pid]
        ti = len(seq["times"])
        x_val[i, :ti] = seq["labs"]
        x_mask[i, :ti] = seq["masks"]
        x_time[i, :ti] = seq["times"]
        y_event[i] = int(event_map.get(pid, 0.0))
    return x_val, x_mask, x_time, y_event, pids


# ── Clase principal ───────────────────────────────────────────────────────────


class ClinicalLongitudinalEDA:
    """EDA sobre un tensor clinico longitudinal irregular (labs).

    Pensado para datos con ausencia estructural (mascara por variable/visita) y
    eje temporal continuo (edad). Todas las figuras se guardan a fichero via
    ``save_path``; los metodos devuelven stats como dict para volcar a JSON.

    Args:
        X_val:   (N, T, D) valores de las analiticas (z-score).
        X_mask:  (N, T, D) binario, 1 = observado.
        X_time:  (N, T) o (N, T, 1) edad del paciente en cada timestep (anos).
        y_event: (N,) etiqueta binaria.
        lab_names: lista de D codigos de laboratorio.
        pids:     lista de N identificadores de paciente (opcional).
        time_unit: unidad de X_time ('years' o 'days').
    """

    def __init__(
        self,
        X_val: np.ndarray,
        X_mask: np.ndarray,
        X_time: np.ndarray,
        y_event: np.ndarray,
        lab_names: list[str],
        pids: Optional[list[str]] = None,
        time_unit: str = "years",
    ) -> None:
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.X_mask = np.asarray(X_mask, dtype=np.float32)
        x_time = np.asarray(X_time, dtype=np.float32)
        if x_time.ndim == 3:
            x_time = x_time[..., 0]
        self.X_time = x_time
        self.y_event = np.asarray(y_event).astype(int)
        self.lab_names = list(lab_names)
        self.pids = (
            list(pids)
            if pids is not None
            else [str(i) for i in range(len(self.y_event))]
        )
        self.time_unit = time_unit
        self.N, self.T, self.D = self.X_val.shape
        self.lab_to_idx = {c: i for i, c in enumerate(self.lab_names)}
        # visit_mask[i, t] = True si el timestep t es una visita real (alguna
        # analitica observada). Distingue visita de padding.
        self.visit_mask = self.X_mask.sum(axis=2) > 0
        sns.set_theme(style="whitegrid", context="notebook")

    # ── A) Trayectorias longitudinales ───────────────────────────────────

    def plot_patient_batch_trajectories(
        self,
        patient_indices: Optional[list[int]] = None,
        batch_size: int = 5,
        lab_codes: Optional[list[str]] = None,
        time_axis: str = "age",
        max_gap_to_connect: float = 1.0,
        seed: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Trayectorias individuales por paciente y biomarcador.

        Solo representa puntos observados (``X_mask > 0``) sobre su edad real.
        Conecta con linea discontinua ligera SOLO entre puntos consecutivos
        cuyo ``dt <= max_gap_to_connect`` (no conecta huecos largos).

        Args:
            patient_indices: indices de pacientes (posiciones en N). Si None,
                muestrea ``batch_size`` aleatoriamente (con ``seed``).
            lab_codes: subconjunto de codigos LOINC. Si None, todos.
            time_axis: 'age' (edad real) o 'index' (indice de visita 0..K).
            max_gap_to_connect: gap maximo (anos) para trazar segmento discontinuo.
            save_path: ruta PNG a guardar la figura.
        """
        if patient_indices is None:
            rng = np.random.default_rng(seed)
            k = min(batch_size, self.N)
            patient_indices = list(rng.choice(self.N, size=k, replace=False))
        if lab_codes is None:
            lab_idx = list(range(self.D))
        else:
            lab_idx = [self.lab_to_idx[c] for c in lab_codes if c in self.lab_to_idx]
        n_pts = len(patient_indices)
        n_labs = len(lab_idx)
        if n_pts == 0 or n_labs == 0:
            raise ValueError("Sin pacientes o labs para graficar trayectorias.")
        fig, axes = plt.subplots(
            n_pts, n_labs, figsize=(2.4 * n_labs, 2.2 * n_pts), squeeze=False
        )
        for r, pi in enumerate(patient_indices):
            t_all = self.X_time[pi]
            for c, li in enumerate(lab_idx):
                ax = axes[r, c]
                obs = self.X_mask[pi, :, li] > 0
                x = (
                    t_all[obs]
                    if time_axis == "age"
                    else np.flatnonzero(obs).astype(float)
                )
                y = self.X_val[pi, obs, li]
                ax.scatter(x, y, s=12, zorder=3)
                if len(x) >= 2:
                    gaps = np.diff(x)
                    seg = 0
                    for k in range(len(gaps)):
                        if gaps[k] > max_gap_to_connect:
                            ax.plot(
                                x[seg : k + 1], y[seg : k + 1], "--", lw=0.7, alpha=0.4
                            )
                            seg = k + 1
                    ax.plot(x[seg:], y[seg:], "--", lw=0.7, alpha=0.4)
                if r == 0:
                    ax.set_title(self.lab_names[li], fontsize=8)
                if c == 0:
                    ax.set_ylabel(
                        f"pid {self.pids[pi]}\n(e={int(self.y_event[pi])})", fontsize=7
                    )
                ax.tick_params(labelsize=6)
        fig.suptitle("Trayectorias longitudinales (puntos observados)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── B) Sparsity y dinamica de visitas ────────────────────────────────

    def _visits_per_patient(self) -> np.ndarray:
        return self.visit_mask.sum(axis=1).astype(int)

    def _deltas_between_visits(self) -> np.ndarray:
        """Delta t entre visitas consecutivas validas (anos)."""
        deltas = []
        for i in range(self.N):
            t = self.X_time[i][self.visit_mask[i]]
            if len(t) >= 2:
                deltas.append(np.diff(t))
        return np.concatenate(deltas) if deltas else np.zeros(0, dtype=np.float32)

    def sparsity_visit_stats(self) -> dict:
        """Stats de numero de visitas/paciente y de deltas entre visitas."""
        vp = self._visits_per_patient()
        dt = self._deltas_between_visits()
        # dias si la unidad es anos
        dt_days = dt * 365.25 if self.time_unit == "years" else dt
        q10, q90 = np.percentile(vp, [10, 90]) if len(vp) else (0.0, 0.0)
        ultra = int(np.sum(vp <= q10)) if len(vp) else 0
        hyper = int(np.sum(vp >= q90)) if len(vp) else 0
        return {
            "visits_per_patient": {
                "mean": float(np.mean(vp)) if len(vp) else 0.0,
                "median": float(np.median(vp)) if len(vp) else 0.0,
                "q25": float(np.percentile(vp, 25)) if len(vp) else 0.0,
                "q75": float(np.percentile(vp, 75)) if len(vp) else 0.0,
                "min": int(np.min(vp)) if len(vp) else 0,
                "max": int(np.max(vp)) if len(vp) else 0,
            },
            "delta_t_years": {
                "mean": float(np.mean(dt)) if len(dt) else 0.0,
                "median": float(np.median(dt)) if len(dt) else 0.0,
                "min": float(np.min(dt)) if len(dt) else 0.0,
                "max": float(np.max(dt)) if len(dt) else 0.0,
            },
            "delta_t_days": {
                "mean": float(np.mean(dt_days)) if len(dt) else 0.0,
                "median": float(np.median(dt_days)) if len(dt) else 0.0,
            },
            "ultra_sparse_patients": ultra,
            "hyper_frequent_patients": hyper,
            "ultra_sparse_threshold": float(q10),
            "hyper_frequent_threshold": float(q90),
        }

    def plot_sparsity_visits(self, save_path: Optional[str] = None) -> plt.Figure:
        """Histograma de visitas/paciente + histograma de deltas (anos y dias)."""
        vp = self._visits_per_patient()
        dt = self._deltas_between_visits()
        dt_days = dt * 365.25 if self.time_unit == "years" else dt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].hist(vp, bins=30, color="#4C72B0", edgecolor="white")
        axes[0].set_title("N de visitas por paciente")
        axes[0].set_xlabel("N visitas")
        axes[0].set_ylabel("N pacientes")
        if len(dt):
            axes[1].hist(dt, bins=40, color="#55A868", edgecolor="white")
            axes[1].set_title("Delta t entre visitas (anos)")
            axes[1].set_xlabel("anos")
            axes[2].hist(dt_days, bins=40, color="#C44E52", edgecolor="white")
            axes[2].set_title("Delta t entre visitas (dias)")
            axes[2].set_xlabel("dias")
        for ax in axes[1:]:
            ax.set_ylabel("N pares consecutivos")
        fig.suptitle("Sparsity y dinamica de visitas", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── C) Carga de observabilidad por visita ────────────────────────────

    def _labs_per_visit(self) -> np.ndarray:
        """N de labs medidos simultaneamente en cada visita valida."""
        per_visit = self.X_mask.sum(axis=2)  # (N, T) labs observados por (paciente, timestep)
        return per_visit[self.visit_mask]  # mascara (N, T) -> array 1D de visitas validas

    def _cooccurrence(self) -> tuple[np.ndarray, np.ndarray]:
        """Matriz de co-ocurrencia de analiticas (cuentas y Jaccard)."""
        flat = self.X_mask[self.visit_mask]  # (V, D) binario por visita
        counts = flat.T @ flat  # (D, D): co[d1,d2] = # visitas con ambos
        diag = np.diag(counts).astype(float)
        denom = diag[:, None] + diag[None, :] - counts
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(denom > 0, counts / denom, 0.0)
        np.fill_diagonal(jacc, 1.0)
        return counts, jacc

    def observability_stats(self) -> dict:
        lpv = self._labs_per_visit()
        _, jacc = self._cooccurrence()
        # Top pares por Jaccard (excluyendo diagonal)
        if self.D >= 2:
            iu = np.triu_indices(self.D, k=1)
            order = np.argsort(-jacc[iu])
            top = [
                {
                    "lab_a": self.lab_names[iu[0][k]],
                    "lab_b": self.lab_names[iu[1][k]],
                    "jaccard": float(jacc[iu[0][k], iu[1][k]]),
                }
                for k in order[:5]
            ]
        else:
            top = []
        return {
            "labs_per_visit": {
                "mean": float(np.mean(lpv)) if len(lpv) else 0.0,
                "median": float(np.median(lpv)) if len(lpv) else 0.0,
                "min": int(np.min(lpv)) if len(lpv) else 0,
                "max": int(np.max(lpv)) if len(lpv) else 0,
            },
            "top_cooccurrent_pairs": top,
        }

    def plot_observability_load(self, save_path: Optional[str] = None) -> plt.Figure:
        """Histograma de labs medidos por visita."""
        lpv = self._labs_per_visit()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(lpv, bins=range(0, self.D + 2), color="#8172B3", edgecolor="white")
        ax.set_title("Carga de observabilidad: labs medidos por visita")
        ax.set_xlabel("N de analiticas en la visita")
        ax.set_ylabel("N de visitas")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_cooccurrence(self, save_path: Optional[str] = None) -> plt.Figure:
        """Heatmap de similitud Jaccard entre analiticas (suelen prescribirse juntas)."""
        _, jacc = self._cooccurrence()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            jacc,
            xticklabels=self.lab_names,
            yticklabels=self.lab_names,
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.3,
            cbar_kws={"label": "Jaccard"},
            ax=ax,
        )
        ax.set_title("Co-ocurrencia de analiticas (Jaccard por visita)")
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── Complementario 2) Heatmap de esparsidad global ───────────────────

    def global_completeness_per_lab(self) -> np.ndarray:
        """Fraccion global de pacientes que tienen al menos una observacion del lab."""
        observed_patient = (self.X_mask.sum(axis=1) > 0).astype(float)  # (N, D)
        return observed_patient.mean(axis=0)

    def plot_global_sparsity_heatmap(
        self, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Heatmap (N_pacientes, labs_dim) de completitud por paciente y variable.

        Celda[i, d] = fraccion de las visitas del paciente i donde el lab d
        fue observado. Pacientes ordenados por completitud total (para que el
        patron de sparsity se vea de menos a mas denso).
        """
        n_visits = self.visit_mask.sum(axis=1).astype(float)  # (N,)
        n_visits_safe = np.where(n_visits == 0, 1.0, n_visits)
        # por paciente y lab: # visitas con el lab / # visitas del paciente
        per_lab_count = (
            np.where(self.X_mask > 0, 1, 0).sum(axis=1).astype(float)
        )  # (N, D)
        completeness = per_lab_count / n_visits_safe[:, None]
        order = np.argsort(completeness.sum(axis=1))[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            completeness[order],
            xticklabels=self.lab_names,
            yticklabels=False,
            cmap="magma",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Fraccion de visitas con el lab"},
            ax=ax,
        )
        ax.set_title("Esparsidad global: completitud por paciente y analitica")
        ax.set_xlabel("Analitica (lab)")
        ax.set_ylabel("Pacientes (ordenados)")
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── Complementario 1) Sesgo por muestreo informativo ─────────────────

    def informative_sampling_test(self) -> dict:
        """Compara el delta t entre y=1 e y=0 (Mann-Whitney U, dos colas).

        Si los pacientes con evento se miden con mayor frecuencia (delta t
        menor), hay muestreo informativo: la gravedad mueve el calendario de
        visitas y sesga la representacion del tensor.
        """
        dt_by_event = {0: [], 1: []}
        for i in range(self.N):
            t = self.X_time[i][self.visit_mask[i]]
            if len(t) < 2:
                continue
            dt_by_event[int(self.y_event[i])].extend(np.diff(t).tolist())
        d0 = np.array(dt_by_event[0], dtype=float)
        d1 = np.array(dt_by_event[1], dtype=float)
        if len(d0) and len(d1):
            stat, p = mannwhitneyu(d0, d1, alternative="two-sided")
        else:
            stat, p = float("nan"), float("nan")
        return {
            "median_delta_t_event0": float(np.median(d0)) if len(d0) else 0.0,
            "median_delta_t_event1": float(np.median(d1)) if len(d1) else 0.0,
            "mean_delta_t_event0": float(np.mean(d0)) if len(d0) else 0.0,
            "mean_delta_t_event1": float(np.mean(d1)) if len(d1) else 0.0,
            "n_pairs_event0": int(len(d0)),
            "n_pairs_event1": int(len(d1)),
            "mannwhitney_u": float(stat),
            "p_value": float(p),
            "interpretation": (
                "informativo (eventos medidos con mayor frecuencia)"
                if (
                    not np.isnan(p)
                    and p < 0.05
                    and len(d1)
                    and np.median(d1) < np.median(d0)
                )
                else "no significativo o eventos menos frecuentes"
            ),
        }

    def plot_informative_sampling(self, save_path: Optional[str] = None) -> plt.Figure:
        """Distribuciones superpuestas de delta t por grupo de evento."""
        dt_by_event = {0: [], 1: []}
        for i in range(self.N):
            t = self.X_time[i][self.visit_mask[i]]
            if len(t) < 2:
                continue
            dt_by_event[int(self.y_event[i])].extend(np.diff(t).tolist())
        d0 = np.array(dt_by_event[0], dtype=float)
        d1 = np.array(dt_by_event[1], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if len(d0):
            ax.hist(d0, bins=40, alpha=0.6, label=f"y=0 (n={len(d0)})", color="#4C72B0")
        if len(d1):
            ax.hist(d1, bins=40, alpha=0.6, label=f"y=1 (n={len(d1)})", color="#C44E52")
        ax.set_title("Sesgo de muestreo: delta t entre visitas por grupo")
        ax.set_xlabel("Delta t (anos)")
        ax.set_ylabel("N pares consecutivos")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── D) t-SNE en landmarks temporales ─────────────────────────────────

    def _landmark_vector(self, age_min: float, age_max: float):
        """Vector medio (mask-weighted) de labs por paciente en una ventana.

        Returns:
            vec (N, D) con NaN donde el paciente no tiene ninguna observacion
            en ventana o donde un canal concreto no se midio en ventana.
            kept_idx: pacientes con al menos una observacion en ventana.
            discarded_idx: pacientes sin ninguna observacion (descartados).
        """
        vec = np.full((self.N, self.D), np.nan, dtype=np.float32)
        for i in range(self.N):
            t = self.X_time[i]
            valid = self.visit_mask[i]
            in_win = valid & (t >= age_min) & (t <= age_max)
            if not np.any(in_win):
                continue
            labs_win = self.X_val[i][in_win]
            mask_win = self.X_mask[i][in_win]
            num = np.sum(labs_win * mask_win, axis=0)
            den = mask_win.sum(axis=0).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                vec[i] = np.where(den > 0, num / np.where(den == 0, 1.0, den), np.nan)
        discarded = np.where(np.all(np.isnan(vec), axis=1))[0]
        kept = np.where(~np.all(np.isnan(vec), axis=1))[0]
        return vec, kept, discarded

    def _tsne_embedding(
        self,
        matrix: np.ndarray,
        perplexity: float,
        n_iter: int,
        random_state: int,
        pca_threshold_dim: int,
    ) -> np.ndarray:
        """Imputa (KNN), PCA si D>umbral, y proyecta a 2D con t-SNE."""
        n = matrix.shape[0]
        if n < 3:
            return np.zeros((n, 2), dtype=np.float32)
        imputer = KNNImputer(n_neighbors=min(5, max(1, n - 1)))
        mat = imputer.fit_transform(matrix)
        if mat.shape[1] > pca_threshold_dim:
            mat = PCA(
                n_components=min(pca_threshold_dim, n - 1, mat.shape[1])
            ).fit_transform(mat)
        perplex = min(perplexity, max(2.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplex,
            max_iter=n_iter,
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(mat)

    def plot_tsne_landmarks(
        self,
        landmark_age_min: float,
        landmark_age_max: float,
        window_half_width: float = 1.0,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        knn_n_neighbors: int = 5,
        random_state: int = 0,
        pca_threshold_dim: int = 30,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, dict]:
        """Proyeccion t-SNE en dos ventanas (basal y tardia), lado a lado.

        Cada ventana: vector medio de labs por paciente (mask-weighted),
        pacientes sin mediciones en ventana descartados, KNN imputa canales
        parciales, PCA previo si labs_dim > umbral, t-SNE a 2D. Color por
        ``y_event`` para evaluar si aumenta la separabilidad fenotipica a edades
        mas avanzadas.
        """
        windows = {
            "basal": (
                landmark_age_min - window_half_width,
                landmark_age_min + window_half_width,
            ),
            "late": (
                landmark_age_max - window_half_width,
                landmark_age_max + window_half_width,
            ),
        }
        emb = {}
        discard = {}
        for name, (a, b) in windows.items():
            vec, kept, disc = self._landmark_vector(a, b)
            discard[name] = {
                "n_kept": int(len(kept)),
                "n_discarded": int(len(disc)),
                "pct_discarded": float(100.0 * len(disc) / max(self.N, 1)),
            }
            if len(kept) >= 3:
                emb[name] = self._tsne_embedding(
                    vec[kept],
                    perplexity=perplexity,
                    n_iter=n_iter,
                    random_state=random_state,
                    pca_threshold_dim=pca_threshold_dim,
                )
            else:
                emb[name] = np.zeros((0, 2), dtype=np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for ax, name in zip(axes, ["basal", "late"]):
            kept = self._landmark_vector(*windows[name])[1]
            e = emb[name]
            if e.shape[0] >= 3:
                yk = self.y_event[kept]
                for ev, col, lab in [(0, "#4C72B0", "y=0"), (1, "#C44E52", "y=1")]:
                    m = yk == ev
                    ax.scatter(e[m, 0], e[m, 1], c=col, s=22, alpha=0.7, label=lab)
                ax.set_title(
                    f"{name} (edad {windows[name][0]:.1f}-{windows[name][1]:.1f})\n"
                    f"n={len(kept)} (descartados {discard[name]['pct_discarded']:.1f}%)"
                )
                ax.legend()
            else:
                ax.set_title(f"{name}: datos insuficientes (n={len(kept)})")
                ax.text(0.5, 0.5, "n < 3", ha="center", va="center")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        fig.suptitle(
            f"t-SNE en landmarks (basal {landmark_age_min} vs tardio {landmark_age_max})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig, discard

    # ── Resumen global ───────────────────────────────────────────────────

    def full_summary(self) -> dict:
        """Agrega todas las stats en un dict listo para JSON."""
        return {
            "n_patients": int(self.N),
            "n_labs": int(self.D),
            "t_max": int(self.T),
            "event_rate": float(self.y_event.mean()) if self.N else 0.0,
            "lab_names": self.lab_names,
            "global_completeness_per_lab": {
                self.lab_names[d]: float(v)
                for d, v in enumerate(self.global_completeness_per_lab())
            },
            "sparsity_visits": self.sparsity_visit_stats(),
            "observability": self.observability_stats(),
            "informative_sampling": self.informative_sampling_test(),
        }


# ── Cuerpo del script (Snakemake) ────────────────────────────────────────────
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]).
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

sys.path.append(str(Path(__file__).parent))
from dataset import RawSequenceDatasetWithICD, load_sequences  # noqa: E402

# 1. Parametros inyectados
out_dir = Path(snakemake.output[0])
out_dir.mkdir(parents=True, exist_ok=True)

landmark_age_min = float(snakemake.params.landmark_age_min)
landmark_age_max = float(snakemake.params.landmark_age_max)
window_half_width = float(snakemake.params.window_half_width)
batch_size = int(snakemake.params.batch_size)
max_gap = float(snakemake.params.max_gap_to_connect)
random_state = int(snakemake.params.random_state)
tsne_perplexity = float(snakemake.params.tsne_perplexity)
tsne_n_iter = int(snakemake.params.tsne_n_iter)
knn_n_neighbors = int(snakemake.params.knn_n_neighbors)
pca_threshold_dim = int(snakemake.params.pca_threshold_dim)
index_age = float(snakemake.params.index_age)
landmark_min = int(snakemake.params.landmark_min)
landmark_max = int(snakemake.params.landmark_max)
max_events = int(snakemake.params.max_events)
age_weight_gamma = float(snakemake.params.age_weight_gamma)

# 2. Cargar secuencias de train (contract: sequences, lab_codes, cohort)
sequences, lab_codes, cohort = load_sequences(snakemake.input.sequences)
print(
    f"[eda] secuencias cargadas: {len(sequences)} pacientes, {len(lab_codes)} labs",
    flush=True,
)

# 3. Filtrar via RawSequenceDatasetWithICD: replicar el code_to_idx de train.py
all_icd_codes = set()
for pid, seq in sequences.items():
    for step_codes in seq["codes"]:
        all_icd_codes.update(step_codes)
code_to_idx = {code: i + 1 for i, code in enumerate(sorted(all_icd_codes))}
landmark_ages = list(range(landmark_min, landmark_max + 1))
model_ds = RawSequenceDatasetWithICD(
    sequences=sequences,
    cohort_df=cohort,
    landmark_ages=landmark_ages,
    index_age=index_age,
    code_to_idx=code_to_idx,
    max_events=max_events,
    age_weight_gamma=age_weight_gamma,
)
model_pids = {s["pid"] for s in model_ds.samples}
print(
    f"[eda] pacientes crudos: {len(sequences)} | "
    f"pacientes que entran al modelo: {len(model_pids)} | "
    f"samples (paciente x landmark): {len(model_ds.samples)}",
    flush=True,
)

# 4. Tensor a nivel paciente y filtrado a los que entran al modelo
x_val, x_mask, x_time, y_event, pids = build_patient_tensors(
    sequences, cohort, lab_codes
)
if pids:
    keep_idx = [i for i, p in enumerate(pids) if p in model_pids]
    sel = np.array(keep_idx, dtype=int)
    x_val, x_mask, x_time = x_val[sel], x_mask[sel], x_time[sel]
    y_event = y_event[sel]
    pids = [pids[i] for i in sel]
    print(f"[eda] tensor paciente: {x_val.shape} (filtrado al modelo)", flush=True)
else:
    print("[eda] WARNING: sin pacientes para EDA", flush=True)

# 5. Instanciar y correr todos los analisis
eda = ClinicalLongitudinalEDA(
    X_val=x_val,
    X_mask=x_mask,
    X_time=x_time,
    y_event=y_event,
    lab_names=lab_codes,
    pids=pids,
    time_unit="years",
)

# A) Trayectorias
eda.plot_patient_batch_trajectories(
    batch_size=batch_size,
    max_gap_to_connect=max_gap,
    seed=random_state,
    save_path=str(out_dir / "trajectories.png"),
)
print("[eda] trajectories.png", flush=True)

# B) Sparsity y dinamica de visitas
eda.plot_sparsity_visits(save_path=str(out_dir / "sparsity_visits.png"))
print("[eda] sparsity_visits.png", flush=True)

# C) Carga de observabilidad + co-ocurrencia
eda.plot_observability_load(save_path=str(out_dir / "observability_load.png"))
eda.plot_cooccurrence(save_path=str(out_dir / "cooccurrence.png"))
print("[eda] observability_load.png + cooccurrence.png", flush=True)

# Complementario 2) Heatmap esparsidad global
eda.plot_global_sparsity_heatmap(save_path=str(out_dir / "global_sparsity_heatmap.png"))
print("[eda] global_sparsity_heatmap.png", flush=True)

# Complementario 1) Muestreo informativo
eda.plot_informative_sampling(save_path=str(out_dir / "informative_sampling.png"))
print("[eda] informative_sampling.png", flush=True)

# D) t-SNE en landmarks
_, tsne_discard = eda.plot_tsne_landmarks(
    landmark_age_min=landmark_age_min,
    landmark_age_max=landmark_age_max,
    window_half_width=window_half_width,
    perplexity=tsne_perplexity,
    n_iter=tsne_n_iter,
    knn_n_neighbors=knn_n_neighbors,
    random_state=random_state,
    pca_threshold_dim=pca_threshold_dim,
    save_path=str(out_dir / "tsne_landmarks.png"),
)
print("[eda] tsne_landmarks.png", flush=True)

# 6. Resumen JSON
summary = eda.full_summary()
summary["model_filter"] = {
    "n_raw_sequences": int(len(sequences)),
    "n_patients_in_model": int(len(model_pids)),
    "n_samples_patient_landmark": int(len(model_ds.samples)),
    "landmark_ages": landmark_ages,
    "max_events": max_events,
}
summary["tsne_landmarks"] = tsne_discard
with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"[eda] resumen en {out_dir / 'eda_summary.json'}", flush=True)
print(f"[eda] DONE -> {out_dir}", flush=True)
