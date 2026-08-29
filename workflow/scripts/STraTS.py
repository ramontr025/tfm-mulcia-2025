import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidad de unpickling de Pandas (Colab vs Local)
# ─────────────────────────────────────────────────────────────────────────────
try:
    _orig_string_dtype_init = pd.StringDtype.__init__

    def _safe_string_dtype_init(self, *args, **kwargs):
        try:
            _orig_string_dtype_init(self, *args, **kwargs)
        except TypeError:
            storage = args[0] if len(args) > 0 else kwargs.get("storage", "python")
            _orig_string_dtype_init(self, storage=storage)

    pd.StringDtype.__init__ = _safe_string_dtype_init
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss Function (Cox Proportional Hazards)
# ─────────────────────────────────────────────────────────────────────────────

def cox_ph_loss(
    risk_scores: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow ties handling."""
    durations, indices = torch.sort(durations, descending=True)
    risk_scores = risk_scores[indices]
    events = events[indices]
    if weights is not None:
        weights = weights[indices]

    max_risk = torch.max(risk_scores)
    exp_risk = torch.exp(risk_scores - max_risk)
    cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
    log_cumsum_exp = torch.log(cumsum_exp_risk) + max_risk

    log_likelihood = risk_scores - log_cumsum_exp

    if weights is not None:
        sum_weights = torch.sum(events * weights)
        if sum_weights == 0:
            loss = (risk_scores * 0.0).sum()
        else:
            loss = -torch.sum(log_likelihood * events * weights) / sum_weights
    else:
        sum_events = torch.sum(events)
        if sum_events == 0:
            loss = (risk_scores * 0.0).sum()
        else:
            loss = -torch.sum(log_likelihood * events) / sum_events
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset & Collate (Triplet representation: t, f, v)
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences(pkl_path: Path | str) -> tuple[dict, list[str], object]:
    """Carga el pickle con las secuencias del split correspondiente."""
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    sequences = {pid: seq for pid, seq in cache["sequences"].items() if seq is not None}
    return sequences, list(cache["lab_codes"]), cache["cohort"]


class STraTSDataset(Dataset):
    """
    Dataset para STraTS: genera una muestra por cada par (paciente, landmark_age)
    con secuencias irregulares estructuradas en tripletas unificadas (t, f, v).
    """

    def __init__(
        self,
        sequences: dict,
        cohort_df,
        landmark_ages: list[float],
        index_age: float,
        code_to_idx: dict,
        max_events: int = 200,
        age_weight_gamma: float = 1.0,
    ):
        self.samples = []
        self.event_flags = []
        self.lab_dim = None
        self.code_to_idx = code_to_idx
        lm_arr = np.array(landmark_ages, dtype=np.float32)
        min_lm, max_lm = float(lm_arr.min()), float(lm_arr.max())

        for row in cohort_df.itertuples(index=False):
            pid = str(row.Id)
            if pid not in sequences:
                continue
            duration = float(row.duration)
            event = float(row.event)
            seq = sequences[pid]
            times = np.asarray(seq["times"], dtype=np.float32)
            labs = np.asarray(seq["labs"], dtype=np.float32)
            masks = np.asarray(seq["masks"], dtype=np.float32)
            sex_male = float(seq["sex_male"])
            codes = seq["codes"]  # Lista de listas de strings (ICD-10)

            if self.lab_dim is None:
                self.lab_dim = labs.shape[1]

            for lm in landmark_ages:
                time_from_lm = duration - (lm - index_age)
                if time_from_lm <= 0:
                    continue

                keep = times <= lm
                if not np.any(keep):
                    times_s = np.array([lm], dtype=np.float32)
                    labs_s = np.zeros((1, self.lab_dim), dtype=np.float32)
                    masks_s = np.zeros((1, self.lab_dim), dtype=np.float32)
                    codes_s = [[]]
                else:
                    times_s = times[keep]
                    labs_s = labs[keep]
                    masks_s = masks[keep]
                    codes_s = [c for k, c in zip(keep, codes) if k]

                # 1. Tripletas de variables continuas (Laboratorios)
                ti, fi = np.nonzero(masks_s)
                t_labs = times_s[ti]
                f_labs = fi.astype(np.int64)  # IDs en el rango [0, lab_dim - 1]
                v_labs = labs_s[ti, fi]

                # 2. Tripletas de códigos categóricos ICD-10 (IDs a partir de lab_dim, valor = 1.0)
                t_codes, f_codes, v_codes = [], [], []
                for step_idx, step in enumerate(codes_s):
                    t_curr = times_s[step_idx]
                    for code in step:
                        if code in self.code_to_idx:
                            c_idx = self.code_to_idx[code]
                            t_codes.append(t_curr)
                            f_codes.append(self.lab_dim + c_idx)
                            v_codes.append(1.0)

                # 3. Unificar ambas fuentes de tripletas
                if len(t_codes) > 0:
                    t_all = np.concatenate([t_labs, np.array(t_codes, dtype=np.float32)])
                    f_all = np.concatenate([f_labs, np.array(f_codes, dtype=np.int64)])
                    v_all = np.concatenate([v_labs, np.array(v_codes, dtype=np.float32)])
                else:
                    t_all = t_labs
                    f_all = f_labs
                    v_all = v_labs

                # Caso borde: paciente sin eventos previos al landmark
                if len(t_all) == 0:
                    t_all = np.array([lm], dtype=np.float32)
                    f_all = np.array([0], dtype=np.int64)
                    v_all = np.array([0.0], dtype=np.float32)

                # Ordenar cronológicamente las tripletas
                order = np.argsort(t_all, kind="stable")
                t_all = t_all[order]
                f_all = f_all[order]
                v_all = v_all[order]

                # Truncar si excede la longitud máxima
                if max_events and len(t_all) > max_events:
                    t_all = t_all[-max_events:]
                    f_all = f_all[-max_events:]
                    v_all = v_all[-max_events:]

                time_since_index_all = t_all - float(index_age)

                # Ponderación basada en la edad del landmark
                lm_norm = (lm - min_lm) / max(max_lm - min_lm, 1e-6)
                weight = 1.0 + age_weight_gamma * lm_norm

                self.samples.append(
                    {
                        "times": t_all,
                        "values": v_all,
                        "features": f_all,
                        "sex_male": sex_male,
                        "time_since_index": time_since_index_all,
                        "duration": time_from_lm,
                        "event": event,
                        "landmark_age": float(lm),
                        "weight": weight,
                        "pid": pid,
                    }
                )
                self.event_flags.append(int(event))

        if self.lab_dim is None:
            self.lab_dim = 0
        n = len(self.samples)
        self.event_rate = sum(self.event_flags) / max(n, 1)
        self.mean_seq_len = (
            float(np.mean([len(s["times"]) for s in self.samples])) if n else 0.0
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["times"], dtype=torch.float32),
            torch.tensor(s["values"], dtype=torch.float32),
            torch.tensor(s["features"], dtype=torch.long),
            torch.tensor(s["sex_male"], dtype=torch.float32),
            torch.tensor(s["time_since_index"], dtype=torch.float32),
            torch.tensor(s["duration"], dtype=torch.float32),
            torch.tensor(s["event"], dtype=torch.float32),
            torch.tensor(s["landmark_age"], dtype=torch.float32),
            torch.tensor(s["weight"], dtype=torch.float32),
            s["pid"],
        )


def strats_pad_collate(batch):
    """
    Collate function para batching de secuencias de tripletas con padding_mask.
    """
    (
        times,
        values,
        features,
        sex_male,
        time_since_index,
        durations,
        events,
        landmarks,
        weights,
        ids,
    ) = zip(*batch)

    lengths = torch.tensor([len(t) for t in times], dtype=torch.long)
    max_len = int(lengths.max()) if int(lengths.max()) > 0 else 1

    padded_times = nn.utils.rnn.pad_sequence(times, batch_first=True, padding_value=0.0)
    padded_values = nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0.0)
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padded_time_since_index = nn.utils.rnn.pad_sequence(
        time_since_index, batch_first=True, padding_value=0.0
    )

    # Máscara booleana: True indica posición de relleno (padding)
    arange = torch.arange(max_len).unsqueeze(0)
    padding_mask = arange >= lengths.unsqueeze(1)

    return (
        padded_values,            # (B, N_max) valores continuos o 1.0
        padded_features.long(),   # (B, N_max) IDs de variables
        padded_times,             # (B, N_max) timestamps absolutos (edades)
        padded_time_since_index,  # (B, N_max) tiempo relativo a index_age
        padding_mask,             # (B, N_max) bool para src_key_padding_mask
        lengths,                  # (B,)
        torch.stack(sex_male),    # (B,)
        torch.stack(durations),   # (B,)
        torch.stack(events),      # (B,)
        torch.stack(landmarks),   # (B,)
        torch.stack(weights),     # (B,)
        list(ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Módulos de la Arquitectura STraTS
# ─────────────────────────────────────────────────────────────────────────────

class CVE(nn.Module):
    """
    Continuous Value Embedding module (MLP 2 capas: 1 -> sqrt(d) -> d).
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = max(int(np.sqrt(hid_dim)), 1)
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: (bsz, max_len)
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # (bsz, max_len, int_dim)
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # (bsz, max_len, hid_dim)
        return x


class FusionAtt(nn.Module):
    """
    Fusion Self-Attention pooling module (pondera tripletas observadas para obtener e^T).
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = hid_dim
        self.W = nn.Parameter(torch.empty(hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: (bsz, max_len, hid_dim)
        # mask: (bsz, max_len) — 1.0 para tokens reales, 0.0 para padding
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # (bsz, max_len, int_dim)
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # (bsz, max_len)
        att = att + (1.0 - mask.float()) * -1e9  # Enmascarar padding
        att = torch.softmax(att, dim=-1)  # (bsz, max_len)
        return att


class STraTS(nn.Module):
    """
    Modelo STraTS completo para predicción de supervivencia.
    """

    def __init__(
        self,
        vocab_size: int,
        lab_dim: int,
        hid_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        demo_dim: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lab_dim = lab_dim
        self.hid_dim = hid_dim

        # 1. Embeddings de tripletas
        self.time_cve = CVE(hid_dim)
        self.feature_emb = nn.Embedding(vocab_size, hid_dim)
        self.val_cve = CVE(hid_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Bloques de Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3. Fusion Attention Pooling
        self.fusion_att = FusionAtt(hid_dim)

        # 4. Demographics Embedding (sex_male)
        self.demo_fc = nn.Linear(1, demo_dim)

        # 5. Capa final de predicción de riesgo (log-hazard)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim + demo_dim, 1),
        )

    def forward(
        self,
        values,            # (B, N_max) float
        features,          # (B, N_max) long
        times,             # (B, N_max) float (edad exacta del evento)
        time_since_index,  # (B, N_max) float
        padding_mask,      # (B, N_max) bool (True for padding)
        lengths,           # (B,) long
        sex_male,          # (B,) float
        **kwargs,
    ):
        # 1. Embeddings de los componentes de la tripleta (t, f, v)
        t_emb = self.time_cve(times)             # (B, N_max, hid_dim)
        f_emb = self.feature_emb(features)       # (B, N_max, hid_dim)
        v_emb = self.val_cve(values)             # (B, N_max, hid_dim)

        # Para diagnósticos ICD-10 (features >= lab_dim), anulamos el embedding continuo
        is_lab = (features < self.lab_dim).unsqueeze(-1).float()
        v_emb = v_emb * is_lab

        triplet_emb = self.dropout(t_emb + f_emb + v_emb)  # (B, N_max, hid_dim)

        # 2. Transformer Encoder con máscara de padding
        out = self.transformer_encoder(
            triplet_emb, src_key_padding_mask=padding_mask
        )  # (B, N_max, hid_dim)

        # 3. Fusion Self-Attention pooling -> e^T
        real_mask = (~padding_mask).float()
        att_weights = self.fusion_att(out, real_mask)  # (B, N_max)
        e_T = torch.sum(out * att_weights.unsqueeze(-1), dim=1)  # (B, hid_dim)

        # 4. Demographics embedding e^d
        e_d = torch.tanh(self.demo_fc(sex_male.unsqueeze(-1)))  # (B, demo_dim)

        # 5. Concatenación [e^T, e^d] y proyección a riesgo
        e = torch.cat([e_T, e_d], dim=-1)  # (B, hid_dim + demo_dim)
        last_risk = self.head(e).squeeze(-1)  # (B,)

        all_risks = None
        h_states = out

        return last_risk, all_risks, h_states


# ─────────────────────────────────────────────────────────────────────────────
# 4. Funciones de Entrenamiento y Evaluación
# ─────────────────────────────────────────────────────────────────────────────

def run_strats_epoch(loader, model, device, optimizer=None):
    """Ejecuta una época de entrenamiento o validación."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_risks, all_dur, all_ev = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            (
                values,
                features,
                times,
                time_since_index,
                padding_mask,
                lengths,
                sex_male,
                durations,
                events,
                landmarks,
                weights,
                ids,
            ) = batch

            values = values.to(device)
            features = features.to(device)
            times = times.to(device)
            time_since_index = time_since_index.to(device)
            padding_mask = padding_mask.to(device)
            lengths = lengths.to(device)
            sex_male = sex_male.to(device)
            durations = durations.to(device)
            events = events.to(device)
            weights = weights.to(device)

            (risk_scores, _, _) = model(
                values=values,
                features=features,
                times=times,
                time_since_index=time_since_index,
                padding_mask=padding_mask,
                lengths=lengths,
                sex_male=sex_male,
            )

            loss = cox_ph_loss(risk_scores, durations, events, weights)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_risks.extend(risk_scores.detach().cpu().numpy())
            all_dur.extend(durations.cpu().numpy())
            all_ev.extend(events.cpu().numpy())

    mean_loss = total_loss / max(len(loader), 1)
    ci = concordance_index(
        np.array(all_dur), -np.array(all_risks), np.array(all_ev, dtype=bool)
    )
    return mean_loss, ci


def collect_strats_outputs(loader, model, device):
    """Recolecta predicciones de riesgo sobre un DataLoader completo."""
    model.eval()
    risks, durs, evs, lms = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            (
                values,
                features,
                times,
                time_since_index,
                padding_mask,
                lengths,
                sex_male,
                durations,
                events,
                landmarks,
                weights,
                ids,
            ) = batch

            r, _, _ = model(
                values=values.to(device),
                features=features.to(device),
                times=times.to(device),
                time_since_index=time_since_index.to(device),
                padding_mask=padding_mask.to(device),
                lengths=lengths.to(device),
                sex_male=sex_male.to(device),
            )
            risks.extend(r.cpu().numpy())
            durs.extend(durations.numpy())
            evs.extend(events.numpy())
            lms.extend(landmarks.numpy())
    return (np.array(risks), np.array(durs), np.array(evs, dtype=float), np.array(lms))


def compute_baseline_hazard(risks, durations, events, horizon):
    """Estimador de Breslow: H0(horizon) a partir de los datos de entrenamiento."""
    risks = np.asarray(risks, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)

    max_r = risks.max()
    shifted = risks - max_r
    H0 = 0.0
    for t in np.unique(durations[events == 1]):
        if t > horizon:
            continue
        d = np.sum((durations == t) & (events == 1))
        risk_set = durations >= t
        denom = np.sum(np.exp(shifted[risk_set]))
        if denom > 0:
            H0 += d / denom
    return H0, max_r


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline Completo: Train & Evaluate STraTS
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_strats(cfg: dict):
    """
    Función principal para entrenar y evaluar STraTS de principio a fin.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Ejecutando STraTS en {device} ===")

    # 1. Cargar secuencias
    print(f"\n1. Cargando datos desde: {cfg['train_path']}, {cfg['val_path']}...")
    sequences_train, lab_codes, cohort_train = load_sequences(cfg["train_path"])
    sequences_val, _, cohort_val = load_sequences(cfg["val_path"])
    sequences_holdout, _, cohort_holdout = (
        load_sequences(cfg["holdout_path"]) if cfg.get("holdout_path") else (None, None, None)
    )

    # 2. Crear vocabulario ICD-10 dinámico
    all_icd_codes = set()
    for pid, seq in sequences_train.items():
        for step_codes in seq["codes"]:
            for code in step_codes:
                all_icd_codes.add(code)
    code_to_idx = {code: i + 1 for i, code in enumerate(sorted(all_icd_codes))}
    lab_dim = len(lab_codes)
    vocab_size = lab_dim + len(code_to_idx) + 1

    print(f"-> Variables clínicas (labs): {lab_dim}")
    print(f"-> Diagnósticos ICD-10: {len(code_to_idx)} códigos únicos")
    print(f"-> Vocabulario total de features: {vocab_size}")

    landmark_ages = list(range(int(cfg["landmark_min"]), int(cfg["landmark_max"]) + 1))

    # 3. Datasets y DataLoaders
    train_dataset = STraTSDataset(
        sequences=sequences_train,
        cohort_df=cohort_train,
        landmark_ages=landmark_ages,
        index_age=cfg["index_age"],
        code_to_idx=code_to_idx,
        max_events=cfg["max_events"],
        age_weight_gamma=cfg["age_weight_gamma"],
    )

    val_dataset = STraTSDataset(
        sequences=sequences_val,
        cohort_df=cohort_val,
        landmark_ages=landmark_ages,
        index_age=cfg["index_age"],
        code_to_idx=code_to_idx,
        max_events=cfg["max_events"],
        age_weight_gamma=cfg["age_weight_gamma"],
    )

    sample_weights = [
        cfg["event_oversample"] if f == 1 else 1.0 for f in train_dataset.event_flags
    ]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        collate_fn=strats_pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=strats_pad_collate,
    )

    # 4. Inicializar modelo STraTS
    model = STraTS(
        vocab_size=vocab_size,
        lab_dim=lab_dim,
        hid_dim=cfg["hid_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        demo_dim=cfg["demo_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Bucle de Entrenamiento
    print("\n2. Iniciando Entrenamiento...", flush=True)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss, train_ci = run_strats_epoch(
            train_loader, model, device, optimizer=optimizer
        )
        val_loss, val_ci = run_strats_epoch(
            val_loader, model, device, optimizer=None
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
            f"Train Loss: {train_loss:.4f} - CI: {train_ci:.4f} | "
            f"Val Loss: {val_loss:.4f} - CI: {val_ci:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  → Early stopping alcanzado en época {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nMejor Val Loss: {best_val_loss:.4f}")

    # 6. Evaluación detallada por Landmark
    print("\n3. Evaluando métricas por Landmark...")
    metrics_by_split = {}
    splits_to_eval = [("train", sequences_train, cohort_train), ("val", sequences_val, cohort_val)]
    if sequences_holdout is not None:
        splits_to_eval.append(("holdout", sequences_holdout, cohort_holdout))

    for split_name, seqs, cohort in splits_to_eval:
        ds_lm = STraTSDataset(
            seqs,
            cohort,
            landmark_ages=landmark_ages,
            index_age=cfg["index_age"],
            code_to_idx=code_to_idx,
            max_events=cfg["max_events"],
            age_weight_gamma=1.0,
        )
        loader_lm = DataLoader(
            ds_lm,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=strats_pad_collate,
        )
        risks, durs, evs, lms = collect_strats_outputs(loader_lm, model, device)

        c_indices_by_lm = {}
        for lm in landmark_ages:
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

        mean_ci = float(np.mean([item["c_index"] for item in c_indices_by_lm.values()]))
        metrics_by_split[split_name] = {
            "mean_c_index": mean_ci,
            "landmarks": c_indices_by_lm,
        }
        print(f"-> {split_name.upper()} C-Index Medio: {mean_ci:.4f}")

    # 7. Riesgo base y calibración (Landmark 55)
    print("\n4. Calculando Riesgo Base y Calibración en Landmark 55...")
    ds_train_55 = STraTSDataset(
        sequences_train,
        cohort_train,
        landmark_ages=[55],
        index_age=cfg["index_age"],
        code_to_idx=code_to_idx,
        max_events=cfg["max_events"],
        age_weight_gamma=1.0,
    )
    train_loader_55 = DataLoader(
        ds_train_55,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=strats_pad_collate,
    )
    train_risks, train_durs, train_evs, _ = collect_strats_outputs(
        train_loader_55, model, device
    )
    H0, max_r = compute_baseline_hazard(train_risks, train_durs, train_evs, cfg["horizon"])

    calibration_results = {}
    for split_name, seqs, cohort in splits_to_eval[1:]:  # val y holdout
        ds_eval_55 = STraTSDataset(
            seqs,
            cohort,
            landmark_ages=[55],
            index_age=cfg["index_age"],
            code_to_idx=code_to_idx,
            max_events=cfg["max_events"],
            age_weight_gamma=1.0,
        )
        eval_loader_55 = DataLoader(
            ds_eval_55,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=strats_pad_collate,
        )
        eval_risks, eval_durs, eval_evs, _ = collect_strats_outputs(
            eval_loader_55, model, device
        )
        shifted_eval = eval_risks - max_r
        pred_prob = 1.0 - np.exp(-H0 * np.exp(shifted_eval))
        obs_event_horizon = ((eval_durs <= cfg["horizon"]) & (eval_evs == 1)).astype(float)
        calibration_results[split_name] = {
            "mean_pred_prob": float(np.mean(pred_prob)),
            "mean_obs_rate": float(np.mean(obs_event_horizon)),
        }
        print(
            f"-> Calibración LM 55 ({split_name}): "
            f"Pred = {calibration_results[split_name]['mean_pred_prob']:.4f} vs "
            f"Obs = {calibration_results[split_name]['mean_obs_rate']:.4f}"
        )

    # 8. Guardar artefacto del modelo y métricas si se especifican salidas
    if cfg.get("output_model_path"):
        Path(cfg["output_model_path"]).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "arch": "strats",
            "vocab_size": vocab_size,
            "lab_dim": lab_dim,
            "hid_dim": cfg["hid_dim"],
            "code_to_idx": code_to_idx,
            "lab_codes": lab_codes,
            "state_dict": model.state_dict(),
        }
        with open(cfg["output_model_path"], "wb") as f:
            pickle.dump(artifact, f)
        print(f"\nArtefacto guardado en: {cfg['output_model_path']}")

    if cfg.get("output_results_path"):
        Path(cfg["output_results_path"]).parent.mkdir(parents=True, exist_ok=True)
        results_json = {
            "model_name": "strats",
            "c_index": metrics_by_split,
            "landmark_55_calibration": {
                "H0_train": float(H0),
                "max_r_train": float(max_r),
                "splits": calibration_results,
            },
        }
        with open(cfg["output_results_path"], "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
        print(f"Métricas guardadas en: {cfg['output_results_path']}")

    return model, metrics_by_split


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ejecución Independiente (Configuración Automática: Local / Colab)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STraTS: Train & Evaluate")
    parser.add_argument(
        "--env",
        choices=["auto", "local", "colab"],
        default="auto",
        help="Entorno de ejecución ('local', 'colab', o 'auto')",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Sobrescribir el número de épocas de entrenamiento",
    )
    cli_args, _ = parser.parse_known_args()

    # Detección del entorno (Colab vs Local)
    if cli_args.env == "colab":
        is_colab = True
    elif cli_args.env == "local":
        is_colab = False
    else:
        # Auto: si existe /content o la ruta de Drive
        is_colab = Path("/content").exists() and not Path(__file__).resolve().parent.parent.parent.joinpath("results").exists()

    if is_colab:
        base_dir = Path("/content/drive/MyDrive/TFM/us-mulcia-2026-ramon")
        print("[Entorno] Detectado Google Colab. Base dir:", base_dir)
    else:
        base_dir = Path(__file__).resolve().parent.parent.parent
        print("[Entorno] Detectado Entorno Local. Base dir:", base_dir)

    CONFIG = {
        # Rutas de entrada
        "train_path": str(base_dir / "results" / "train" / "sequences.pkl"),
        "val_path": str(base_dir / "results" / "val" / "sequences.pkl"),
        "holdout_path": str(base_dir / "results" / "holdout" / "sequences.pkl")
        if (base_dir / "results" / "holdout" / "sequences.pkl").exists()
        else None,
        # Rutas de salida
        "output_model_path": str(base_dir / "results" / "models" / "strats.pkl"),
        "output_results_path": str(base_dir / "results" / "metrics" / "strats.json"),
        # Hiperparámetros de entrenamiento
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": cli_args.max_epochs if cli_args.max_epochs is not None else 30,
        "patience": 5,
        "event_oversample": 3.0,
        # Parámetros temporales y de cohortes
        "index_age": 50.0,
        "landmark_min": 45,
        "landmark_max": 70,
        "horizon": 5.0,
        "max_events": 200,
        "age_weight_gamma": 1.0,
        # Arquitectura STraTS
        "hid_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.2,
        "demo_dim": 8,
    }

    train_and_evaluate_strats(CONFIG)
