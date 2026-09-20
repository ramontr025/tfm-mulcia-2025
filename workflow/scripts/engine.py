import numpy as np
import torch
from lifelines.utils import concordance_index

from loss import cox_ph_loss


def run_raw_loss_epoch_icd(loader, model, device, optimizer=None):
    """
    One forward pass over the dataset.

    Returns (mean_loss, c_index).
    If optimizer is not None, also runs backward + step (training mode).

    Batch format: (labs, masks, deltas, lengths, sex_male, time_since_index,
                   durations, events, landmarks, weights, ids, flat_codes, offsets).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_risks, all_dur, all_ev = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            (
                labs,
                masks,
                deltas,
                lengths,
                sex_male,
                time_since_index,
                durations,
                events,
                landmarks,
                weights,
                ids,
                flat_codes,
                offsets,
            ) = batch

            labs = labs.to(device)
            masks = masks.to(device)
            deltas = deltas.to(device)
            lengths = lengths.to(device)
            sex_male = sex_male.to(device)
            time_since_index = time_since_index.to(device)
            durations = durations.to(device)
            events = events.to(device)
            weights = weights.to(device)
            flat_codes = flat_codes.to(device)
            offsets = offsets.to(device)

            # ── TODO 7 ── Forward pass, loss, and backward ───────────────────
            # 1. Call the model to get risk_scores.
            #    Signature: model(labs, masks, deltas, lengths, sex_male, time_since_index)
            #    Returns (last_risk, all_risks, h_states) — you only need the first.
            (risk_scores, _, _) = model(
                labs,
                masks,
                deltas,
                lengths,
                sex_male,
                time_since_index,
                flat_codes,
                offsets,
            )
            #
            # 2. Compute the loss with cox_ph_loss(risk_scores, durations, events, weights).
            loss = cox_ph_loss(risk_scores, durations, events, weights)
            #
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

    # ── TODO 8 ── C-index ────────────────────────────────────────────────────
    # Compute the concordance index (C-index) from the accumulated arrays:
    #   all_risks  — model log-risk scores  (higher = more risk)
    #   all_dur    — time from landmark to event or censoring
    #   all_ev     — event indicator (1 = lung cancer, 0 = censored)
    #
    # Use lifelines.utils.concordance_index(event_times, predicted_scores, event_observed).
    #
    # Think before you code:
    #   - concordance_index expects *higher* predicted_scores to mean *longer* survival.
    #     Our model outputs log-risk: higher risk → shorter survival.
    #     How should you transform all_risks before passing it?
    #   - The function expects event_observed as a boolean array.
    #
    ci = concordance_index(
        np.array(all_dur), -np.array(all_risks), np.array(all_ev, dtype=bool)
    )
    return mean_loss, ci


def compute_baseline_hazard(risks, durations, events, horizon):
    """Breslow estimator: H0(horizon) from training data."""
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


def collect_raw_outputs(loader, model, device):
    """
    Collect model outputs over a full loader.

    Returns (risks, durations, events, landmarks, patient_ids) as numpy arrays
    (patient_ids as an object array of strings, for patient-level bootstrap CIs).
    """
    model.eval()
    risks, durs, evs, lms, pids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            (
                labs,
                masks,
                deltas,
                lengths,
                sex_male,
                time_since_index,
                durations,
                events,
                landmarks,
                weights,
                ids,
                flat_codes,
                offsets,
            ) = batch
            r, _, _ = model(
                labs.to(device),
                masks.to(device),
                deltas.to(device),
                lengths.to(device),
                sex_male.to(device),
                time_since_index.to(device),
                flat_codes.to(device),
                offsets.to(device),
            )
            risks.extend(r.cpu().numpy())
            durs.extend(durations.numpy())
            evs.extend(events.numpy())
            lms.extend(landmarks.numpy())
            pids.extend(ids)
    return (
        np.array(risks),
        np.array(durs),
        np.array(evs, dtype=float),
        np.array(lms),
        np.array(pids, dtype=object),
    )

# Definición del engine para STraTS

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

def predict_STraTS(
    artifact: dict,
    seqs: dict,
    cohort,
    landmark_times: list,
    index_age: float,
    device: torch.device = None,
):
    """
    Reconstruye el modelo STraTS y devuelve sus predicciones en el mismo formato que `engine.collect_raw_outputs`:
    (risks, durs, evs, lms, ids). Así eval.py puede tratar STraTS igual que los modelos
    secuenciales (GRU-D/LSTM) sin una rama de evaluación separada.
    """
    from dataset import STraTSDataset, strats_pad_collate
    from model import STraTS
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = artifact["vocab_size"]
    hidden_dim = artifact["hidden_dim"]
    num_heads = artifact["num_heads"]
    num_layers = artifact["num_layers"]
    dim_feedforward = artifact["dim_feedforward"]
    dropout = artifact["dropout"]
    demo_dim = artifact["demo_dim"]
    lab_dim = artifact["lab_dim"]
    code_to_idx = artifact["code_to_idx"]
    state_dict = artifact["state_dict"]

    model = STraTS(
        vocab_size=vocab_size,
        lab_dim=lab_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        demo_dim=demo_dim,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    ds = STraTSDataset(
        sequences=seqs,
        cohort_df=cohort,
        landmark_ages=landmark_times,
        index_age=index_age,
        code_to_idx=code_to_idx,
        max_events=200,
        age_weight_gamma=1.0,
    )
    if len(ds) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([], dtype=float),
            np.array([]),
            np.array([], dtype=object),
        )

    loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        collate_fn=strats_pad_collate,
    )

    risks, durs, evs, lms, pids = [], [], [], [], []
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
            pids.extend(ids)
    return (
        np.array(risks),
        np.array(durs),
        np.array(evs, dtype=float),
        np.array(lms),
        np.array(pids, dtype=object),
    )