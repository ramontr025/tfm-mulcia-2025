import copy
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from loss import cox_ph_loss


class SubsetDataset:
    """Helper ligero para pasar una lista de samples a build_locf_matrix."""

    def __init__(self, samples, code_to_idx):
        self.samples = samples
        self.code_to_idx = code_to_idx

    def __len__(self):
        return len(self.samples)


def compute_lab_medians(dataset, lab_dim: int) -> np.ndarray:
    """
    Calcula la mediana de cada laboratorio a lo largo de todo el dataset (ej. train),
    utilizando únicamente las observaciones reales (donde mask == 1).
    """
    medians = np.zeros(lab_dim, dtype=np.float32)
    for j in range(lab_dim):
        valid_vals = []
        for s in dataset.samples:
            m = s["masks"][:, j] == 1
            if np.any(m):
                valid_vals.append(s["labs"][m, j])
        if valid_vals:
            concat_vals = np.concatenate(valid_vals)
            medians[j] = np.median(concat_vals)
        else:
            medians[j] = 0.0
    return medians


def build_locf_matrix(
    dataset,
    lab_dim: int,
    median_x: np.ndarray = None,
    use_icd: bool = False,
    num_icd: int | None = None,
) -> np.ndarray:
    """
    Construye la matriz de características X (N, D) para el modelo Cox baseline.

    Dimensión final de X:
    - Si use_icd=False: (N, lab_dim + 1)
    - Si use_icd=True:  (N, lab_dim + 1 + num_icd)
    """
    if median_x is None:
        median_x = compute_lab_medians(dataset, lab_dim)

    if use_icd:
        if num_icd is None:
            if hasattr(dataset, "code_to_idx") and dataset.code_to_idx:
                num_icd = len(dataset.code_to_idx)
            else:
                num_icd = 0
    else:
        num_icd = 0

    n = len(dataset)
    total_dim = lab_dim + 1 + (num_icd if use_icd else 0)
    X = np.zeros((n, total_dim), dtype=np.float32)
    col_indices = np.arange(lab_dim)

    for i in range(n):
        sample = dataset.samples[i]
        labs = sample["labs"]  # (T, lab_dim)
        masks = sample["masks"]  # (T, lab_dim)
        sex_male = float(sample.get("sex_male", 0.0))

        # 1. Laboratorios (LOCF)
        reversed_labs = labs[::-1]
        reversed_masks = masks[::-1]

        has_obs = np.any(reversed_masks == 1, axis=0)
        last_obs_idx = np.argmax(reversed_masks == 1, axis=0)

        locf_labs = reversed_labs[last_obs_idx, col_indices]
        imputed_labs = np.where(has_obs, locf_labs, median_x)

        # 2. Vector base: [labs, sexo]
        feature_list = [imputed_labs, np.array([sex_male], dtype=np.float32)]

        # 3. ICD-10 multi-hot (opcional)
        if use_icd and num_icd > 0:
            icd_onehot = np.zeros(num_icd, dtype=np.float32)
            for step_codes in sample.get("codes_idx", []):
                for code_idx in step_codes:
                    if 1 <= code_idx <= num_icd:
                        icd_onehot[code_idx - 1] = 1.0
            feature_list.append(icd_onehot)

        X[i] = np.concatenate(feature_list)

    return X


class CoxPHNet(nn.Module):
    """
    Modelo lineal para regresión de Cox:
    log-hazard = X @ beta
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def run_cox_epoch(
    loader: DataLoader, model: nn.Module, device: torch.device, optimizer=None
):
    """
    Ejecuta una época de entrenamiento o validación para el modelo Cox tabular.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_risks, all_dur, all_ev = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_b, dur_b, ev_b, w_b in loader:
            x_b = x_b.to(device)
            dur_b = dur_b.to(device)
            ev_b = ev_b.to(device)
            w_b = w_b.to(device)

            risk_scores = model(x_b)  # (B,)
            loss = cox_ph_loss(risk_scores, dur_b, ev_b, w_b)

            if is_train and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_risks.extend(risk_scores.detach().cpu().numpy())
            all_dur.extend(dur_b.cpu().numpy())
            all_ev.extend(ev_b.cpu().numpy())

    mean_loss = total_loss / max(len(loader), 1)

    all_risks = np.array(all_risks)
    all_dur = np.array(all_dur)
    all_ev = np.array(all_ev, dtype=bool)

    if np.sum(all_ev) > 0:
        ci = float(concordance_index(all_dur, -all_risks, all_ev))
    else:
        ci = 0.5

    return mean_loss, ci


def train_cox(
    train_dataset,
    val_dataset,
    landmark_ages: list[int],
    lab_codes: list[str],
    code_to_idx: dict,
    use_icd: bool,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 128,
    event_oversample: float = 1.0,
    device: torch.device = None,
) -> dict:
    """
    Entrena un modelo Cox separado para cada landmark_age en landmark_ages.
    Retorna el diccionario con el artefacto consolidado con todos los modelos.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lab_dim = train_dataset.lab_dim
    num_icd = len(code_to_idx) if (use_icd and code_to_idx) else 0
    input_dim = lab_dim + 1 + num_icd

    print("Calculando medianas de laboratorios en Train...")
    median_labs = compute_lab_medians(train_dataset, lab_dim)

    state_dicts = {}
    print(
        f"\nIniciando entrenamiento Cox para {len(landmark_ages)} landmarks (input_dim={input_dim}, use_icd={use_icd})..."
    )

    for lm in landmark_ages:
        lm_float = float(lm)
        train_samples_lm = [
            s for s in train_dataset.samples if s["landmark_age"] == lm_float
        ]
        val_samples_lm = [
            s for s in val_dataset.samples if s["landmark_age"] == lm_float
        ]

        if not train_samples_lm:
            print(f"  [Landmark {lm}] Sin muestras de entrenamiento. Omitiendo.")
            continue

        ev_train_lm = np.array([s["event"] for s in train_samples_lm], dtype=np.float32)

        # Si no hay ningún evento en este landmark, inicializamos a ceros y omitimos optimización
        if np.sum(ev_train_lm) == 0:
            model_lm = CoxPHNet(input_dim=input_dim).to(device)
            nn.init.zeros_(model_lm.linear.weight)
            if model_lm.linear.bias is not None:
                nn.init.zeros_(model_lm.linear.bias)
            state_dicts[str(lm)] = model_lm.state_dict()
            continue

        train_sub = SubsetDataset(train_samples_lm, code_to_idx)
        val_sub = SubsetDataset(val_samples_lm, code_to_idx)

        X_train_lm = build_locf_matrix(
            train_sub, lab_dim, median_x=median_labs, use_icd=use_icd, num_icd=num_icd
        )
        dur_train_lm = np.array(
            [s["duration"] for s in train_samples_lm], dtype=np.float32
        )
        w_train_lm = np.array([s["weight"] for s in train_samples_lm], dtype=np.float32)

        X_val_lm = (
            build_locf_matrix(
                val_sub, lab_dim, median_x=median_labs, use_icd=use_icd, num_icd=num_icd
            )
            if val_samples_lm
            else np.zeros((0, input_dim), dtype=np.float32)
        )
        dur_val_lm = np.array([s["duration"] for s in val_samples_lm], dtype=np.float32)
        ev_val_lm = np.array([s["event"] for s in val_samples_lm], dtype=np.float32)
        w_val_lm = np.array([s["weight"] for s in val_samples_lm], dtype=np.float32)

        train_tensors = TensorDataset(
            torch.tensor(X_train_lm, dtype=torch.float32),
            torch.tensor(dur_train_lm, dtype=torch.float32),
            torch.tensor(ev_train_lm, dtype=torch.float32),
            torch.tensor(w_train_lm, dtype=torch.float32),
        )
        if event_oversample > 1.0:
            sample_weights = [event_oversample if e == 1 else 1.0 for e in ev_train_lm]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
            train_loader = DataLoader(
                train_tensors, batch_size=batch_size, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_tensors, batch_size=batch_size, shuffle=True
            )

        if len(val_samples_lm) > 0:
            val_tensors = TensorDataset(
                torch.tensor(X_val_lm, dtype=torch.float32),
                torch.tensor(dur_val_lm, dtype=torch.float32),
                torch.tensor(ev_val_lm, dtype=torch.float32),
                torch.tensor(w_val_lm, dtype=torch.float32),
            )
            val_loader = DataLoader(val_tensors, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        model_lm = CoxPHNet(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(
            model_lm.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, max_epochs + 1):
            train_loss, train_ci = run_cox_epoch(
                train_loader, model_lm, device, optimizer=optimizer
            )

            if val_loader is not None:
                val_loss, val_ci = run_cox_epoch(
                    val_loader, model_lm, device, optimizer=None
                )
                scheduler.step(val_loss)
                target_loss = val_loss
            else:
                target_loss = train_loss

            if target_loss < best_val_loss - 1e-4:
                best_val_loss = target_loss
                best_state = copy.deepcopy(model_lm.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model_lm.load_state_dict(best_state)

        state_dicts[str(lm)] = model_lm.state_dict()

    print(
        f"-> Entrenamiento Cox finalizado. Total de modelos guardados: {len(state_dicts)}"
    )

    artifact = {
        "arch": "cox",
        "lab_dim": lab_dim,
        "input_dim": input_dim,
        "use_icd": use_icd,
        "num_icd": num_icd,
        "median_x": median_labs,
        "lab_codes": lab_codes,
        "code_to_idx": code_to_idx,
        "state_dicts": state_dicts,
        "landmark_ages": landmark_ages,
    }
    return artifact


def predict_cox_landmarks(
    artifact: dict,
    seqs: dict,
    cohort,
    landmark_times: list,
    index_age: float,
    device: torch.device = None,
):
    """
    Reconstruye los modelos Cox por landmark del artefacto (uno por edad) y devuelve
    sus predicciones concatenadas en el mismo formato que `engine.collect_raw_outputs`:
    (risks, durs, evs, lms, ids). Así eval.py puede tratar Cox igual que los modelos
    secuenciales (GRU-D/LSTM) sin una rama de evaluación separada.
    """
    from dataset import RawSequenceDatasetWithICD

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lab_dim = artifact["lab_dim"]
    input_dim = artifact["input_dim"]
    use_icd = artifact.get("use_icd", False)
    num_icd = artifact.get("num_icd", 0)
    median_labs = artifact["median_x"]
    code_to_idx = artifact["code_to_idx"]
    state_dicts = artifact.get("state_dicts", {})

    all_risks, all_durs, all_evs, all_lms, all_ids = [], [], [], [], []

    for lm in landmark_times:
        lm_str = str(lm)
        if lm_str not in state_dicts:
            continue

        model_lm = CoxPHNet(input_dim=input_dim).to(device)
        model_lm.load_state_dict(state_dicts[lm_str])
        model_lm.eval()

        ds_lm = RawSequenceDatasetWithICD(
            sequences=seqs,
            cohort_df=cohort,
            landmark_ages=[lm],
            index_age=index_age,
            code_to_idx=code_to_idx,
            max_events=200,
            age_weight_gamma=1.0,
        )
        if len(ds_lm) == 0:
            continue

        X_lm = build_locf_matrix(
            ds_lm, lab_dim, median_x=median_labs, use_icd=use_icd, num_icd=num_icd
        )
        with torch.no_grad():
            x_tensor = torch.tensor(X_lm, dtype=torch.float32).to(device)
            risks_lm = model_lm(x_tensor).cpu().numpy()

        n_lm = len(ds_lm)
        all_risks.append(risks_lm.astype(np.float64))
        all_durs.append(
            np.array([s["duration"] for s in ds_lm.samples], dtype=np.float64)
        )
        all_evs.append(np.array([s["event"] for s in ds_lm.samples], dtype=np.float64))
        all_lms.append(np.full(n_lm, float(lm), dtype=np.float64))
        all_ids.append(np.array([s["pid"] for s in ds_lm.samples], dtype=object))

    if not all_risks:
        empty_f = np.array([], dtype=np.float64)
        return (
            empty_f,
            empty_f.copy(),
            empty_f.copy(),
            empty_f.copy(),
            np.array([], dtype=object),
        )

    return (
        np.concatenate(all_risks),
        np.concatenate(all_durs),
        np.concatenate(all_evs),
        np.concatenate(all_lms),
        np.concatenate(all_ids),
    )
