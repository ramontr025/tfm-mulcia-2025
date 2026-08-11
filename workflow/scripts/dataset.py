import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_sequences(pkl_path: Path) -> tuple[dict, list[str], object]:
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    sequences = {pid: seq for pid, seq in cache["sequences"].items() if seq is not None}
    return sequences, list(cache["lab_codes"]), cache["cohort"]


    
class RawSequenceDatasetWithICD(Dataset):
    """
    One sample per (patient, landmark_age) pair, including ICD-10 code indices.
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
        self.samples     = []
        self.event_flags = []
        self.lab_dim     = None
        self.code_to_idx = code_to_idx
        lm_arr = np.array(landmark_ages, dtype=np.float32)
        min_lm, max_lm = float(lm_arr.min()), float(lm_arr.max())
        for row in cohort_df.itertuples(index=False):
            pid      = str(row.Id)
            if pid not in sequences:
                continue
            duration = float(row.duration)
            event    = float(row.event)
            seq      = sequences[pid]
            times    = np.asarray(seq["times"],  dtype=np.float32)
            labs     = np.asarray(seq["labs"],   dtype=np.float32)
            masks    = np.asarray(seq["masks"],  dtype=np.float32)
            sex_male = float(seq["sex_male"])
            codes    = seq["codes"] # Lista de listas de strings (ICD-10)
            if self.lab_dim is None:
                self.lab_dim = labs.shape[1]
            for lm in landmark_ages:
                time_from_lm = duration - (lm - index_age)
                if time_from_lm <= 0:
                    continue
                keep = times <= lm
                if not np.any(keep):
                    times_s = np.array([lm], dtype=np.float32)
                    labs_s  = np.zeros((1, self.lab_dim), dtype=np.float32)
                    masks_s = np.zeros((1, self.lab_dim), dtype=np.float32)
                    codes_s = [[]]
                else:
                    times_s = times[keep]
                    labs_s  = labs[keep]
                    masks_s = masks[keep]
                    codes_s = [c for k, c in zip(keep, codes) if k]
                # Aplicar truncación por longitud máxima
                if max_events and len(times_s) > max_events:
                    times_s = times_s[-max_events:]
                    labs_s  = labs_s[-max_events:]
                    masks_s = masks_s[-max_events:]
                    codes_s = codes_s[-max_events:]
                time_since_index_s = times_s - float(index_age)
                # Mapear strings ICD-10 a índices enteros del vocabulario
                codes_idx_s = [[self.code_to_idx.get(code, 0) for code in step] for step in codes_s]
                # Peso basado en la edad del landmark
                lm_norm = (lm - min_lm) / max(max_lm - min_lm, 1e-6)
                weight  = 1.0 + age_weight_gamma * lm_norm
                self.samples.append({
                    "times":            times_s,
                    "labs":             labs_s,
                    "masks":            masks_s,
                    "sex_male":         sex_male,
                    "time_since_index": time_since_index_s,
                    "duration":         time_from_lm,
                    "event":            event,
                    "landmark_age":     float(lm),
                    "weight":           weight,
                    "pid":              pid,
                    "codes_idx":        codes_idx_s, # lista de listas de enteros
                })
                self.event_flags.append(int(event))
        if self.lab_dim is None:
            self.lab_dim = 0
        n = len(self.samples)
        self.event_rate   = sum(self.event_flags) / max(n, 1)
        self.mean_seq_len = float(np.mean([len(s["times"]) for s in self.samples])) if n else 0.0
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["times"],            dtype=torch.float32),
            torch.tensor(s["labs"],             dtype=torch.float32),
            torch.tensor(s["masks"],            dtype=torch.float32),
            torch.tensor(s["sex_male"],         dtype=torch.float32),
            torch.tensor(s["time_since_index"], dtype=torch.float32),
            torch.tensor(s["duration"],         dtype=torch.float32),
            torch.tensor(s["event"],            dtype=torch.float32),
            torch.tensor(s["landmark_age"],     dtype=torch.float32),
            torch.tensor(s["weight"],           dtype=torch.float32),
            s["pid"],
            s["codes_idx"], # Retornamos la estructura de códigos
        )

def raw_pad_collate_with_icd(batch):
    """
    Collate function that structures ICD-10 bags using flat_codes and offsets.
    """
    times, labs, masks, sex_male, time_since_index, durations, events, landmarks, weights, ids, batch_codes = zip(*batch)
    lengths = torch.tensor([len(t) for t in times])
    max_len = int(lengths.max())
    padded_labs             = nn.utils.rnn.pad_sequence(labs,             batch_first=True, padding_value=0.0)
    padded_masks            = nn.utils.rnn.pad_sequence(masks,            batch_first=True, padding_value=0.0)
    padded_times            = nn.utils.rnn.pad_sequence(times,            batch_first=True, padding_value=0.0)
    padded_time_since_index = nn.utils.rnn.pad_sequence(time_since_index, batch_first=True, padding_value=0.0)
    deltas = torch.zeros_like(padded_times)
    deltas[:, 1:] = torch.clamp(padded_times[:, 1:] - padded_times[:, :-1], min=0.0)
    # Construcción de flat_codes y offsets para EmbeddingBag
    flat_codes = []
    offsets = []
    current_offset = 0
    for i in range(len(batch)):
        patient_codes = batch_codes[i] # Lista de listas de longitud seq_len
        seq_len = len(times[i])

        # Iteramos por cada timestep hasta la longitud máxima del batch (rellenando con 0 si es padding)
        for t in range(max_len):
            offsets.append(current_offset)
            if t < seq_len:
                step_codes = patient_codes[t]
                if not step_codes: # Si la lista está vacía en este timestep
                    step_codes = [0]
            else:
                step_codes = [0] # Timesteps de padding para alinear la matriz

            flat_codes.extend(step_codes)
            current_offset += len(step_codes)
    flat_codes_t = torch.tensor(flat_codes, dtype=torch.long)
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    return (
        padded_labs,            # x        (B, T, D)
        padded_masks,           # mask     (B, T, D)
        deltas,                 # delta_t  (B, T)
        lengths,                # lengths  (B,)
        torch.stack(sex_male),
        padded_time_since_index,
        torch.stack(durations),
        torch.stack(events),
        torch.stack(landmarks),
        torch.stack(weights),
        list(ids),
        flat_codes_t,           # flat_codes para el modelo (EmbeddingBag input)
        offsets_t,              # offsets para el modelo    (EmbeddingBag offsets)
    )


