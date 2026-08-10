import copy
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Permitir importar módulos locales desde workflow/scripts
sys.path.append(str(Path(__file__).parent))


# Leer parámetros e hiperparámetros inyectados desde Snakemake
arch_name = snakemake.params.arch
icd = int(snakemake.params.icd)
embed_dim = 16 if icd else 0

HIDDEN_DIM = int(snakemake.params.hidden_dim)
BATCH_SIZE = int(snakemake.params.batch_size)
LEARNING_RATE = float(snakemake.params.learning_rate)
WEIGHT_DECAY = float(snakemake.params.weight_decay)
MAX_EPOCHS = int(snakemake.params.max_epochs)
PATIENCE = int(snakemake.params.patience)
EVENT_OVERSAMPLE = float(snakemake.params.event_oversample)
max_events = int(snakemake.params.max_events)
age_weight_gamma = float(snakemake.params.age_weight_gamma)

index_age = float(snakemake.params.index_age)
landmark_min = int(snakemake.params.landmark_min)
landmark_max = int(snakemake.params.landmark_max)



if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake

from dataset import RawSequenceDatasetWithICD, raw_pad_collate_with_icd, load_sequences
from engine import run_raw_loss_epoch_icd
from model import TimeAwareGRUD


# ── Cuerpo del script ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): cada regla deja su
# propio rastro en logs/, con los prints y cualquier traceback.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Cargar el pickle de secuencias de entrada (train, val, holdout, etc.)

sequences_train, lab_codes, cohort_train = load_sequences(snakemake.input.train)
sequences_val, _, cohort_val = load_sequences(snakemake.input.val)

# 2. Parámetros inyectados desde Snakemake
landmark_ages = list(getattr(snakemake.params, "landmark_ages", getattr(snakemake.params, "landmark_age", range(45, 71))))
index_age = float(getattr(snakemake.params, "index_age", 45.0))
max_events = int(getattr(snakemake.params, "max_events", 200))
age_weight_gamma = float(getattr(snakemake.params, "age_weight_gamma", 1.0))


# 3. Vocabulario code_to_idx dinámico a partir de los códigos ICD-10 de las secuencias
if hasattr(snakemake.params, "code_to_idx") and snakemake.params.code_to_idx:
    code_to_idx = snakemake.params.code_to_idx
else:
    print("Construyendo vocabulario de códigos ICD-10...")
    all_icd_codes = set()
    for pid, seq in sequences_train.items():
        for step_codes in seq["codes"]:
            for code in step_codes:
                all_icd_codes.add(code)         
# Mapeamos cada código único a un entero empezando en 1 (el 0 se reserva para padding/sin códigos)
code_to_idx = {code: i + 1 for i, code in enumerate(sorted(all_icd_codes))}
vocab_size = len(code_to_idx) + 1
print(f"-> Vocabulario creado: {len(code_to_idx)} códigos únicos. vocab_size = {vocab_size}")

landmark_ages=list(range(int(landmark_min), int(landmark_max)))

# 4. Instanciación del dataset para la entrada actual
train_dataset_icd = RawSequenceDatasetWithICD(
    sequences=sequences_train,
    cohort_df=cohort_train,
    landmark_ages=landmark_ages,
    index_age=index_age,
    code_to_idx=code_to_idx,
    max_events=max_events,
    age_weight_gamma=age_weight_gamma,
)

val_dataset_icd = RawSequenceDatasetWithICD(
    sequences=sequences_val,
    cohort_df=cohort_val,
    landmark_ages=landmark_ages,
    index_age=index_age,
    code_to_idx=code_to_idx,
    max_events=max_events,
    age_weight_gamma=age_weight_gamma,
)

mean_x = np.zeros(train_dataset_icd.lab_dim, dtype=np.float32)

# 5. Definición del modelo

# Registro de modelos soportados (facilita añadir LSTM, Transformer, etc. en futuros issues)
MODEL_REGISTRY = {
    "grud": TimeAwareGRUD,
}

arch_name = snakemake.params.arch       # Ej: "grud"
icd = int(snakemake.params.icd)         # Ej: 1 o 0
embed_dim = 16 if icd else 0            # La opción de ablación ICD de la config

if arch_name not in MODEL_REGISTRY:
    raise ValueError(f"Arquitectura desconocida '{arch_name}'. Soportadas: {list(MODEL_REGISTRY.keys())}")

ModelClass = MODEL_REGISTRY[arch_name]

model = ModelClass(
    input_dim=train_dataset_icd.lab_dim,
    vocab_size=vocab_size,
    mean_x= mean_x,
    embed_dim=embed_dim,
    hidden_dim=HIDDEN_DIM,
    dropout=0.2,
).to(device)


# 6. Entrenamiento


# Event oversampling: give event patients EVENT_OVERSAMPLE× more draw probability
sample_weights = [EVENT_OVERSAMPLE if f == 1 else 1.0 for f in train_dataset_icd.event_flags]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset_icd, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=raw_pad_collate_with_icd)
val_loader   = DataLoader(val_dataset_icd,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=raw_pad_collate_with_icd)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")
best_state    = None
no_improve    = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # ── TODO 5 ── Training step ───────────────────────────────────────────────
    # Call run_raw_loss_epoch in training mode to run one full epoch and get
    # (train_loss, train_ci). Replace the two placeholder lines below.
    #
    # Hint: check the function signature above — what argument switches it
    #       between training and evaluation mode?
    #
    train_loss, train_ci = run_raw_loss_epoch_icd(train_loader, model, device, optimizer=optimizer)  # replace with the real call
    # ─────────────────────────────────────────────────────────────────────────

    val_loss, val_ci = run_raw_loss_epoch_icd(val_loader, model, device, optimizer=None)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
          f"train_loss={train_loss:.4f}  train_ci={train_ci:.4f}  "
          f"val_loss={val_loss:.4f}  val_ci={val_ci:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state    = copy.deepcopy(model.state_dict())
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  → early stopping at epoch {epoch}")
            break

# 7. Guardar el artefacto del modelo entrenado en snakemake.output[0]
if best_state is not None:
    model.load_state_dict(best_state)

print(f"\nBest val loss: {best_val_loss:.4f}")

artifact = {
    "lab_dim":     train_dataset_icd.lab_dim,
    "hidden_dim":  HIDDEN_DIM,
    "vocab_size":  vocab_size,
    "embed_dim":   embed_dim,
    "mean_x":      mean_x,
    "lab_codes":   lab_codes,
    "code_to_idx": code_to_idx,
    "state_dict":  model.state_dict(),
}

with open(snakemake.output[0], "wb") as f:
    pickle.dump(artifact, f)

print(f"[{snakemake.wildcards.name}] Artefacto guardado correctamente en: {snakemake.output[0]}", flush=True)
