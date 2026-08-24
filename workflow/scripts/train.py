from model import SimpleLSTM
import copy
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Permitir importar módulos locales desde workflow/scripts
sys.path.append(str(Path(__file__).parent))

from cox import CoxPHNet, train_cox
from dataset import RawSequenceDatasetWithICD, raw_pad_collate_with_icd, load_sequences
from engine import run_raw_loss_epoch_icd
from model import TimeAwareGRUD


if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake


# ── Cuerpo del script ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0])
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

# 1. Leer parámetros e hiperparámetros inyectados desde Snakemake
arch_name = snakemake.params.arch
icd = int(snakemake.params.icd)
embed_dim = 16 if icd else 0

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Cargar el pickle de secuencias de entrada (train, val, holdout, etc.)
sequences_train, lab_codes, cohort_train = load_sequences(snakemake.input.train)
sequences_val, _, cohort_val = load_sequences(snakemake.input.val)

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
print(
    f"-> Vocabulario creado: {len(code_to_idx)} códigos únicos. vocab_size = {vocab_size}"
)

landmark_ages = list(range(int(landmark_min), int(landmark_max) + 1))

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

# 5. Definición y entrenamiento según el modelo
# ── Rama de entrenamiento para Cox Baseline (tabular / LOCF) ──
if arch_name == "cox":
    print("\n── Entrenando Modelo Cox Baseline ──")
    artifact = train_cox(
        train_dataset=train_dataset_icd,
        val_dataset=val_dataset_icd,
        landmark_ages=landmark_ages,
        lab_codes=lab_codes,
        code_to_idx=code_to_idx,
        use_icd=bool(icd),
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        event_oversample=EVENT_OVERSAMPLE,
        device=device,
    )

    with open(snakemake.output[0], "wb") as f:
        pickle.dump(artifact, f)

    print(
        f"[{snakemake.wildcards.name}] Artefacto Cox guardado correctamente en: {snakemake.output[0]}",
        flush=True,
    )
    sys.exit(0)

# ── Rama de entrenamiento para Modelos Secuenciales (GRU-D, LSTM) ──

HIDDEN_DIM = int(snakemake.params.hidden_dim) # Lo leemos aqui para que no explote en el caso de Cox

all_labs = [s["labs"] for s in train_dataset_icd.samples]
if all_labs:
    all_labs_concat = np.concatenate(all_labs, axis=0)
    mean_x = all_labs_concat.mean(axis=0)
else:
    mean_x = np.zeros(train_dataset_icd.lab_dim, dtype=np.float32)

def build_model(arch, input_dim, vocab_size, mean_x, embed_dim, hidden_dim, dropout=0.2):
    if arch == "grud":
        return TimeAwareGRUD(
            input_dim=input_dim,
            vocab_size=vocab_size,
            mean_x=mean_x,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif arch == "lstm":
        return SimpleLSTM(
            input_dim=input_dim,
            vocab_size=vocab_size,
            mean_x=mean_x,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Arquitectura secuencial no soportada: {arch}")

# Instanciación
model = build_model(
    arch=arch_name,
    input_dim=train_dataset_icd.lab_dim,
    vocab_size=vocab_size,
    mean_x=mean_x,
    embed_dim=embed_dim,
    hidden_dim=HIDDEN_DIM,
    dropout=0.2,
).to(device)



# 6. Entrenamiento


# Event oversampling: give event patients EVENT_OVERSAMPLE× more draw probability
sample_weights = [
    EVENT_OVERSAMPLE if f == 1 else 1.0 for f in train_dataset_icd.event_flags
]
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(
    train_dataset_icd,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=raw_pad_collate_with_icd,
)
val_loader = DataLoader(
    val_dataset_icd,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=raw_pad_collate_with_icd,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

best_val_loss = float("inf")
best_state = None
no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_ci = run_raw_loss_epoch_icd(
        train_loader, model, device, optimizer=optimizer
    )  # replace with the real call
    val_loss, val_ci = run_raw_loss_epoch_icd(val_loader, model, device, optimizer=None)
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
        f"train_loss={train_loss:.4f}  train_ci={train_ci:.4f}  "
        f"val_loss={val_loss:.4f}  val_ci={val_ci:.4f}"
    )

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
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
    "arch": arch_name,
    "lab_dim": train_dataset_icd.lab_dim,
    "hidden_dim": HIDDEN_DIM,
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "mean_x": mean_x,
    "lab_codes": lab_codes,
    "code_to_idx": code_to_idx,
    "state_dict": model.state_dict(),
}

with open(snakemake.output[0], "wb") as f:
    pickle.dump(artifact, f)

print(
    f"[{snakemake.wildcards.name}] Artefacto guardado correctamente en: {snakemake.output[0]}",
    flush=True,
)
