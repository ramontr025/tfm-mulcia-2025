"""Snakemake ``script:`` for split_cohort (Stage 1).

Reparte la cohorte de discovery en train/val, a nivel de paciente, estratificado por
evento y con semilla fija. Es el UNICO paso especifico de discovery: holdout no se
reparte, ya es su propio pool. Va aparte de build_cohort a proposito, para que el
preprocesamiento (build_cohort) sea identico para los dos pools y el reparto quede
como un paso claramente posterior.

El reparto tiene que ocurrir AQUI, antes de build_sequences, y no despues: las
secuencias se normalizan (z-score) con la media y desviacion de train, asi que hay
que saber quien es train antes de construir ninguna secuencia.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no se importa: el
bloque TYPE_CHECKING de abajo solo le da tipo al linter/IDE, nunca se ejecuta).

Reads:  snakemake.input.discovery       results/discovery/cohort.parquet.
Writes: snakemake.output.{train,val}    una cohorte por split.
"""

import sys
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake

# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): cada regla deja su
# propio rastro en logs/, con los prints y cualquier traceback.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

cohort = pd.read_parquet(snakemake.input.discovery)
train_df, val_df = train_test_split(
    cohort,
    test_size=(1.0 - float(snakemake.params.train_frac)),
    random_state=int(snakemake.params.seed),
    stratify=cohort["event"],
)

train_df.reset_index(drop=True).to_parquet(snakemake.output.train, index=False)
val_df.reset_index(drop=True).to_parquet(snakemake.output.val, index=False)
print(
    f"[train] {len(train_df)} pacientes  event_rate={train_df['event'].mean():.3f}",
    flush=True,
)
print(
    f"[val]   {len(val_df)} pacientes  event_rate={val_df['event'].mean():.3f}",
    flush=True,
)
