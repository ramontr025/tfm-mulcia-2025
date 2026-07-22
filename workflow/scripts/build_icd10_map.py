"""Snakemake ``script:`` for build_icd10_map (Stage 1).

Los diagnosticos crudos de Synthea vienen en SNOMED CT (conditions.CODE). El modelo,
sin embargo, trabaja con capitulos ICD-10: un vocabulario mucho mas pequeno y denso
(unas decenas de capitulos, no cientos de miles de conceptos SNOMED). Esta regla lee
el mapa SNOMED->ICD-10 (~100 MB, fuera de git) UNA vez y lo reduce a un diccionario
compacto {snomed_code: capitulo ICD-10 de 3 caracteres}, que escribe en
``results/snomed_to_icd10.json``.

Es el mismo patron que fit_lab_stats: un artefacto que se calcula una vez y comparten
los tres build_sequences, en lugar de re-parsear los 100 MB en cada trabajo.

El objeto ``snakemake`` lo inyecta Snakemake en tiempo de ejecucion (no se importa: el
bloque TYPE_CHECKING de abajo solo le da tipo al linter/IDE, nunca se ejecuta).

Reads:  snakemake.input.snomed_map    resources/snomed_icd10_map.csv (mapa crudo).
Writes: snakemake.output[0]           results/snomed_to_icd10.json {snomed: icd10_parent}.
"""

import csv
import json
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo para el linter/IDE (tipo + autocompletado). En tiempo de ejecucion
    # TYPE_CHECKING es False, no se importa nada, y `snakemake` es el objeto inyectado.
    from snakemake.iocontainers import snakemake

# Un codigo ICD-10 es una letra + dos digitos + extension decimal opcional. Synthea a
# veces omite el punto o mete ruido; esto lo limpia a la forma canonica "A12.3".
_ICD10_NOISE = re.compile(r"[^A-Z0-9.]")


def normalize_icd10(code):
    """Normaliza un ICD-10 crudo a "A12.3" (o cadena vacia si no es valido)."""
    code = str(code).strip().upper()
    if not code or code in {"\\N", "NULL", "N/A", "NA"} or "?" in code:
        return ""
    code = _ICD10_NOISE.sub("", code)
    if not code:
        return ""
    if "." in code:
        return code
    if len(code) <= 3:
        return code
    return f"{code[:3]}.{code[3:]}"


def build_snomed_map(map_csv_path):
    """Construye {snomed_code: capitulo ICD-10 de 3 caracteres} desde el mapa crudo.

    En el CSV, la columna 5 es el codigo SNOMED y la 11 el ICD-10 destino. Cuando un
    SNOMED mapea a varios ICD-10, nos quedamos con el primer capitulo (3 caracteres) por
    orden alfabetico: las features de codigo son binarias (presencia de un capitulo de
    diagnostico), asi que perder la subclase es aceptable.
    """
    snomed_to_icds = {}
    with open(map_csv_path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) <= 11:
                continue
            snomed_code = str(row[5]).strip()
            icd10_norm = normalize_icd10(row[11])
            if snomed_code and icd10_norm:
                snomed_to_icds.setdefault(snomed_code, set()).add(icd10_norm)
    return {snomed: sorted(icds)[0][:3] for snomed, icds in snomed_to_icds.items()}


# ── Cuerpo del script ──
# Manda stdout y stderr al log de ESTE trabajo (snakemake.log[0]): cada regla deja su
# propio rastro en logs/, con los prints y cualquier traceback.
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")

snomed_to_icd10 = build_snomed_map(snakemake.input.snomed_map)
with open(snakemake.output[0], "w", encoding="utf-8") as f:
    json.dump(snomed_to_icd10, f)
print(
    f"[icd10_map] {len(snomed_to_icd10)} codigos SNOMED mapeados a capitulo ICD-10",
    flush=True,
)
