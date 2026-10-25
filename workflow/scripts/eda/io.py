"""Contrato del artefacto del EDA: esquema, procedencia y escritura.

Este modulo fija lo que el resto del paquete puede escribir y como. Cualquier cifra
que vaya a la memoria tiene que salir de aqui: JSON con ``schema_version`` y
``provenance``, o CSV en formato largo. Nada de numeros sueltos en el texto.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "2.0"

#: Bloques del protocolo. ``C`` es la cohorte (§4.1); ``1``-``5`` son las secuencias.
BLOCK_KEYS = ("C", "1", "2", "3", "4", "5")

#: Nombres de figura por bloque, en el orden en que se dibujan.
FIGURE_NAMES = {
    "C": "fig_cohort",
    "1": "fig_sample_coverage",
    "2": "fig_time_structure",
    "3": "fig_observation_density",
    "4": "fig_lab_redundancy",
    "5": "fig_value_scale",
}


def _clean(value: Any) -> Any:
    """Convierte ``NaN``/``Inf`` en ``None`` para que el JSON sea valido."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def git_commit() -> str | None:
    """Hash corto del commit, o ``None`` si git no esta disponible."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def sha256_file(path: Path) -> str:
    """SHA-256 de un fichero, en bloques de 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, exclude: Iterable[Path] = ()) -> dict[str, str]:
    """Manifiesto ``{ruta relativa: sha256}`` de un arbol, con exclusiones."""
    excluded = {Path(p).resolve() for p in exclude}
    manifest: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in excluded or any(
            parent in path.resolve().parents for parent in excluded
        ):
            continue
        manifest[path.relative_to(root).as_posix()] = sha256_file(path)
    return manifest


def build_provenance(
    params: dict,
    inputs: dict,
    runtime_seconds: float | None = None,
    tool_versions: dict | None = None,
) -> dict:
    """Bloque ``provenance``: que se ejecuto, con que y sobre que."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "params": _clean(params),
        "seed": params.get("seed"),
        "days_per_year": params.get("days_per_year", 365.25),
        "inputs": _clean(inputs),
        "runtime_seconds": runtime_seconds,
        "tool_versions": tool_versions or {},
    }


def write_json(obj: dict, path: Path) -> None:
    """Escribe el JSON con ``NaN``/``Inf`` como ``null`` y UTF-8 sin escapes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_clean(obj), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(rows: list[dict], path: Path, columns: list[str] | None = None) -> None:
    """Escribe un CSV en formato largo; ``NaN``/``Inf`` quedan como celda vacia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(row.get(key), float)
                        and (math.isnan(row[key]) or math.isinf(row[key]))
                        else row.get(key, "")
                    )
                    for key in columns
                }
            )


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    """Lee ``loinc_labels.csv`` y lo indexa por codigo LOINC."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["lab_code"]: row for row in rows}


def pc(value: float) -> float:
    """Fraccion (0-1) a porcentaje, sin redondear."""
    return float(value) * 100.0
