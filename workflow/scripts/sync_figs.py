"""Copia las figuras y tablas del pipeline a MemoriaTFM/images/ (tarea `pixi run figs`).

La memoria de Quarto lee sus imagenes de ``MemoriaTFM/images/``, pero esa carpeta no
esta versionada (ver .gitignore): se regenera desde las salidas del pipeline. Este
script hace esa copia de forma idempotente y sin borrar nunca nada; si un origen no
existe, avisa y sigue (las figuras pendientes de los apendices no las genera el
pipeline y no deben romper la tarea).

Origenes -> destino:
    results/eda/figures/*.{png,pdf}   resultados del EDA (etapa 1.5)
    results/eda/tables/*.csv          tablas del EDA
    results/eda/eda_summary.json      resumen del EDA
    results/*.png                     curvas y figuras de evaluacion (etapa 3)
    results/comparison.csv            tabla comparativa de modelos
    figs/dag.png, figs/rulegraph.png  grafos de Snakemake

Uso:
    pixi run figs
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "MemoriaTFM" / "images"

# (directorio origen, patrones, recursivo) -> se copia al mismo nombre en DEST
SOURCES = [
    (ROOT / "results" / "eda" / "figures", ("*.png", "*.pdf"), False),
    (ROOT / "results" / "eda" / "tables", ("*.csv",), False),
    (ROOT / "results", ("*.png", "comparison.csv"), False),
    (ROOT / "figs", ("dag.png", "rulegraph.png"), False),
]

# Ficheros sueltos que no encajan en los patrones anteriores.
EXTRA = [
    ROOT / "results" / "eda" / "eda_summary.json",
]


def _same(src: Path, dst: Path) -> bool:
    """True si dst ya esta al dia respecto a src (mismo tamano y no mas antiguo)."""
    if not dst.is_file():
        return False
    src_stat, dst_stat = src.stat(), dst.stat()
    return (
        dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime >= src_stat.st_mtime
    )


def _copy(src: Path) -> bool:
    """Copia src a DEST. Devuelve True si habia algo que copiar (aunque ya estuviera)."""
    dst = DEST / src.name
    if _same(src, dst):
        print(f"  = {src.relative_to(ROOT)}")
        return True
    shutil.copy2(src, dst)
    print(f"  + {src.relative_to(ROOT)}")
    return True


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    copies = 0
    missing = []
    for directory, patterns, recursive in SOURCES:
        for pattern in patterns:
            if not directory.is_dir():
                missing.append(directory.relative_to(ROOT))
                continue
            glob = directory.rglob if recursive else directory.glob
            for src in sorted(glob(pattern)):
                if src.is_file():
                    _copy(src)
                    copies += 1

    for src in EXTRA:
        if src.is_file():
            _copy(src)
            copies += 1
        else:
            missing.append(src.relative_to(ROOT))

    print(f"\n{copies} ficheros sincronizados en {DEST.relative_to(ROOT)}/")
    if missing:
        print("Origenes que no existen todavia (se omiten):")
        for path in dict.fromkeys(missing):
            print(f"  - {path}")
    if copies == 0:
        print("Nada que copiar: ejecuta antes el pipeline (pixi run pipeline).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
