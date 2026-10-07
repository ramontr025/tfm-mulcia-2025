# Documentación del script `workflow/scripts/build_icd10_map.py`

Este documento explica, bloque a bloque, cómo funciona el script `build_icd10_map.py`, qué datos recibe y cómo transforma el mapa SNOMED->ICD-10 crudo en un diccionario compacto listo para reutilizar en el resto del flujo.

## Propósito general

El script implementa un paso de preprocesado auxiliar de la etapa 1. Su objetivo es leer una tabla grande de correspondencias entre SNOMED CT e ICD-10 y reducirla a un artefacto pequeño, reutilizable y más rápido de cargar.

La idea central es esta:

- los diagnósticos originales vienen en SNOMED CT
- el modelo trabaja con capítulos ICD-10, que son mucho más compactos
- en lugar de releer y reprocesar el CSV grande en cada ejecución, el script lo convierte una sola vez a un JSON con el mapeo mínimo necesario

El resultado es un diccionario del tipo:

```python
{snomed_code: icd10_chapter}
```

donde `icd10_chapter` es el prefijo de 3 caracteres del código ICD-10.

## Qué recibe el script

El script se ejecuta como `snakemake script:`. Por eso el objeto `snakemake` no se importa manualmente: Snakemake lo inyecta en tiempo de ejecución.

### Entrada esperada

El script consume un único fichero CSV crudo:

- `map_csv_path`: `resources/snomed_icd10_map.csv`

Ese CSV contiene el mapa grande SNOMED->ICD-10. Según el propio código, el SNOMED está en la columna 5 y el ICD-10 destino en la columna 11.

### Salida esperada

El script escribe un único JSON en `snakemake.output[0]`:

- `results/snomed_to_icd10.json`

Ese JSON contiene un mapa compacto de SNOMED a capítulo ICD-10 de 3 caracteres.

## Importaciones y preparación inicial

```python
import csv
import json
import re
import sys
from typing import TYPE_CHECKING
```

- `csv` permite leer el fichero de correspondencias sin cargar dependencias extra.
- `json` escribe el diccionario final.
- `re` limpia ruido en los códigos ICD-10.
- `sys` redirige la salida al log de Snakemake.
- `TYPE_CHECKING` se usa solo para tipado en el editor.

## Bloque `TYPE_CHECKING`

```python
if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake
```

Este bloque no se ejecuta en runtime. Solo existe para que el editor entienda el tipo del objeto `snakemake`.

## Expresión regular `_ICD10_NOISE`

```python
_ICD10_NOISE = re.compile(r"[^A-Z0-9.]")
```

Esta regex elimina cualquier carácter que no sea:

- letra mayúscula
- dígito
- punto decimal

Se usa para limpiar códigos ICD-10 que llegan con ruido, espacios, símbolos o formatos inconsistentes.

## Función `normalize_icd10`

```python
def normalize_icd10(code):
```

Esta función normaliza un código ICD-10 crudo a una forma consistente. Devuelve una cadena vacía si el código no es usable.

### Conversión inicial

```python
code = str(code).strip().upper()
```

Se fuerza a texto, se eliminan espacios extremos y se pasa a mayúsculas.

### Filtros de valores inválidos

```python
if not code or code in {"\\N", "NULL", "N/A", "NA"} or "?" in code:
    return ""
```

Se descartan valores vacíos o marcadores típicos de ausencia o ruido.

### Limpieza de caracteres extraños

```python
code = _ICD10_NOISE.sub("", code)
if not code:
    return ""
```

Se eliminan caracteres no permitidos. Si después de limpiar no queda nada, se devuelve vacío.

### Canonización del formato

```python
if "." in code:
    return code
if len(code) <= 3:
    return code
return f"{code[:3]}.{code[3:]}"
```

La lógica final intenta reconstruir el formato ICD-10 estándar:

- si ya tiene punto, se conserva
- si tiene 3 caracteres o menos, se conserva tal cual
- si es más largo y no tiene punto, se inserta después de los 3 primeros caracteres

Esto convierte formas como `A123` en `A12.3`.

## Función `build_snomed_map`

```python
def build_snomed_map(map_csv_path):
```

Esta función construye el diccionario final `snomed_code -> capítulo ICD-10 de 3 caracteres` a partir del CSV crudo.

### Estructura acumuladora

```python
snomed_to_icds = {}
```

Se usa un diccionario de SNOMED a conjunto de ICD-10 normalizados. Eso permite acumular múltiples destinos por SNOMED sin duplicados.

### Lectura fila a fila del CSV

```python
with open(map_csv_path, encoding="utf-8", newline="") as f:
    for row in csv.reader(f):
```

El archivo se lee de forma secuencial, sin cargarlo entero en memoria. Eso es útil porque el mapa crudo es grande.

### Filtro de filas demasiado cortas

```python
if len(row) <= 11:
    continue
```

La lógica espera que existan al menos 12 columnas. Si no, la fila no contiene la información necesaria y se ignora.

### Extracción de columnas relevantes

```python
snomed_code = str(row[5]).strip()
icd10_norm = normalize_icd10(row[11])
```

Se toma:

- `row[5]` como código SNOMED
- `row[11]` como código ICD-10 destino

El ICD-10 se normaliza antes de usarlo.

### Acumulación del mapa

```python
if snomed_code and icd10_norm:
    snomed_to_icds.setdefault(snomed_code, set()).add(icd10_norm)
```

Si ambos valores son válidos:

- se crea el conjunto para ese SNOMED si todavía no existe
- se añade el ICD-10 normalizado al conjunto

Así se preservan todas las correspondencias vistas para cada SNOMED.

### Reducción al capítulo ICD-10

```python
return {snomed: sorted(icds)[0][:3] for snomed, icds in snomed_to_icds.items()}
```

Esta es la parte clave del preprocesado.

Para cada SNOMED:

1. se ordenan alfabéticamente los ICD-10 asociados
2. se toma el primero
3. se recorta a los 3 primeros caracteres

El resultado ya no es un ICD-10 completo, sino su capítulo/prefijo de 3 caracteres.

La decisión simplifica el espacio de etiquetas y hace que las features de código sean más densas y manejables.

## Cuerpo ejecutable del script

```python
sys.stdout = sys.stderr = open(snakemake.log[0], "w", encoding="utf-8")
```

La salida estándar y de error se redirige al log de la regla. Así cada ejecución deja trazabilidad propia en `logs/`.

```python
snomed_to_icd10 = build_snomed_map(snakemake.input.snomed_map)
```

Se construye el diccionario reducido a partir del CSV de entrada.

```python
with open(snakemake.output[0], "w", encoding="utf-8") as f:
    json.dump(snomed_to_icd10, f)
```

El resultado se escribe como JSON en la ruta de salida configurada por Snakemake.

```python
print(
    f"[icd10_map] {len(snomed_to_icd10)} codigos SNOMED mapeados a capitulo ICD-10",
    flush=True,
)
```

Se emite un resumen con el número de códigos SNOMED mapeados.

## Qué hace exactamente el preproceso

En términos prácticos, este script:

1. Lee un CSV grande de correspondencias SNOMED->ICD-10.
2. Limpia y normaliza los códigos ICD-10.
3. Agrupa todas las variantes por código SNOMED.
4. Reduce cada correspondencia al capítulo ICD-10 de 3 caracteres.
5. Guarda el resultado en un JSON compacto reutilizable.

El objetivo no es conservar toda la granularidad clínica del ICD-10, sino obtener una representación más pequeña y estable para construir secuencias o variables categóricas downstream.