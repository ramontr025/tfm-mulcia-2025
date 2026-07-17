# Descripción de las secuencias (formato pickle)

## Archivos

- `data_tmp/sequences_train.pkl` — **2500** pacientes
- `data_tmp/sequences_val.pkl` — **1508** pacientes

Cada archivo es un `pickle` con un diccionario Python de 3 claves:

```
{
  "sequences":  dict[str, dict],   # un entry por paciente
  "lab_codes":  list[str],         # 23 códigos LOINC
  "cohort":     pd.DataFrame,      # cohorte con etiquetas de supervivencia
}
```

---

## 1. `sequences` — datos por paciente

Tipo: `dict[str, dict]`. Cada clave es un UUID (ID de paciente, ej. `"d5cbf89d-8021-b42b-2fe6-799e6734b154"`). Algunas entradas pueden ser `None` (se filtran en `load_sequences()`).

Cada secuencia contiene:

| Campo | Tipo real | Dimensión conceptual | Descripción |
|---|---|---|---|
| `times` | `list[float]` | `(T,)` | Edad del paciente (en años) en cada observación. **Lista nativa de Python, no numpy array.** |
| `labs` | `list[np.ndarray]` | `(T, 23)` | Valores de laboratorio **z-score normalizados**. 23 features (una por LOINC). 0 cuando no medido. Cada elemento es un `np.ndarray` de `float32` con forma `(23,)`. |
| `masks` | `list[np.ndarray]` | `(T, 23)` | Máscara binaria: 1 si el lab fue medido en ese timestep, 0 si no. Misma estructura que `labs`. |
| `codes` | `list[list[str]]` | `(T,)` | Códigos ICD-10 registrados en cada visita (lista de strings por timestep). La lista exterior tiene la misma longitud que T (número de visitas). No se usa en la versión básica del modelo. |
| `sex_male` | `int` | escalar | 1 = hombre, 0 = mujer. Tipo entero Python (no float). |

### Notas sobre `times`

- Las edades son **irregulares**: los intervalos entre observaciones varían mucho (p. ej., 31 → 62 → 70 años, con gaps de +31 y +8 años).
- Es un **array unidimensional** pero almacenado como lista Python.
- Sirve para calcular `delta_t` (gap inter-observación), que el modelo GRU-D usa para decidir cuánto decaer.

### Notas sobre `labs`

- Los valores están **z-score normalizados** (media poblacional = 0 por construcción tras el normalizado).
- Cuando un lab **no se midió** en un timestep, su valor es **0** (no NaN). La máscara (`masks`) permite distinguir "medido y vale 0" de "no medido".
- Hay **23 features** de laboratorio, cuyos códigos LOINC están en `lab_codes`.

### Notas sobre `masks`

- Misma forma que `labs`: `(T, 23)`.
- Valores: solo 0.0 y 1.0 (almacenado como `float32`).
- Es la señal que permite al GRU-D saber qué imputar vs. qué usar directamente.

### Notas sobre `codes`

- Es una **lista de listas** de strings. La lista exterior tiene exactamente la **misma longitud que `T`** (cada elemento corresponde a una visita médica o observación del paciente a lo largo del tiempo).
- Cada elemento de la lista exterior (cada visita en el tiempo $t$) es una lista de strings que contiene los diagnósticos específicos (códigos ICD-10) registrados en esa consulta (ej. `[]` si no hay diagnósticos, `['E66']`, o `['D49', 'D07']` para múltiples).
- Las enfermedades de cada visita no pertenecen a un capítulo de la CIE-10 fijo por lista, sino que son simplemente todas las enfermedades presentes y diagnosticadas al paciente en esa visita específica (pudiendo ser de cualquier capítulo de la CIE-10).
- No hay un vocabulario preconstruido — el notebook lo crea opcionalmente en la extensión con `nn.EmbeddingBag`.
- En el forward pass básico estos códigos **no se usan**.

### Notas sobre `sex_male`

- Entero simple (`int`), no array.
- Se expande a `(B, 1)` dentro del modelo para concatenarlo como entrada estática al GRU cell.

---

## 2. `lab_codes` — etiquetas de laboratorio

Tipo: `list[str]` con **23** códigos LOINC:

```
['8480-6', '8462-4', '9279-1', '8867-4', '29463-7', '8302-2',
 '39156-5', '70274-6', '718-7', '777-3', '785-6', '786-4',
 '787-2', '32623-1', '32207-3', '4544-3', '6690-2', '789-8',
 '21000-5', '2085-9', '2571-8', '2093-3', '18262-6']
```

Cada código LOINC se corresponde con una columna de `labs` y `masks` (en orden posicional).

---

## 3. `cohort` — DataFrame de cohorte

Tipo: `pandas.DataFrame` con **4 columnas** y tantas filas como pacientes.

| Columna | Tipo | Descripción |
|---|---|---|
| `Id` | `str` | UUID del paciente (coincide con clave en `sequences`) |
| `event` | `int` (0/1) | 1 = cáncer de pulmón dentro del horizonte de predicción, 0 = censurado |
| `duration` | `float` | Años desde la edad de índice (18 años) hasta el evento o censura (ver detalles detallados abajo) |
| `status` | `int` | Indicador de estado (parece redundante con `event`, pero se mantiene) |

### Estadísticas

| | Train | Val |
|---|---|---|
| Pacientes | 2500 | 1508 |
| Tasa de evento | 3.56% | 3.58% |
| Hombres | 1323 (52.9%) | — |
| Mujeres | 1177 (47.1%) | — |

---

## Detalles adicionales (más allá de lo obvio)

1. **Tipo mixto de almacenamiento**: `times` es lista Python, `labs`/`masks` son listas de numpy arrays, y `sex_male` es int. Esto es relevante si se quiere inspeccionar o modificar los datos: hay que tratar cada campo con su tipo concreto.

2. **`labs` = 0 donde no medido**: Los valores faltantes se representan con 0 (no NaN). Combinado con `masks`, el modelo sabe si ese 0 es real o es placeholder. Es común en GRU-D, pero puede causar confusión si se examinan los datos sin la máscara.

3. **Duración y Censura respecto a la Edad de Índice (18 años):**
   * **Edad de Índice (`INDEX_AGE = 18.0`):** Define la entrada del paciente al estudio de seguimiento. Dado que es un modelo enfocado en adultos (riesgo de cáncer de pulmón), la cohorte está filtrada para incluir a pacientes adultos. Si hubiera observaciones previas a los 18 años, estas ocurren antes del origen del estudio. La duración se calcula estrictamente a partir de los 18 años.
   * **Definición de Censura (`event = 0`):** Significa que el evento de interés (diagnóstico de cáncer de pulmón) no fue observado durante el seguimiento del paciente. Ocurre por tres razones principales:
     * *Fin de la recopilación de datos:* El paciente sigue libre de cáncer y vivo en la fecha de la última visita clínica disponible en el sistema.
     * *Pérdida de seguimiento:* El paciente deja de asistir a consultas en el sistema de salud (ej. se mudó de región), por lo que su última consulta marca su tiempo de censura.
     * *Causas competitivas:* El paciente fallece por una causa ajena al estudio (ej. accidente, infarto) antes de desarrollar cáncer de pulmón, marcando el momento del fallecimiento como el punto en el que se censura su seguimiento respecto al cáncer.
   * **Cálculo de `duration` global:** Representa los años transcurridos desde la entrada a la cohorte hasta el evento/censura:
     $$\text{duration} = \text{Edad al Evento o Censura} - 18.0$$
   * **Cálculo de `time_from_lm` (tiempo de supervivencia para el modelo):** Al evaluar en un landmark (`lm`), se calcula el tiempo restante hasta el evento o censura:
     $$\text{time\_from\_lm} = \text{duration} - (\text{lm} - 18.0) = \text{Edad al Evento o Censura} - \text{lm}$$
     Si este valor es $\le 0$, el paciente ya experimentó el evento o fue censurado antes del landmark y se excluye de ese análisis de supervivencia.

4. **Longitud de secuencias muy variable**: Van de **2 a 335** observaciones por paciente (media ~25, mediana ~18). Para ahorrar memoria se truncan a 200 (`MAX_SEQ_LEN`), cogiendo las más recientes.

5. **Z-score normalization**: Los labs están normalizados con media 0 y desviación 1 (poblacional). El `mean_x` se pasa al modelo como buffer para la imputación por decay.

6. **`load_sequences()` filtra `None`**: La función de carga omite entradas del dict `sequences` cuyo valor sea `None`.

7. **Sobremuestreo de eventos**: En el DataLoader de train se usa `WeightedRandomSampler` con multiplicador `EVENT_OVERSAMPLE=2.0` para que los pacientes con evento tengan el doble de probabilidad de ser muestreados, compensando el desbalance (solo ~3.6% de eventos).

8. **Cohorte homogénea train/val**: La tasa de evento es prácticamente idéntica entre train y val (3.56% vs 3.58%), lo que sugiere un split estratificado o generado con la misma distribución sintética (Synthea).
