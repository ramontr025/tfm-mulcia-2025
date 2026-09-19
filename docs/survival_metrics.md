# Guía de Métricas de Supervivencia y Uso de `scikit-survival` (`sksurv`)

Este documento detalla las **5 métricas de evaluación** utilizadas en el pipeline del TFM (implementadas en [`workflow/scripts/eval.py`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/workflow/scripts/eval.py)), las funciones correspondientes de la librería `scikit-survival` (`sksurv`), los requisitos de entrada/salida (formatos, dimensiones y tipos de datos), así como su significado clínico y metodológico.

---

## 1. Fundamentos y Convenios en `scikit-survival`

Antes de analizar cada métrica, es indispensable entender dos convenios esenciales de diseño de `sksurv`:

### 1.1 Array Estructurado de Supervivencia (`Surv.from_arrays`)
A diferencia de librerías como `scikit-learn` (donde la variable dependiente `y` es un vector 1D de clases o regresión), los modelos de supervivencia requieren modelar conjuntamente el **tiempo hasta el evento** y el **indicador de censura**.

`sksurv` utiliza arrays estructurados de `numpy` creados con la función auxiliar:
```python
from sksurv.util import Surv
y = Surv.from_arrays(event=event_array, time=duration_array)
```
- **`event`**: Array 1D de booleanos (`dtype=bool_`), donde `True` indica que el evento ocurrió y `False` que la observación fue censurada a la derecha.
- **`time`**: Array 1D de números en coma flotante (`dtype=float64`), indicando el tiempo de seguimiento hasta el evento o la censura.
- **Forma resultante (`shape`)**: Vector 1D estructurado de longitud `(n,)` con campos con nombre `('event', '?')` y `('time', '<f8')`.

### 1.2 Convenio de Signos de las Predicciones (`estimate`)
- **Métricas de discriminación (`c_index`, `c_index_ipcw`, `auroc`)**: El parámetro `estimate` espera una estimación del **riesgo relativo** (por ejemplo, el predictor lineal $\eta = \beta^T x$ o la salida sin activar del modelo neuronal). **A mayor valor predicho, mayor riesgo de que el evento ocurra antes**. *(Nota: otras librerías como `lifelines` asumen tiempo esperado de supervivencia, es decir, signo inverso)*.
- **Métricas de error/precisión (`brier`)**: El parámetro `estimate` espera **probabilidades de supervivencia** acumuladas $\hat{S}(t|x) = P(T > t \mid x) \in [0, 1]$. **A mayor valor predicho, mayor probabilidad de seguir vivo/libre de evento**.

### 1.3 Prevención de Fuga de Datos (*Data Leakage*) con IPCW
Para técnicas basadas en ponderación por el inverso de la probabilidad de censura (**IPCW**, *Inverse Probability of Censoring Weighting*), `sksurv` requiere pasar `survival_train` y `survival_test`:
- La distribución de censura $G(t) = P(C > t)$ se estima **exclusivamente con los datos de entrenamiento (`train`)** mediante el estimador de Kaplan-Meier sobre los tiempos de censura.
- Esos pesos calculados en `train` se aplican para reponderar las observaciones del conjunto de evaluación (`val` o `holdout`). De este modo se evita cualquier filtración de información entre splits.

---

## 2. Detalle de las 5 Métricas Evaluadas

Las 5 métricas registradas en `SCALAR_METRIC_KEYS` dentro de `eval.py` son:
1. `c_index` (Harrell's Concordance Index)
2. `auroc` (Cumulative/Dynamic Time-dependent ROC AUC)
3. `c_index_ipcw` (Uno's IPCW Concordance Index)
4. `brier` (IPCW Brier Score en el horizonte)
5. `ipa` (Index of Prediction Accuracy frente al modelo nulo)

---

### Métrica 1: `c_index` — Índice de Concordancia de Harrell

#### ¿Qué mide y para qué sirve?
Mide la **capacidad discriminativa global** del modelo. Evalúa si el modelo es capaz de ordenar adecuadamente a los pacientes según su riesgo relativo a lo largo de todo el periodo de seguimiento.
Específicamente, calcula la proporción de pares de pacientes evaluables en los que el paciente con mayor riesgo predicho sufre el evento antes que el paciente con menor riesgo predicho.

- **Rango**: $[0.0, 1.0]$.
  - $0.5$: Predicción no mejor que el azar.
  - $1.0$: Discriminación perfecta.
  - $< 0.5$: Predicción invertida.
- **Limitación**: El estimador clásico de Harrell se vuelve optimista o sesgado cuando la censura a la derecha depende fuertemente del tiempo o cuando el seguimiento es incompleto, ya que los pares censurados antes del tiempo de evento del otro individuo se descartan sin reponderación.

#### Función de `sksurv`
```python
from sksurv.metrics import concordance_index_censored
```

#### Argumentos de Entrada
| Parámetro | Tipo / Forma | Descripción |
| :--- | :--- | :--- |
| `event_indicator` | `np.ndarray` (1D `bool_`, forma `(n,)`) | Indicador booleano de si ocurrió el evento (`True`) o fue censurado (`False`). |
| `event_time` | `np.ndarray` (1D `float64`, forma `(n,)`) | Tiempos observados de seguimiento. |
| `estimate` | `np.ndarray` (1D `float64`, forma `(n,)`) | Puntuación de riesgo predicha por el modelo. Mayor valor = mayor riesgo. |
| `tied_tol` *(opcional)* | `float` (defecto: `1e-8`) | Tolerancia numérica para considerar dos predicciones de riesgo como empatadas. |

#### Argumentos de Salida
Devuelve una tupla de 5 elementos: `(cindex, concordant, discordant, tied_risk, tied_time)`
1. **`cindex`** (`float`): Valor escalar del C-index de Harrell.
2. **`concordant`** (`int`): Número de pares ordenados concordantemente.
3. **`discordant`** (`int`): Número de pares discordantes.
4. **`tied_risk`** (`int`): Número de pares donde los tiempos difieren pero las predicciones de riesgo empatan (suman $0.5$ al numerador).
5. **`tied_time`** (`int`): Número de pares con tiempos de evento idénticos.

#### Obtención en `eval.py`
```python
ci, conc, disc, tied_risk, _tied_time = concordance_index_censored(
    evs_sub.astype(bool), durs_sub, risks_sub
)
out["c_index"] = float(ci)
out["concordant"] = int(conc)
out["discordant"] = int(disc)
out["tied_risk"] = int(tied_risk)
out["n_comparable"] = int(conc + disc + tied_risk)
```
*(Los conteos individuales se guardan para posibilitar el cálculo exacto del C-index agrupado/pooled entre múltiples landmarks)*.

---

### Métrica 2: `auroc` — AUROC Acumulativo / Dinámico en el Horizonte Temporal

#### ¿Qué mide y para qué sirve?
Mide la **capacidad discriminativa en una ventana temporal concreta** (por ejemplo, a los 3 años del landmark, $t = \text{horizon}$).
A diferencia del C-index que resume todo el historial temporal, el AUC acumulativo/dinámico (Hung & Chiang, 2010; Blanche et al., 2013) define de forma binaria a los pacientes respecto al horizonte $t$:
- **Casos acumulativos**: Pacientes que sufrieron el evento antes o en el instante $t$ ($T_i \le t, \delta_i = 1$).
- **Controles dinámicos**: Pacientes que continúan vivos y en seguimiento estricto después de $t$ ($T_i > t$).
- Utiliza **IPCW** para corregir a aquellos pacientes censurados antes de $t$ cuya condición final no se conoce directamente.

Responde a la pregunta clínica: *"¿Con qué precisión distingue el modelo a un paciente que enfermará/morirá dentro de los próximos 3 años de aquel que permanecerá sano más allá de ese periodo?"*.

- **Rango**: $[0.0, 1.0]$.
  - $0.5$: Azar.
  - $1.0$: Separación perfecta entre casos antes de $t$ y controles más allá de $t$.

#### Función de `sksurv`
```python
from sksurv.metrics import cumulative_dynamic_auc
```

#### Argumentos de Entrada
| Parámetro | Tipo / Forma | Descripción |
| :--- | :--- | :--- |
| `survival_train` | `np.ndarray` estructurado `(n_train,)` | Array con campos `('event', '?')` y `('time', '<f8')` de train, utilizado para ajustar el Kaplan-Meier de la censura $G(t)$. |
| `survival_test` | `np.ndarray` estructurado `(n_test,)` | Array estructurado con los datos de evaluación. |
| `estimate` | `np.ndarray` (1D `(n_test,)` o 2D `(n_test, n_times)`) | Score de riesgo predicho por el modelo. Mayor valor = mayor riesgo. |
| `times` | array-like de floats (longitud `k`) | Instantes temporales donde se evalúa el AUC. En `eval.py` se pasa una lista de un elemento `[horizon]`. |
| `tied_tol` *(opcional)* | `float` (defecto: `1e-8`) | Tolerancia para empates. |

#### Argumentos de Salida
Devuelve una tupla de 2 elementos: `(auc, mean_auc)`
1. **`auc`** (`np.ndarray`, forma `(len(times),)`): Array 1D con el AUC calculado en cada instante de `times`.
2. **`mean_auc`** (`float`): Media ponderada/integrada del AUC a lo largo de todo el vector `times` calculada mediante la regla del trapecio.

#### Obtención en `eval.py`
```python
# times=[horizon] genera un array auc_t de longitud 1
auc_t = cumulative_dynamic_auc(y_train, y_test, risks_sub, times=[horizon])[0]
out["auroc"] = float(auc_t[0])
```

---

### Métrica 3: `c_index_ipcw` — Índice de Concordancia de Uno con Ponderación IPCW

#### ¿Qué mide y para qué sirve?
Es una formulación del índice de concordancia propuesta por **Uno et al. (2011)** que resuelve el sesgo del C-index de Harrell frente a censura dependiente del tiempo o truncamiento.
Pondera cada par de observaciones por el inverso del producto de las probabilidades estimadas de censura $\hat{G}(T_i) \cdot \hat{G}(T_j)$.
- Es un estimador **consistente y asintóticamente no sesgado** de la probabilidad de concordancia de la población subyacente.
- Es más conservador y robusto que el C-index de Harrell cuando la tasa de censura es elevada hacia el final del seguimiento.

- **Rango**: $[0.0, 1.0]$ (interpretación análoga a Harrell: $0.5$ azar, $1.0$ discriminación perfecta).

#### Función de `sksurv`
```python
from sksurv.metrics import concordance_index_ipcw
```

#### Argumentos de Entrada
| Parámetro | Tipo / Forma | Descripción |
| :--- | :--- | :--- |
| `survival_train` | `np.ndarray` estructurado `(n_train,)` | Datos de entrenamiento con `event` y `time` para modelar la censura. |
| `survival_test` | `np.ndarray` estructurado `(n_test,)` | Datos de evaluación con `event` y `time`. |
| `estimate` | `np.ndarray` (1D `(n_test,)`) | Puntuación de riesgo del modelo (a mayor valor, mayor riesgo predicho). |
| `tau` *(opcional)* | `float` | Tiempo de corte máximo para la evaluación (si es `None`, se utiliza el tiempo máximo de evento común). |
| `tied_tol` *(opcional)* | `float` (defecto: `1e-8`) | Tolerancia para considerar predicciones empatadas. |

#### Argumentos de Salida
Devuelve una tupla de 5 elementos: `(cindex, concordant, discordant, tied_risk, tied_time)`
1. **`cindex`** (`float`): Valor del C-index IPCW corregido.
2. **`concordant`**, **`discordant`**, **`tied_risk`**, **`tied_time`**: Conteos brutos de pares *(Importante: estos conteos en `sksurv` no están reponderados por IPCW; sólo el escalar `cindex` lo está)*.

#### Obtención en `eval.py`
```python
out["c_index_ipcw"] = float(
    concordance_index_ipcw(y_train, y_test, risks_sub)[0]
)
```

---

### Métrica 4: `brier` — Brier Score IPCW en el Horizonte Clínico

#### ¿Qué mide y para qué sirve?
Evalúa la **precisión predictiva general** combinando simultáneamente **discriminación y calibración**.
Propuesto para supervivencia por Graf et al. (1999), mide el error cuadrático medio entre la probabilidad de supervivencia predicha para un sujeto en el tiempo $t$ y su estado real observado (si sigue libre del evento o ya lo sufrió), reponderando cada error individual por la probabilidad de censura (IPCW):
$$\text{BS}(t) = \frac{1}{n} \sum_{i=1}^n w_i(t) \left( \mathbb{I}(T_i > t) - \hat{S}(t \mid x_i) \right)^2$$

- **Rango**: $[0.0, 1.0]$.
  - **Cuanto menor sea el valor, mejor es el modelo** (función de pérdida / error).
  - $0.0$: Predicciones probabilísticas perfectas.
  - Típicamente un valor $\le 0.25$ es el umbral de referencia (un modelo desinformado que prediga $0.5$ para todos obtiene un Brier de $0.25$ si la prevalencia es uniforme).

#### Función de `sksurv`
```python
from sksurv.metrics import brier_score
```

#### Argumentos de Entrada
| Parámetro | Tipo / Forma | Descripción |
| :--- | :--- | :--- |
| `survival_train` | `np.ndarray` estructurado `(n_train,)` | Datos de train para ajustar la censura Kaplan-Meier. |
| `survival_test` | `np.ndarray` estructurado `(n_test,)` | Datos de test a evaluar. |
| `estimate` | `np.ndarray` (2D `(n_test, len(times))`) | **Probabilidades de supervivencia predichas** $\hat{S}(t \mid x)$. Valores en $[0, 1]$. |
| `times` | array-like de floats (longitud `k`) | Vector con los tiempos de evaluación (en `eval.py` se usa `[horizon]`). |

> [!WARNING]
> **Diferencia crítica de dimensiones y concepto**:
> A diferencia de las funciones de C-index y AUROC que reciben riesgos relativos 1D `(n,)`, `brier_score` exige:
> 1. Que las entradas sean **probabilidades de supervivencia** (no riesgos).
> 2. Que el array sea **bidimensional** de forma `(n_test, len(times))`. Si se pasa un vector 1D `(n,)`, `sksurv` lanzará un error de dimensiones.

#### Obtención en `eval.py`
En `eval.py`, los modelos (redes neuronales recurrentes basadas en Cox/hazard o modelos de Cox clásicos) predicen un riesgo relativo `risks_sub`. La probabilidad de supervivencia $\hat{S}(t \mid x)$ se deriva a partir del riesgo acumulado basal de Breslow $H_0(t)$ estimado fijado en train (`compute_baseline_hazard`):
$$\hat{S}(t \mid x) = \exp\left(-H_0(t) \cdot \exp(\text{risk} - \text{max\_risk})\right)$$

```python
# surv_prob_model tiene dimensión (n,)
surv_prob_model = np.exp(-h0 * np.exp(risks_sub - maxr))

# Se expande a 2D con [:, None] para tener shape (n, 1) y tiempos=[horizon]
brier_model = brier_score(
    y_train, y_test, surv_prob_model[:, None], [horizon]
)[1][0]

out["brier"] = float(brier_model)
```

#### Argumentos de Salida
Devuelve una tupla de 2 elementos: `(times, brier_scores)`
1. **`times`** (`np.ndarray`, forma `(len(times),)`): Tiempos evaluados.
2. **`brier_scores`** (`np.ndarray`, forma `(len(times),)`): Brier score obtenido en cada tiempo.
*(En el código se indexa `[1][0]` para extraer el escalar del Brier en `times[0]`)*.

---

### Métrica 5: `ipa` — Index of Prediction Accuracy (Índice de Precisión Predictiva)

#### ¿Qué mide y para qué sirve?
Propuesto por Kattan y Gerds (2018), el **IPA** (también conocido como $R^2$ de Brier o *Scaled Brier Score*) evalúa la **ganancia de información real** que aporta el modelo respecto a un modelo nulo o de referencia que no utiliza ninguna variable predictora (es decir, una predicción basada únicamente en la incidencia acumulada basal observada en la población).

Se define formalmente como:
$$\text{IPA} = 1 - \frac{\text{Brier}_{\text{modelo}}}{\text{Brier}_{\text{nulo}}}$$

- **Rango**: $(-\infty, 1.0]$.
  - $\text{IPA} = 1.0$ ($100\%$): Modelo perfecto (Brier del modelo = 0).
  - $\text{IPA} = 0.0$ ($0\%$): El modelo no aporta absolutamente nada sobre la simple incidencia media histórica.
  - $\text{IPA} > 0.0$: Porcentaje de reducción del error de predicción atribuible a las variables del paciente (ej. $\text{IPA}=0.15 \rightarrow 15\%$ de mejora sobre el modelo nulo).
  - $\text{IPA} < 0.0$: El modelo está **peor calibrado** que predecir simplemente la probabilidad media poblacional (alerta clara de descalibración o sobreajuste).

#### Cómo se calcula en `eval.py`
No existe una función única llamada `ipa` en `sksurv`, sino que se calcula a partir de dos llamadas a `brier_score`:
1. Cálculo de $\text{Brier}_{\text{modelo}}$ (descrito en la métrica 4).
2. Estimación de la probabilidad de supervivencia del modelo nulo fijada en train:
   $$S_{\text{null}}(t) = \exp(-H_{0,\text{null}}(t))$$
3. Evaluación de $\text{Brier}_{\text{nulo}}$:
   ```python
   surv_prob_null = np.full(n, np.exp(-h0_null))
   brier_null = brier_score(
       y_train, y_test, surv_prob_null[:, None], [horizon]
   )[1][0]
   ```
4. Cálculo relativo del IPA:
   ```python
   out["ipa"] = float(1.0 - brier_model / brier_null) if brier_null > 0 else np.nan
   ```

---

## 3. Resumen Comparativo de las Métricas

| Métrica | Qué evalúa | Dependencia temporal | Corrige censura (IPCW) | Función `sksurv` | Forma del `estimate` | Rango / Sentido |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`c_index`** | Discriminación global | Global (todo el seguimiento) | No (estimador clásico de Harrell) | `concordance_index_censored` | 1D `(n,)` (Riesgo) | $[0.5, 1.0]$ (Mayor es mejor) |
| **`auroc`** | Discriminación local en ventana | Puntual ($t = \text{horizon}$) | Sí (Blanche / Hung & Chiang) | `cumulative_dynamic_auc` | 1D `(n,)` (Riesgo) | $[0.5, 1.0]$ (Mayor es mejor) |
| **`c_index_ipcw`** | Discriminación global insesgada | Global (hasta $\tau$) | Sí (Uno et al., 2011) | `concordance_index_ipcw` | 1D `(n,)` (Riesgo) | $[0.5, 1.0]$ (Mayor es mejor) |
| **`brier`** | Calibración + Discriminación | Puntual ($t = \text{horizon}$) | Sí (Graf et al., 1999) | `brier_score` | 2D `(n, 1)` (Probabilidad Superv.) | $[0.0, 0.25]$ (Menor es mejor) |
| **`ipa`** | Rendimiento relativo ($R^2$) | Puntual ($t = \text{horizon}$) | Sí (derivado de Brier) | Derivada de `brier_score` | 2D `(n, 1)` (Probabilidad Superv.) | $[0.0, 1.0]$ (Mayor es mejor) |

---

## 4. Agrupación por Estratos Landmark (*Pooling*)

En la arquitectura del pipeline, las métricas se evalúan individualmente para cada corte temporal (*landmark*) y posteriormente se agregan en un estimador unificado por split (`_pool_landmark_metrics` en [`eval.py`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/workflow/scripts/eval.py#L131)):

- **Harrell's `c_index` (Exacto)**: Se suman los pares concordantes, discordantes y empatados a través de todos los landmarks y se evalúa la fracción global:
  $$\text{C-index}_{\text{pooled}} = \frac{\sum \text{concordant} + 0.5 \sum \text{tied}}{\sum \text{concordant} + \sum \text{discordant} + \sum \text{tied}}$$
  *(Esto asegura que nunca se comparen pares de pacientes entre landmarks distintos, preservando la validez del análisis dinámico)*.
- **`brier` e `ipa` (Exacto por media ponderada)**: Al ser el Brier la media de los errores individuales, la combinación es exactamente la media ponderada por el tamaño muestral $n_{\text{lm}}$ de cada corte.
- **`auroc` y `c_index_ipcw` (Media ponderada aproximada)**: Dado que `sksurv` no exporta conteos reponderados por IPCW por separado, se aproximan mediante la media ponderada por el número de pacientes y pares comparables respectivamente.

---

> [!TIP]
> **Inferencia e Intervalos de Confianza (IC 95%)**:
> Para conocer el procedimiento de cálculo de intervalos de confianza mediante remuestreo por paciente (Cluster Bootstrap) tanto por landmark como sobre el estimando agrupado, consulta la guía detallada en [`docs/bootstrap.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/bootstrap.md).

