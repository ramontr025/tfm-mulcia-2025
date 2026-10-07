# Evaluación de Supervivencia Dinámica: Landmarking y Calibración en el Modelo GRU-D

Este documento detalla el funcionamiento teórico y práctico de las dos últimas celdas de código del cuaderno [grud_student.ipynb](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025/2DA%20FASE/us-mulcia-2026-ramon/grud_student.ipynb). Ambas celdas implementan el flujo de evaluación y calibración de un modelo de supervivencia de tiempo al evento basado en GRU-D y la verosimilitud parcial de Cox.

---

## 1. Evaluación General: C-index Across All Landmarks

La primera de estas dos celdas calcula el **Índice de Concordancia (C-index)** del modelo a lo largo de una secuencia de edades de landmark (hitos temporales) que van de los **45 a los 70 años**.

### ¿Qué es el Análisis Landmark (Hito Temporal)?
En estudios longitudinales clínicos, los datos de los pacientes (como las mediciones de laboratorio o diagnósticos) cambian continuamente a lo largo del tiempo. Para evaluar el riesgo de manera dinámica sin incurrir en sesgos de tiempo inmortal, se utiliza el **Análisis Landmark**:
1. Se fija una edad "landmark" ($t_{\text{LM}}$), por ejemplo, 55 años.
2. Se selecciona únicamente a los pacientes que han sobrevivido libres del evento (por ejemplo, cáncer de pulmón) y no han sido censurados antes de esa edad $t_{\text{LM}}$.
3. Se recopilan todos los datos longitudinales disponibles de cada paciente *hasta* $t_{\text{LM}}$.
4. Se realiza la predicción de si experimentarán el evento dentro de una ventana de tiempo futura fija, denominada **horizonte** ($\Delta t = 5$ años en este caso).

### ¿Por qué se espera que el C-index aumente con landmarks mayores?
La hipótesis usual es que un landmark mayor (más tardío) suele mejorar el rendimiento predictivo (C-index más alto). Esto ocurre principalmente por dos razones:
*   **Mayor historial clínico:** Un paciente a los 65 años tiene una secuencia de mediciones longitudinales (analíticas de laboratorio, signos vitales) mucho más larga y rica que cuando tenía 45 años, lo que proporciona al modelo recurrent GRU-D una trayectoria de salud más completa y detallada.
*   **Proximidad al evento y señal biológica:** El horizonte de predicción de 5 años se sitúa en un rango de edad más avanzado, donde la prevalencia y la señal biológica de la enfermedad (como marcadores de deterioro o desajustes metabólicos acumulados) suelen ser más pronunciados y estables.

> [!NOTE]
> A edades de landmark muy tardías (ej. >68), el rendimiento puede fluctuar o decaer debido a la reducción en el tamaño de la muestra ($N$) y el número de eventos observados, ya que muchos pacientes ya habrán experimentado el evento o habrán sido censurados antes.

---

## 2. La Función `collect_raw_outputs`

Esta función se encarga de ejecutar la inferencia del modelo sobre todo un conjunto de datos (a través de un `DataLoader`) y recolectar las salidas crudas.

```python
def collect_raw_outputs(loader, model, device):
    """
    Collect model outputs over a full loader.

    Returns (risks, durations, events, landmarks) as numpy arrays.
    """
    model.eval()
    risks, durs, evs, lms = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            (labs, masks, deltas, lengths, sex_male,
             time_since_index, durations, events, landmarks, weights, ids) = batch
            r, _, _ = model(
                labs.to(device), masks.to(device), deltas.to(device),
                lengths.to(device), sex_male.to(device), time_since_index.to(device),
            )
            risks.extend(r.cpu().numpy())
            durs.extend(durations.numpy())
            evs.extend(events.numpy())
            lms.extend(landmarks.numpy())
    return (np.array(risks), np.array(durs),
            np.array(evs, dtype=float), np.array(lms))
```

### ¿Qué hace exactamente?
1.  **Inferencia por Lotes (Batch Inference):** Pone el modelo en modo de evaluación (`model.eval()`) y desactiva el cálculo de gradientes (`torch.no_grad()`). Recorre secuencialmente cada lote del cargador de datos, alimentando al modelo con las variables dinámicas de laboratorio (`labs`, `masks`, `deltas`, `lengths`) y demográficas (`sex_male`, `time_since_index`).
2.  **Extracción del Log-Riesgo ($r$ o $\theta$):** El modelo devuelve para cada paciente un valor escalar $r_i$ que representa su **log-riesgo relativo** (la salida de la última capa lineal de la red neuronal, equivalente a $\theta_i$ en un modelo de riesgos proporcionales).
3.  **Acumulación Global:** En lugar de promediar las pérdidas o los C-index por lotes, concatena los vectores de:
    *   `risks`: los riesgos relativos $\theta_i$ predichos.
    *   `durations`: el tiempo real observado $T_i$ hasta el evento o censura.
    *   `events`: el indicador binario $E_i$ de si ocurrió el evento (1) o censura (0).
    *   `landmarks`: la edad landmark del ejemplo.
4.  **Retorno:** Devuelve todo en forma de vectores NumPy para un cálculo estadístico global.

### ¿Para qué se usa?
Las métricas en supervivencia, como el Índice de Concordancia (C-index) o los estimadores de riesgo acumulado, son métricas **globales no lineales**. No se pueden calcular de forma local por lotes y promediar después (como la pérdida de entropía cruzada), sino que requieren ordenar y comparar todos los tiempos de evento y riesgos de la cohorte completa simultáneamente. `collect_raw_outputs` proporciona los datos crudos consolidados necesarios para realizar estos cálculos a nivel global sobre el conjunto de datos completo.

---

## 3. Estimación del Baseline Hazard ($H_0$) y la Función `compute_baseline_hazard`

### Fundamento Teórico: ¿Qué es $H_0$ y por qué lo necesitamos?
En la formulación de riesgos proporcionales de Cox, la función de riesgo instantáneo (*hazard rate*) de un paciente $i$ con características $\mathbf{x}_i$ en el tiempo $t$ es:

$$h_i(t) = h_0(t) e^{\theta_i}$$

Donde:
*   $\theta_i = g(\mathbf{x}_i)$ es el log-riesgo relativo (la salida de la red neuronal).
*   $h_0(t)$ es la **función de riesgo base (baseline hazard)**. Representa el riesgo instantáneo para un individuo con log-riesgo neutro ($\theta = 0$).

Durante el entrenamiento, el modelo optimiza los parámetros de la red neuronal mediante la **Verosimilitud Parcial (Partial Likelihood)**. Como se detalla en el documento de [cox_partial_likelihood.md](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025/2DA%20FASE/us-mulcia-2026-ramon/docs/cox_partial_likelihood.md), el factor $h_0(t)$ se cancela en la verosimilitud condicional. Esto permite aprender a clasificar el riesgo relativo entre pacientes sin modelar el riesgo base absoluto.

Sin embargo, para calcular la **probabilidad absoluta** de que un paciente experimente el evento antes de un horizonte de tiempo determinado $t$, la clasificación relativa no es suficiente; necesitamos el riesgo absoluto. La probabilidad acumulada de sufrir el evento (riesgo absoluto predicho) se deriva de la función de supervivencia $S(t)$:

$$P(\text{Evento} \le t) = 1 - S(t) = 1 - e^{-H_i(t)} = 1 - e^{-H_0(t) e^{\theta_i}}$$

Donde **$H_0(t) = \int_0^t h_0(u) du$** es la **función de riesgo acumulado base (baseline cumulative hazard)**. Por lo tanto, $H_0$ es el factor indispensable para convertir predicciones de orden relativo (log-riesgos) en probabilidades reales de evento clínico.

### El Estimador de Breslow
La función `compute_baseline_hazard` estima $H_0(t)$ usando el método no paramétrico clásico conocido como **Estimador de Breslow** aplicado al conjunto de entrenamiento:

$$\hat{H}_0(t) = \sum_{t_{(k)} \le t} \frac{d_k}{\sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j}}$$

Donde:
*   $t_{(k)}$ son los tiempos únicos de evento en el conjunto de entrenamiento.
*   $d_k$ es el número de eventos que ocurren en el instante $t_{(k)}$.
*   $\mathcal{R}(t_{(k)})$ es el conjunto de riesgo en el instante $t_{(k)}$ (pacientes cuya duración observada es mayor o igual a $t_{(k)}$).
*   $\theta_j$ es el log-riesgo predicho para el paciente $j$ del conjunto de entrenamiento.

### Funcionamiento de la Función y Estabilidad Numérica (Shift)
El código de la función implementa este estimador con un truco de estabilidad numérica para evitar desbordamientos del tipo `NaN` o `inf` al elevar al exponencial números muy grandes:

```python
def compute_baseline_hazard(risks, durations, events, horizon):
    risks     = np.asarray(risks,     dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events    = np.asarray(events,    dtype=np.float64)

    max_r   = risks.max()
    shifted = risks - max_r
    H0 = 0.0
    for t in np.unique(durations[events == 1]):
        if t > horizon:
            continue
        d        = np.sum((durations == t) & (events == 1))
        risk_set = durations >= t
        denom    = np.sum(np.exp(shifted[risk_set]))
        if denom > 0:
            H0 += d / denom
    return H0, max_r
```

1.  **Cálculo del Desplazamiento (`max_r`):** Obtiene el valor máximo de los log-riesgos predichos del conjunto de entrenamiento: `max_r` $= \max(\theta_{\text{train}})$.
2.  **Desplazamiento de Riesgos (`shifted`):** Resta este máximo a todos los riesgos: $\theta^{\text{shifted}}_j = \theta_j - \max(\theta_{\text{train}})$. Esto asegura que todos los valores pasados a la función exponencial sean no-positivos ($\le 0$), limitando $\exp(\theta^{\text{shifted}}_j) \le 1.0$ y previniendo desbordamientos del tipo `inf` al calcular $\exp(\theta^{\text{shifted}}_j)$.
3.  **Sumatorio de Breslow Desplazado:** Suma las fracciones de riesgo para todos los eventos únicos que ocurren antes del `horizon` temporal ($t \le \Delta t$):
    $$H_0^{\text{shifted}}(\text{horizon}) = \sum_{t_{(k)} \le \text{horizon}} \frac{d_k}{\sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j - \max(\theta_{\text{train}})}}$$
4.  **Relación Matemática con el Estimador Real:**
    Puesto que $e^{\theta_j - \max(\theta_{\text{train}})} = e^{\theta_j} \cdot e^{-\max(\theta_{\text{train}})}$, podemos reescribir el denominador:
    $$H_0^{\text{shifted}}(\text{horizon}) = \sum_{t_{(k)} \le \text{horizon}} \frac{d_k}{e^{-\max(\theta_{\text{train}})} \sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j}} = e^{\max(\theta_{\text{train}})} \cdot \hat{H}_0(\text{horizon})$$
    Por lo tanto, la función devuelve $H_0^{\text{shifted}}$ (que está escalado por un factor constante) y la constante `max_r`.

---

## 4. Calibración en el Landmark 55

La última celda evalúa la **Calibración** del modelo a la edad fija de 55 años con un horizonte de predicción de 5 años.

### Cálculo de Probabilidades de Riesgo
Una vez obtenido el baseline hazard escalado $H_0^{\text{shifted}}$ a partir de los datos de entrenamiento, calculamos la probabilidad de evento absoluta para cada paciente $i$ del conjunto de validación mediante la línea:

```python
pred_prob = 1.0 - np.exp(-H0 * np.exp(val_risks - max_r))
```

Esta línea es matemáticamente idéntica a la fórmula no desplazada, ya que cancela el factor de escala:

$$P(\text{Evento}_i \le \text{horizon}) = 1 - \exp\left( - H_0^{\text{shifted}} \cdot e^{\theta_{i,\text{val}} - \max(\theta_{\text{train}})} \right)$$
$$= 1 - \exp\left( - \left[ \hat{H}_0 \cdot e^{\max(\theta_{\text{train}})} \right] \cdot \left[ e^{\theta_{i,\text{val}}} \cdot e^{-\max(\theta_{\text{train}})} \right] \right)$$
$$= 1 - \exp\left( - \hat{H}_0 \cdot e^{\theta_{i,\text{val}}} \right)$$

Este paso demuestra cómo el truco de estabilidad computacional mantiene la exactitud matemática del modelo de Cox en la fase de predicción.

### Métricas Generales de Calibración
La celda imprime tres métricas consolidadas:
*   **Mean predicted 5-yr risk (`pred_prob.mean()`):** La probabilidad promedio predicha por el modelo de sufrir el evento en los próximos 5 años para la población viva a los 55 años.
*   **Observed event rate (`val_evs.mean()`):** El porcentaje de pacientes que efectivamente presentaron el evento dentro de ese horizonte temporal en el conjunto de validación.
*   **E/O ratio (Expected / Observed):** Es la tasa esperada sobre la observada:
    $$\text{E/O Ratio} = \frac{\text{Media de Probabilidad Predicha}}{\text{Tasa de Eventos Observada}}$$
    *   **E/O = 1.0 (Ideal):** El modelo predice exactamente la cantidad de eventos que ocurren en la cohorte.
    *   **E/O > 1.0 (Sobreestimación):** El modelo predice un número de eventos mayor de lo que ocurre en realidad.
    *   **E/O < 1.0 (Subestimación):** El modelo predice menos riesgo del real.

### Calibración por Decil (Decile Table)
Dado que un modelo puede tener un buen E/O promedio general pero estar descalibrado en subgrupos específicos de la población (ej. predecir excesivo riesgo a pacientes sanos y poco riesgo a pacientes enfermos), se realiza un desglose por deciles de riesgo:
1.  **Ordenamiento:** Se ordenan los pacientes de validación de menor a mayor riesgo predicho (`pred_prob`).
2.  **Partición:** Se divide a la población en 10 subgrupos de igual tamaño (`np.array_split(order, 10)`).
3.  **Comparación:** Para cada decil (del 1 al 10) se calcula:
    *   `Mean pred`: El riesgo promedio que el modelo asignó a los pacientes en ese decil.
    *   `Obs rate`: La proporción de pacientes de ese decil que sufrieron el evento en la realidad.
    *   `N`: La cantidad de pacientes en el decil.
    *   `Events`: Cuántos eventos reales ocurrieron en el decil.

**Interpretación:** En un modelo bien calibrado, para cada fila de la tabla, la columna `Mean pred` debe ser extremadamente cercana a `Obs rate`. Además, a medida que ascendemos del decil 1 al 10, la tasa observada (`Obs rate`) debería ir incrementando de forma progresiva, validando la capacidad del modelo para separar y cuantificar el riesgo real tanto en pacientes sanos como enfermos.

---

## 5. El Papel del "Horizon" (Horizonte Temporal) en los Datos y el Modelo

Para comprender plenamente el papel del horizonte temporal ($\Delta t$, en este caso de 5 años, definido como `HORIZON = 5.0`), es fundamental distinguir entre lo que hace la red neuronal durante el entrenamiento y cómo evaluamos sus predicciones después.

### A. La Arquitectura y la Función de Pérdida no "saben" del Horizonte
Tienes toda la razón en tu intuición: **la arquitectura del modelo recurrent (GRU-D) y su función de pérdida no discriminan por horizonte**.
*   **GRU-D:** Se limita a procesar la secuencia temporal y emitir un único valor de log-riesgo $\theta_i$ al final.
*   **Función de Pérdida (`cox_ph_loss`):** Se alimenta de los riesgos predichos y de los datos del dataset landmark:
    *   `durations`: el tiempo restante hasta el evento o censura (ej. si el landmark es 55 y el paciente desarrolla cáncer a los 62, su duración en este dataset es de 7.0 años).
    *   `events`: si experimentó o no el evento durante todo el seguimiento restante (ej. 1).
    La función de pérdida simplemente optimiza el modelo para que los pacientes con menor tiempo de supervivencia tengan riesgos más altos. **El modelo no está entrenado específicamente para un horizonte de 5 años, sino para clasificar a los pacientes por su supervivencia general restante.**

### B. ¿Dónde interviene el Horizonte?
El horizonte se introduce **exclusivamente en la fase de evaluación y calibración** (post-entrenamiento), y actúa en dos puntos críticos:

1.  **En la Estimación del Baseline Hazard ($H_0$):**
    En la llamada `compute_baseline_hazard(..., HORIZON)`, limitamos la acumulación del riesgo de referencia únicamente hasta el tiempo del horizonte:
    ```python
    for t in np.unique(durations[events == 1]):
        if t > horizon:  # Si el evento ocurre después del horizonte (ej. a los 7 años), se ignora
            continue
        # ... acumulación de H0 ...
    ```
    Esto permite que la estimación del riesgo acumulado base de referencia $H_0$ corresponda estrictamente al periodo $[0, \text{HORIZON}]$.

2.  **En la Probabilidad de Riesgo Absoluto:**
    Calculamos la probabilidad absoluta de evento para el horizonte especificado:
    $$P(\text{Evento}_i \le \text{horizon}) = 1 - e^{-H_0(\text{horizon}) \cdot e^{\theta_i}}$$
    Si quisiéramos evaluar un horizonte de 3 años, calcularíamos un $H_0(3.0)$ y obtendríamos probabilidades distintas, sin necesidad de reentrenar la red neuronal.

### C. Una Discrepancia Importante en el Cuaderno: El "Observed Event Rate"
Has hecho una observación sumamente aguda al plantear qué pasa con los pacientes que sufren el evento después de los 5 años (por ejemplo, a los 7 años del landmark).

En el código del cuaderno:
```python
print(f"Observed event rate      : {val_evs.mean():.4f}")
```
*   `val_evs` contiene la etiqueta cruda `event` del dataset. Esta etiqueta marca si el paciente *alguna vez* tuvo el evento durante todo el estudio (incluso si ocurrió a los 7 u 8 años del landmark).
*   Por lo tanto, la línea `val_evs.mean()` está calculando la tasa de eventos observados **para todo el seguimiento restante**, y no está restringida a los 5 años del horizonte.
*   Esto genera una discrepancia técnica en la calibración, ya que estamos comparando una probabilidad predicha a 5 años (`pred_prob`) con una tasa de eventos observada a mayor plazo.

> [!WARNING]
> **Corrección conceptual:** Para que la comparación de calibración sea matemáticamente rigurosa, la tasa de eventos observada debería censurarse al horizonte de 5 años. Es decir, se debería calcular:
> ```python
> val_evs_horizon = ((val_durs <= HORIZON) & (val_evs == 1)).astype(float)
> print(f"Observed 5-yr event rate : {val_evs_horizon.mean():.4f}")
> ```
> De esta forma, si un paciente experimenta el evento después del horizonte (ej. a los 7 años), se le trata como libre de evento a los 5 años (censura en el horizonte). Esta discrepancia es un detalle técnico muy común en implementaciones simplificadas de cuadernos de aprendizaje y tu capacidad para detectarla es excelente.

