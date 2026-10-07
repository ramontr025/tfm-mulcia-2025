# Fundamentos Teóricos del Modelo de Riesgos Proporcionales de Cox y su Función de Pérdida

El modelo de **Riesgos Proporcionales de Cox** (Cox, 1972) es la piedra angular del análisis de supervivencia moderno. A diferencia de los modelos de clasificación estándar, su objetivo no es predecir una probabilidad binaria de ocurrencia de un evento en un instante fijo, sino **modelar el tiempo hasta que ocurre dicho evento**, asignando a cada individuo una puntuación de riesgo coherente con su orden temporal de supervivencia.

A continuación, se presenta la derivación matemática formal y la corrección técnica de las premisas que conducen a la formulación de su **Verosimilitud Parcial (Partial Likelihood)** y su consiguiente función de pérdida.

---

## 1. La Premisa de la Función de Riesgo (Hazard Function)

La primera premisa del modelo establece que la **función de riesgo** (o *hazard rate*) para un individuo $i$ con un vector de covariables $\mathbf{x}_i$ en el instante $t$ se modela de forma multiplicativa:

$$h_i(t) = h_0(t) e^{\theta_i}$$

Donde:
*   $h_0(t)$ es la **función de riesgo base** (o *baseline hazard*), que describe cómo cambia el riesgo general a lo largo del tiempo para un individuo con covariables nulas ($\mathbf{x} = \mathbf{0}$). Esta función no se parametriza (se mantiene libre), lo que convierte a Cox en un modelo **semiparamétrico**.
*   $\theta_i = f(\mathbf{x}_i)$ representa el **log-riesgo** o riesgo relativo asociado a las características del individuo $i$. En la formulación clásica de Cox, es una combinación lineal ($\theta_i = \mathbf{\beta}^T \mathbf{x}_i$), mientras que en aproximaciones modernas de aprendizaje profundo (como *DeepSurv*), $\theta_i = g_{\mathbf{W}}(\mathbf{x}_i)$ es la salida escalar de una red neuronal.
*   $e^{\theta_i}$ es el factor de riesgo exponencial, siempre positivo, que escala de manera proporcional el riesgo base.

---

## 2. Probabilidad Condicional ante un Único Evento

Supongamos que en un instante de tiempo $t_i$ observamos que **exactamente un individuo $i$ sufre el evento**. Queremos calcular la probabilidad condicional de que, dado que ocurrió un evento en $t_i$, este haya sido sufrido específicamente por el individuo $i$ y no por cualquier otro que seguía en riesgo en ese momento.

Definimos el **conjunto de riesgo** en el instante $t_i$, denotado como $\mathcal{R}(t_i)$, como el grupo de todos los individuos que siguen bajo observación inmediatamente antes de $t_i$ (es decir, no han sufrido el evento aún ni han sido censurados). 

Bajo la teoría de procesos puntuales y asumiendo que los eventos no ocurren simultáneamente (tiempos continuos sin empates), la probabilidad condicional es:

$$P(\text{El evento en } t_i \text{ es del individuo } i \mid \text{Un evento ocurre en } t_i, \mathcal{R}(t_i)) = \frac{h_i(t_i)}{\sum_{j \in \mathcal{R}(t_i)} h_j(t_i)}$$

Sustituyendo la premisa de riesgos proporcionales $h_k(t) = h_0(t) e^{\theta_k}$:

$$P(\text{Individuo } i \text{ sufre el evento}) = \frac{h_0(t_i) e^{\theta_i}}{\sum_{j \in \mathcal{R}(t_i)} h_0(t_i) e^{\theta_j}} = \frac{e^{\theta_i}}{\sum_{j \in \mathcal{R}(t_i)} e^{\theta_j}}$$

**Corrección técnica crucial:** Observa cómo la función de riesgo base $h_0(t_i)$ se cancela tanto en el numerador como en el denominador. Esta elegante simplificación permite estimar los parámetros de riesgo $	heta$ sin necesidad de conocer ni modelar la forma de $h_0(t)$, que suele ser altamente compleja e irregular.

---

## 3. Generalización a Múltiples Eventos y Ordenamiento Temporal

En un escenario real, disponemos de una cohorte con múltiples individuos donde se observan diversos tiempos de evento no simultáneos $T_1 < T_2 < \dots < T_K$. 

### ¿Por qué es necesario ordenar por duración?
La naturaleza de los datos de supervivencia exige un **ordenamiento cronológico estricto de menor a mayor duración**. Esto se debe a que el conjunto de riesgo $\mathcal{R}(t)$ es dinámico y decrece con el tiempo:
1. Al inicio del estudio, todos los sujetos están en riesgo.
2. A medida que transcurre el tiempo, los individuos abandonan el conjunto de riesgo $\mathcal{R}(t)$ por dos motivos: porque experimentan el evento (muerte, fallo, etc.) o porque son **censurados** (pérdida de seguimiento, fin del estudio).
3. Por lo tanto, para evaluar la probabilidad del evento de $i$ en el instante $t_i$, el denominador **solo** debe incluir a los sujetos que llegaron vivos y sin censurar a ese instante preciso ($j \in \mathcal{R}(t_i)$). El ordenamiento por duración garantiza que el cálculo del denominador se actualice correctamente de forma secuencial.

### Verosimilitud Parcial (Partial Likelihood)
Bajo la hipótesis de que, condicionado a la historia acumulada del proceso, las ocurrencias de los eventos en cada instante ordenado $t_{(k)}$ son estadísticamente independientes, la probabilidad conjunta de observar la secuencia exacta de eventos se construye mediante el producto de las probabilidades condicionales individuales:

$$L_p(\theta) = \prod_{k=1}^{K} \frac{e^{\theta_{(k)}}}{\sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j}}$$

Donde el productorio recorre únicamente los $K$ instantes temporales donde se observó un evento real (excluyendo los tiempos de censura del numerador, aunque los individuos censurados sí participan en los conjuntos de riesgo $\mathcal{R}$ hasta su momento de censura).

---

## 4. Obtención de la Función de Pérdida (Negative Log-Partial Likelihood)

En optimización y aprendizaje profundo, el objetivo es minimizar una función de pérdida. Como la verosimilitud parcial $L_p(\theta)$ se desea maximizar, definimos nuestra función de pérdida $\mathcal{L}(\theta)$ como el **logaritmo negativo de la verosimilitud parcial**:

$$\mathcal{L}(\theta) = - \ln L_p(\theta) = - \sum_{k=1}^{K} \ln \left( \frac{e^{\theta_{(k)}}}{\sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j}} \right)$$

Aplicando las propiedades del logaritmo:

$$\mathcal{L}(\theta) = - \sum_{k=1}^{K} \left[ \theta_{(k)} - \ln \left( \sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j} \right) 
\right]$$

Esta es la expresión matemática final de la función de pérdida que se utiliza para entrenar redes neuronales en análisis de supervivencia (como en *loss functions* personalizadas de PyTorch o TensorFlow).

---

## 5. Ejemplo Práctico Paso a Paso (Ficticio)

Para asentar los conceptos, supongamos que tenemos una pequeña cohorte ficticia de **4 sujetos** (llamados $A$, $B$, $C$ y $D$) con los siguientes datos observados de tiempo de supervivencia ($T$), estado del evento ($E$, donde 1 indica evento y 0 indica censura) y sus log-riesgos predichos ($\theta$):

| Sujeto | Tiempo de Supervivencia ($T_i$) | ¿Ocurrió el Evento? ($E_i$) | Log-riesgo Predicho ($\theta_i$) |
| :---: | :---: | :---: | :---: |
| **$A$** | 2 meses | Sí ($E_A = 1$) | $\theta_A$ |
| **$B$** | 5 meses | Sí ($E_B = 1$) | $\theta_B$ |
| **$C$** | 7 meses | No (Censurado, $E_C = 0$) | $\theta_C$ |
| **$D$** | 10 meses | Sí ($E_D = 1$) | $\theta_D$ |

---

### Paso 1: Ordenamiento Cronológico e Identificación de Instantes con Evento
Para computar la verosimilitud parcial, solo consideramos los instantes de tiempo en los que ocurre **al menos un evento**. En este caso:
- **$t_1 = 2$** (fallece el sujeto $A$)
- **$t_2 = 5$** (fallece el sujeto $B$)
- **$t_3 = 10$** (fallece el sujeto $D$)

*Nota: A los 7 meses el sujeto $C$ se censura. No hay evento en ese instante, por lo que este tiempo no añade un término directo en el productorio de la verosimilitud parcial, pero el sujeto sí influye en los conjuntos de riesgo previos.*

---

### Paso 2: Cálculo de los Términos de Pérdida en Cada Instante de Evento (Sumas Acumulativas)
Para cada instante de tiempo ordenado $t_{(k)}$ donde ocurre un evento, aplicamos directamente la fórmula de la loss teórica para dicho instante:
$$\mathcal{L}_{(k)}(\theta) = \ln \left( \sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j} \right) - \theta_{(k)}$$

Donde:
- $\theta_{(k)}$ es el log-riesgo del sujeto que sufre el evento en $t_{(k)}$.
- El sumatorio $\sum_{j \in \mathcal{R}(t_{(k)})} e^{\theta_j}$ representa la **suma acumulativa de riesgos** del conjunto de sujetos que continúan en riesgo (es decir, que su duración de supervivencia es $\ge t_{(k)}$).

Procedemos a evaluar cada instante de evento cronológicamente:

#### Instante $t_1 = 2$ (Sujeto $A$)
- **Sujeto del evento:** $A$, por tanto $\theta_{(1)} = \theta_A$.
- **Sujetos en riesgo $\mathcal{R}(2)$:** $\{A, B, C, D\}$ (todos los sujetos del estudio, ya que todos tienen una duración $\ge 2$ meses).
- **Suma acumulativa de riesgos:** $e^{\theta_A} + e^{\theta_B} + e^{\theta_C} + e^{\theta_D}$.
- **Término de pérdida $\mathcal{L}_1(\theta)$:**
  $$\mathcal{L}_1(\theta) = \ln\left(e^{\theta_A} + e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_A$$

#### Instante $t_2 = 5$ (Sujeto $B$)
- **Sujeto del evento:** $B$, por tanto $\theta_{(2)} = \theta_B$.
- **Sujetos en riesgo $\mathcal{R}(5)$:** $\{B, C, D\}$ (el sujeto $A$ ya experimentó su evento a los 2 meses, por lo que sale del conjunto de riesgo; solo quedan los sujetos con duración $\ge 5$ meses).
- **Suma acumulativa de riesgos:** $e^{\theta_B} + e^{\theta_C} + e^{\theta_D}$.
- **Término de pérdida $\mathcal{L}_2(\theta)$:**
  $$\mathcal{L}_2(\theta) = \ln\left(e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_B$$

#### Instante $t_3 = 10$ (Sujeto $D$)
- **Sujeto del evento:** $D$, por tanto $\theta_{(3)} = \theta_D$.
- **Sujetos en riesgo $\mathcal{R}(10)$:** $\{D\}$ (los sujetos $A$ y $B$ ya experimentaron sus eventos y el sujeto $C$ se censuró a los 7 meses, por lo que el único sujeto con duración $\ge 10$ meses es $D$).
- **Suma acumulativa de riesgos:** $e^{\theta_D}$.
- **Término de pérdida $\mathcal{L}_3(\theta)$:**
  $$\mathcal{L}_3(\theta) = \ln\left(e^{\theta_D}\right) - \theta_D = \theta_D - \theta_D = 0$$

---

### Paso 3: Obtención de la Loss Total (Negative Log-Partial Likelihood)
La loss total del modelo para esta cohorte es la suma de los términos de pérdida individuales de cada instante de evento:

$$\mathcal{L}(\theta) = \sum_{k=1}^{3} \mathcal{L}_{(k)}(\theta) = \mathcal{L}_1(\theta) + \mathcal{L}_2(\theta) + \mathcal{L}_3(\theta)$$

Sustituyendo los términos que acabamos de calcular:

$$\mathcal{L}(\theta) = \left[ \ln\left(e^{\theta_A} + e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_A \right] + \left[ \ln\left(e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_B \right] + 0$$

$$\mathcal{L}(\theta) = \left[ \ln\left(e^{\theta_A} + e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_A \right] + \left[ \ln\left(e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_B \right]$$

### Interpretación de la Loss en este Ejemplo
Para que esta pérdida $\mathcal{L}(\theta)$ sea lo más pequeña posible (cercana a 0):
1. **$\theta_A$** debe ser lo suficientemente grande en relación con el resto de componentes en su respectivo conjunto de riesgo. Esto tiene sentido porque el sujeto $A$ es el que fallece primero y por tanto su riesgo estimado debería ser alto.
2. **$\theta_B$** debe ser lo suficientemente grande en relación con el conjunto de riesgo remanente en su respectivo instante.
3. Nótese que aunque $C$ fue censurado y no generó un término propio en la suma, **sí penaliza la loss indirectamente si su log-riesgo predicho $\theta_C$ es muy alto**, ya que influye en los denominadores de los instantes anteriores ($t=2$ y $t=5$), aumentando el valor de la loss si $\theta_C$ supera a $\theta_A$ o $\theta_B$.

---

### Paso 5: Escenarios de Ejemplo Numérico

Para ilustrar cómo influyen los log-riesgos predichos en el valor final de la loss, evaluemos dos escenarios opuestos utilizando nuestro ejemplo de 4 sujetos.

La ecuación de la loss simplificada que dedujimos es:
$$\mathcal{L}(\theta) = \left[ \ln\left(e^{\theta_A} + e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_A \right] + \left[ \ln\left(e^{\theta_B} + e^{\theta_C} + e^{\theta_D}\right) - \theta_B \right]$$

---

#### Escenario A: Predicción Correcta (El Modelo Acierta $\rightarrow$ Loss Baja)
En este caso, la red neuronal asigna riesgos acordes al orden temporal de fallecimiento real (mayor riesgo para el sujeto $A$ que muere primero, riesgo menor para los que viven más tiempo):
- $\theta_A = 3.0$ (Riesgo muy alto para el fallecimiento más temprano)
- $\theta_B = 1.0$ (Riesgo alto/intermedio)
- $\theta_C = -1.0$ (Riesgo bajo para el censurado a los 7 meses)
- $\theta_D = -3.0$ (Riesgo muy bajo para el que sobrevive más tiempo, 10 meses)

Calculamos las exponenciales de riesgo $e^{\theta_i}$:
- $e^{\theta_A} = e^3 \approx 20.086$
- $e^{\theta_B} = e^1 \approx 2.718$
- $e^{\theta_C} = e^{-1} \approx 0.368$
- $e^{\theta_D} = e^{-3} \approx 0.050$

Sustituimos en los dos términos de la loss:
1. **Primer término ($t=2$):** 
   $$\ln(20.086 + 2.718 + 0.368 + 0.050) - 3.0 = \ln(23.222) - 3.0 \approx 3.145 - 3.0 = 0.145$$
2. **Segundo término ($t=5$):** 
   $$\ln(2.718 + 0.368 + 0.050) - 1.0 = \ln(3.136) - 1.0 \approx 1.143 - 1.0 = 0.143$$

**Loss Total en Escenario A:**
$$\mathcal{L}(\theta) \approx 0.145 + 0.143 = 0.288$$
*El valor de la pérdida es muy cercano a 0, lo cual es óptimo y nos dice que el ordenamiento predicho es muy preciso.*

---

#### Escenario B: Predicción Incorrecta (El Modelo Falla $\rightarrow$ Loss Alta)
Aquí, el modelo predice lo contrario de la realidad, asignando un riesgo muy bajo al sujeto $A$ que muere primero y un riesgo muy alto a los que viven más tiempo:
- $\theta_A = -3.0$ (Riesgo muy bajo para quien muere primero)
- $\theta_B = -1.0$ (Riesgo bajo)
- $\theta_C = 1.0$ (Riesgo alto)
- $\theta_D = 3.0$ (Riesgo muy alto para quien sobrevive más tiempo)

Calculamos las exponenciales de riesgo $e^{\theta_i}$:
- $e^{\theta_A} = e^{-3} \approx 0.050$
- $e^{\theta_B} = e^{-1} \approx 0.368$
- $e^{\theta_C} = e^1 \approx 2.718$
- $e^{\theta_D} = e^3 \approx 20.086$

Sustituimos en los dos términos de la loss:
1. **Primer término ($t=2$):** 
   $$\ln(0.050 + 0.368 + 2.718 + 20.086) - (-3.0) = \ln(23.222) + 3.0 \approx 3.145 + 3.0 = 6.145$$
2. **Segundo término ($t=5$):** 
   $$\ln(0.368 + 2.718 + 20.086) - (-1.0) = \ln(23.172) + 1.0 \approx 3.143 + 1.0 = 4.143$$

**Loss Total en Escenario B:**
$$\mathcal{L}(\theta) \approx 6.145 + 4.143 = 10.288$$
*La pérdida es muy elevada (10.288 frente a 0.288) debido a que el orden de riesgos predicho es totalmente opuesto al orden real de supervivencia.*

---

## Filosofía del Modelo: ¿Por qué no es una Clasificación?

Es fundamental comprender que la salida de la red neuronal $	heta_i$ **no es una probabilidad** (no está acotada entre 0 y 1). 

*   **Sin noción de umbral estático:** En una clasificación binaria estándar (ej. regresión logística), se busca predecir si un paciente sobrevivirá o no a los 5 años. Esto ignora por completo la valiosa información temporal de *cuándo* ocurrió el deceso dentro de ese rango y penaliza por igual fallos tempranos y tardíos.
*   **Enfoque en el Riesgo Relativo y Ordenamiento:** La salida $	heta_i$ es un **riesgo relativo escalar** de carácter ordinal. Si el individuo $A$ tiene un tiempo de supervivencia menor que el individuo $B$ ($T_A < T_B$), la función de pérdida forzará a la red a ajustar sus pesos para que la salida de $A$ sea mayor que la de $B$ ($	heta_A > 	heta_B$). 
*   **Alineamiento temporal:** La pérdida penaliza severamente las incongruencias de ordenamiento: si un sujeto con un $	heta$ muy alto sobrevive mucho tiempo, o si un sujeto con un $	heta$ muy bajo fallece inmediatamente, la pérdida se dispara. Por ende, el modelo aprende a **ordenar los riesgos de los individuos en perfecta consonancia con su trayectoria temporal de supervivencia**.
