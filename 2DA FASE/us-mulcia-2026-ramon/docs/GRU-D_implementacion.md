# Dudas de Implementación y Diseño de la Red GRU-D

Este documento recopila las dudas resueltas sobre la implementación de la arquitectura **GRU-D** en PyTorch en el notebook `grud_student.ipynb`, explicando las correcciones de código y las justificaciones matemáticas detrás del diseño del modelo.

---

## 1. Corrección de los TODOs del Forward Pass (`TimeAwareGRUD`)

En la implementación inicial del estudiante se propusieron algunas aproximaciones con funciones nativas de Python/NumPy (como `np.exp` y `max`). Sin embargo, para no romper el flujo de gradientes (*backpropagation*) de PyTorch y operar correctamente con tensores de forma vectorizada, la implementación debe utilizar funciones de PyTorch:

```python
# ── TODO 1 ── Lab-value decay ─────────────────────────────────────
# Factor de decaimiento gamma_x para variables clínicas
gamma_x = torch.exp(-torch.clamp(self.decay_x(delta_t), min=0.0))

# ── TODO 2 ── Hidden-state decay ──────────────────────────────────
# Factor de decaimiento gamma_h para el estado oculto
gamma_h = torch.exp(-torch.clamp(self.decay_h(delta_t), min=0.0))

# ── TODO 3 ── Impute missing values ───────────────────────────────
# Imputación de variables clínicas (mezcla lineal de la última observación,
# la media empírica de la población y el factor gamma_x)
x_hat = m_t * x_t + (1.0 - m_t) * (gamma_x * x_last + (1.0 - gamma_x) * self.mean_x)

# ── TODO 4 ── Decay the hidden state ──────────────────────────────
# Decaimiento del estado oculto anterior h_{t-1} antes de la celda recurrente
h_t = gamma_h * h_t
```

### Razón del cambio a PyTorch:
- **`torch.clamp(..., min=0.0)`**: La función `max()` nativa de Python no puede procesar tensores de PyTorch de forma elemento a elemento. Se utiliza `clamp` o `torch.relu` para replicar la operación $\max(0, \cdot)$.
- **`torch.exp`**: Asegura que el gradiente fluya correctamente a través del cálculo del decaimiento temporal durante el entrenamiento.
- **`1.0` en flotante**: Se utilizan constantes flotantes (`1.0` en lugar de `1`) para garantizar compatibilidad de tipos en PyTorch y evitar errores de casteo implícito en la GPU.

---

## 2. ¿Cómo funciona la creación de la red GRU-D paso a paso?

El bucle temporal del método `forward` recorre cada paso de tiempo $t$ de la secuencia aplicando las siguientes transformaciones:

1. **Tasas de Decaimiento Entrenables ($\boldsymbol{\gamma}_{x_t}$ y $\boldsymbol{\gamma}_{h_t}$)**:
   Mide la pérdida de relevancia clínica en función del intervalo de tiempo $\boldsymbol{\delta}_t$ transcurrido desde la última observación válida:
   - **Para las variables ($\boldsymbol{\gamma}_{x_t}$)**: Se proyecta a través de una red lineal diagonal (`self.decay_x`) de tamaño `(1, input_dim)`.
   - **Para el estado oculto ($\boldsymbol{\gamma}_{h_t}$)**: Se proyecta a través de una red lineal densa (`self.decay_h`) de tamaño `(1, hidden_dim)`.

2. **Imputación Temporal Fisiológica ($\mathbf{\hat{x}}_t$)**:
   Si una variable clínica está presente en el instante actual ($m_t^d = 1$), se usa su valor medido. Si no ($m_t^d = 0$), se calcula un valor intermedio que decae desde el último valor observado ($x_{last}$) hacia el valor medio de la población ($\tilde{x}^d$) de forma exponencial.

3. **Decaimiento del Estado Oculto Anterior ($\mathbf{\hat{h}}_{t-1}$)**:
   Antes de alimentar la celda recurrente, el estado oculto del paso anterior se multiplica elemento a elemento por $\boldsymbol{\gamma}_{h_t}$. Esto simula la pérdida paulatina de influencia de la historia clínica antigua conforme pasa el tiempo sin nuevos registros.

4. **Actualización de la Celda Recurrente**:
   Se concatenan la entrada imputada $\mathbf{\hat{x}}_t$, la máscara binaria $\mathbf{m}_t$ y las variables estáticas y temporales directas, y se introducen en una unidad `nn.GRUCell` clásica utilizando el estado oculto decaído como estado inicial del paso.

---

## 3. Justificación del vector concatenado y la matriz de pesos de la máscara ($\mathbf{V}$)

### La duda conceptual:
La formulación matemática de GRU-D añade una matriz de pesos $\mathbf{V}$ que multiplica a la máscara de observación $\mathbf{m}_t$ para cada compuerta:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{\hat{x}}_t + \mathbf{U}_r \mathbf{\hat{h}}_{t-1} + \mathbf{V}_r \mathbf{m}_t + \mathbf{b}_r)$$

Sin embargo, en el código se concatenan todas las variables en un único vector y se introducen en una `nn.GRUCell` convencional:
`input_t = torch.cat([x_hat, m_t, sex_t, time_t], dim=-1)`

### ¿Dónde queda la matriz $\mathbf{V}$?
`nn.GRUCell` internamente multiplica una única matriz de pesos de entrada ($\mathbf{W}_{ih}$) por la entrada concatenada ($\mathbf{I}_t$). 
Matemáticamente, si dividimos esta matriz en bloques de acuerdo con las dimensiones de cada componente de la entrada, la multiplicación matricial se comporta así:

$$\mathbf{W}_{ih} \mathbf{I}_t = \begin{bmatrix} \mathbf{W}_{ih\_x} & \mathbf{W}_{ih\_m} & \mathbf{W}_{ih\_sex} & \mathbf{W}_{ih\_time} \end{bmatrix} \cdot \begin{bmatrix} \mathbf{\hat{x}}_t \\ \mathbf{m}_t \\ \text{sex}_t \\ \text{time}_t \end{bmatrix}$$

Al realizar el producto matricial por bloques obtenemos:
$$\mathbf{W}_{ih} \mathbf{I}_t = \mathbf{W}_{ih\_x} \mathbf{\hat{x}}_t + \mathbf{W}_{ih\_m} \mathbf{m}_t + \mathbf{W}_{ih\_sex} \text{sex}_t + \mathbf{W}_{ih\_time} \text{time}_t$$

### Equivalencia directa:
- La submatriz $\mathbf{W}_{ih\_x}$ actúa exactamente como la matriz de entrada $\mathbf{W}$ de GRU-D.
- La submatriz $\mathbf{W}_{ih\_m}$ actúa exactamente como la matriz $\mathbf{V}$ que pondera la máscara binaria $\mathbf{m}_t$.
- Las submatrices restantes ponderan de forma lineal las covariables estáticas y temporales directas sobre las compuertas de control.

### Ventaja de esta implementación:
Este diseño es sumamente eficiente puesto que una multiplicación matricial única de gran tamaño está altamente optimizada en GPU a nivel de hardware (CUDA). Permite heredar toda la velocidad de la celda recurrente nativa de PyTorch sin tener que reescribir una celda de GRU personalizada desde cero en Python.
