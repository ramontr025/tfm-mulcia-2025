# Riesgo competitivo — guía técnica y didáctica

> Documento de apoyo para el TFM. Explica **qué es el riesgo competitivo**, por qué
> el pipeline actual **no lo modela** (trata la muerte como censura), y qué habría
> que hacer si en el futuro se decidiera pasar a un análisis con riesgos competitivos.
> Es un documento de comprensión y diseño, **no** una propuesta de cambio del pipeline.
>
> Enlaces relacionados: [`docs/survival_metrics.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/survival_metrics.md)
> (definición de las 5 métricas), [`docs/eval.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/eval.md)
> (implementación de la evaluación) y [`docs/build_cohort.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/build_cohort.md)
> (definición de evento y censura).

---

## 0. Resumen en cinco frases

1. **Censura** es no saber qué pasó con un paciente (se perdió el seguimiento); el IPCW la corrige.
2. **Riesgo competitivo** es saber qué pasó y que sea otra cosa (murió de otra causa); el IPCW **no** lo corrige.
3. El pipeline codifica la muerte como censura, así que estima $1-S_{KM}$ (riesgo **neto** o causa-específica), que es una **cota superior** de la incidencia acumulada real $CIF$.
4. La diferencia entre ambas **crece con la mortalidad por otras causas**; en datos con pocos eventos es pequeña, pero no nula.
5. El Brier/IPA actual **no está mal**, pero mide el target equivocado para la frase "probabilidad de desarrollar cáncer a 5 años"; las métricas de discriminación (C-index, AUROC) apenas se ven afectadas.

---

## 1. El ejemplo que rompe todo (2 pacientes)

Queremos estimar: **"¿qué proporción de estos pacientes desarrollará cáncer de pulmón en 2 años?"**

Dos pacientes, seguidos desde su entrada a cohorte ($t=0$):

| Paciente | Desenlace | Tiempo (años) |
|----------|-----------|---------------|
| A | Cáncer de pulmón (**evento de interés**) | 2 |
| B | Muerte por infarto (**otra causa**) | 1 |

### Lo que hace el pipeline hoy (muerte = censura)

Trata a B como si se hubiera "perdido": lo saca del grupo de riesgo y sigue como
si nada. Entonces, "entre los que seguían en riesgo al final", A tuvo cáncer:

$$1 - S_{KM}(2) = 1 - \left(1 - \tfrac{1}{1}\right) = 1.0 \;\Rightarrow\; \textbf{100 \%}$$

### Lo que dice la realidad (muerte = riesgo competitivo)

B **ya no puede** desarrollar cáncer: murió. No es información que falta, es un
desenlace observado. Aporta un "no" a la incidencia de cáncer:

$$CIF_{\text{cáncer}}(2) = S(1^-) \times \frac{1}{1} = 0.5 \times 1 = 0.5 \;\Rightarrow\; \textbf{50 \%}$$

### Moraleja

> Con solo 2 pacientes, el tratamiento "muerte = censura" da **100 %** y el
> correcto da **50 %**. El sesgo es enorme porque la causa competidora era
> frecuente (1 de cada 2). **El sesgo no viene de la censura** —B no era censura,
> era un desenlace— **viene de haber metido la muerte en el mismo cubo que la
> censura.**

Este es, en miniatura, exactamente el mecanismo del pipeline del TFM (§8).

---

## 2. Vocabulario mínimo

| Término | Definición | En tu tabla de cohorte |
|---------|-----------|------------------------|
| **Tiempo al evento** | Instante en que ocurre el desenlace desde la entrada | `duration` |
| **Evento de interés** | El desenlace que quieres predecir | cáncer de pulmón (`event=1`) |
| **Censura a la derecha** | Dejaste de observar al paciente **sin** saber su desenlace | fin de seguimiento (`event=0`) |
| **Censura no informativa** | El motivo de la censura es **independiente** del riesgo de evento de interés | supuesto de las métricas IPCW |
| **Riesgo competitivo** | Un **desenlace distinto** que impide el evento de interés | muerte por otra causa |
| **Causa competidora** | Ese otro desenlace | muerte (causa 2) |

La clave está en la frontera entre las dos últimas filas y la tercera: **la muerte
no es censura**. Un paciente que muere por otra causa no "desaparece": sabemos
que **no** tendrá cáncer (dentro de esa ventana), y ese "no" debe contar.

---

## 3. Censura y riesgo competitivo: no son lo mismo

Producen el mismo "no veo el evento de interés", pero se tratan de forma opuesta:

| | **Censura** (loss to follow-up) | **Riesgo competitivo** (muerte) |
|---|---|---|
| ¿Sabemos el desenlace? | No | Sí (es otro desenlace) |
| ¿Podría aún tener cáncer luego? | Sí, quizá | No, ya no |
| ¿Cómo se aporta a la incidencia? | Se repondera con IPCW | Cuenta como un "0" en la incidencia de cáncer |
| ¿Qué herramienta lo maneja? | IPCW (Graf) | Incidencia acumulada (Aalen–Johansen) / Fine–Gray |
| ¿El IPCW lo resuelve? | **Sí** | **No** |

**Trampa habitual:** como ambas se acaban representando igual en una tabla
`(tiempo, indicador 0/1)`, es tentador tratarlas igual. Pero la censura es
incertidumbre y el riesgo competitivo es información.

### 3.1 ¿Cómo funciona el estimador de censura $\hat{G}(t)$ y la ponderación IPCW?

Para entender por qué el IPCW corrige la censura pero falla con la muerte, conviene tener cristalino cómo se calcula y qué significa:

#### 1. ¿Qué significa $G(t) = P(C > t)$? (El sentido del signo $>$)
* $C$ es la variable aleatoria: **tiempo hasta la censura** (fecha de fin de seguimiento administrativo o pérdida).
* $C > t$ significa que el instante de censura ocurrirá **después** de $t$; es decir, **en el instante $t$ el paciente todavía no ha sido censurado (sigue en observación activa)**.
* Es el análogo simétrico de la función de supervivencia al evento:
  - $S(t) = P(T > t)$: probabilidad de no haber tenido el evento (cáncer) para el instante $t$.
  - $G(t) = P(C > t)$: probabilidad de no haber sido censurado para el instante $t$.

#### 2. El cálculo de $\hat{G}(t)$ es discreto y finito (no un producto infinito)
La fórmula de Kaplan–Meier aplicada a los tiempos de censura es:

$$\hat{G}(t) = \prod_{s \le t} \left(1 - \frac{d_s}{n_s}\right)$$

* **No es un producto continuo ni infinito**: en cualquier instante $s$ donde no se censura a nadie, $d_s = 0$, el factor es $(1 - 0/n_s) = 1$, y multiplicar por 1 no altera el valor acumulado.
* El producto solo se evalúa en los **instantes discretos observados** ($s_1 < s_2 < \dots \le t$) en los que efectivamente se registra al menos una censura en los datos.
* **$d_s$ es puntual, no acumulativo**: cuenta únicamente las personas censuradas en ese día o instante exacto $s$, sin arrastrar las censuras de fechas anteriores.
* **$n_s$ decrece**: es el número de pacientes aún en seguimiento justo antes de $s$. Las censuras y eventos previos ya han salido de $n_s$.

#### 3. La intuición del peso IPCW: ¿por qué quien llega lejos pesa más?
El peso asignado a una observación no censurada en el horizonte $t$ es el inverso:

$$w(t) = \frac{1}{\hat{G}(t)}$$

Como $\hat{G}(t)$ decrece con el tiempo conforme hay bajas ($1.0 \to 0.8 \to 0.5 \dots$), el peso $w(t)$ crece ($1.0 \to 1.25 \to 2.0 \dots$).

**La analogía de la sala de 100 personas:**
* **$t=0$**: Entran 100 pacientes a seguimiento. Nadie se ha marchado ($\hat{G}=1.0$). Cada uno pesa $w = 1/1.0 = 1$ (se representa a sí mismo).
* **$t=3$ años**: Se han marchado 20 pacientes por pérdida de seguimiento. Quedan 80 observados ($\hat{G}=0.80$). Cada uno pasa a pesar $w = 1/0.80 = 1.25$ para compensar a los 20 que faltan.
* **$t=5$ años**: Se han marchado 50 pacientes en total; solo quedan 50 observados ($\hat{G}=0.50$). Cada uno de los que aguantó hasta el horizonte sin censurarse pesa $w = 1/0.50 = 2.0$: **pesa por dos (por él mismo y por un compañero que se perdió en el camino)**.

#### 4. Por qué el IPCW se rompe si metes la muerte en el saco
Este mecanismo asume que los que se perdieron tenían el mismo riesgo que los que se quedaron (censura no informativa). Pero si metes a los **fallecidos** en $d_s$:
- Se infla artificialmente el peso de los que siguen vivos ($w$ se hace demasiado grande).
- Y se asume implícitamente que los muertos "se habrían diagnosticado de cáncer al mismo ritmo que los vivos si hubiéramos seguido mirándolos", lo cual es biológicamente imposible.

---

## 4. Los tres estimandos (y por qué solo uno es "el riesgo real")

Con dos causas (1 = cáncer, 2 = muerte) se definen tres cantidades distintas. **No son intercambiables.**

### 4.1 Riesgo causa-específica (hazard causa-específica $h_1(t)$)

Tasa instantánea de cáncer entre quienes **siguen vivos y sin cáncer**. Los que
mueren se sacan del cálculo (se "censuran" para causa 1). Responde a:

> "De los que llegan vivos a tiempo $t$, ¿cuál es el ritmo instantáneo de diagnóstico?"

Útil para **entender mecanismos** (¿qué covariables aceleran el cáncer?), pero
**no** es la probabilidad acumulada de cáncer.

### 4.2 Incidencia acumulada $CIF_1(t)$ (riesgo "bruto" o real)

La probabilidad **real** de haber desarrollado cáncer por tiempo $t$, en un
mundo donde la muerte también existe:

$$\boxed{CIF_1(t) = \int_0^t S(u)\, h_1(u)\, du}$$

donde $S(u)$ es la supervivencia **global** (libre de cáncer **y** de muerte,
todas las causas). Es **la** cantidad que uno quiere cuando dice "probabilidad de
que a este paciente le diagnostiquen cáncer a 5 años".

### 4.3 Subdistribución (hazard de subdistribución, Fine–Gray)

Un hazard "artificial" definido sobre un conjunto de riesgo que **mantiene**
dentro a quienes ya murieron (con pesos), calibrado para que

$$CIF_1(t) = 1 - \exp\!\left(-\int_0^t \lambda_1(u)\,du\right).$$

Sirve para **modelar directamente la CIF**: en un modelo Fine–Gray, $\exp(\beta)$
es un cociente de *subdistribution hazards* y va directo a la incidencia
acumulada. Es el análogo, para la CIF, de lo que Cox es para la supervivencia.

### 4.4 La relación incómoda

La supervivencia de Kaplan–Meier aplicada al cáncer **tratando la muerte como
censura** estima $1 - e^{-H_1(t)}$, con $H_1 = \int h_1$. Y como $S(u) \le e^{-H_1(u)}$
(su producto incluye además el efecto de la causa 2):

$$1 - S_{KM,1}(t) = 1 - e^{-H_1(t)} \;\ge\; CIF_1(t).$$

Es decir: **la vía "muerte = censura" sobreestima la incidencia acumulada**, y la
diferencia es exactamente el efecto de la mortalidad por otras causas.

---

## 5. Ejemplo completo a mano (10 pacientes)

Usamos la misma cohorte en todo el cálculo. Entradas a $t=0$, horizonte $t=5$ años.

| Paciente | Tiempo (años) | Desenlace |
|----------|--------------|-----------|
| A | 1 | Cáncer (causa 1) |
| B | 2 | Muerte (causa 2) |
| C | 3 | Cáncer (causa 1) |
| D | 4 | Muerte (causa 2) |
| E | 5 | Cáncer (causa 1) |
| F, G, H, I, J | 6 | Censura (fin de seguimiento) |

### 5.1 Vía pipeline actual: $1-S_{KM}$ tratando la muerte como censura

Solo los cánceres reducen $S$; las muertes y censuras salen del grupo de riesgo.

| $t$ | Suceso en $t$ | En riesgo $n_i$ | $d$ (cáncer) | $S_{KM}(t)$ | $1-S_{KM}(t)$ |
|-----|---------------|-----------------|--------------|-------------|----------------|
| 1 | cáncer A | 10 | 1 | $0.9$ | 0.100 |
| 2 | muerte B | 9 | 0 | $0.9$ | 0.100 |
| 3 | cáncer C | 8 | 1 | $0.9\times\frac78 = 0.7875$ | 0.2125 |
| 4 | muerte D | 7 | 0 | $0.7875$ | 0.2125 |
| 5 | cáncer E | 6 | 1 | $0.7875\times\frac56 = 0.65625$ | **0.34375** |
| 6 | censuras F–J | 5 | 0 | $0.65625$ | 0.34375 |

$$\Rightarrow\; 1 - S_{KM}(5) = 0.34375 \;\approx\; \textbf{34.4 \%}$$

### 5.2 Vía correcta: incidencia acumulada por Aalen–Johansen

Aquí $S$ es la supervivencia **global** (cáncer **y** muerte reducen $S$):

| $t$ | Suceso | En riesgo | Caída de $S$ | $S(t^-)$ (antes) | Aporte a $CIF_1$ | $CIF_1$ acumulada |
|-----|--------|-----------|--------------|------------------|------------------|-------------------|
| 1 | cáncer A | 10 | $1/10$ | 1.0 | $1.0\times\frac1{10}=0.1$ | 0.1 |
| 2 | muerte B | 9 | $1/9$ | 0.9 | — | 0.1 |
| 3 | cáncer C | 8 | $1/8$ | 0.8 | $0.8\times\frac18=0.1$ | 0.2 |
| 4 | muerte D | 7 | $1/7$ | 0.7 | — | 0.2 |
| 5 | cáncer E | 6 | $1/6$ | 0.6 | $0.6\times\frac1{6}=0.1$ | **0.3** |

$$\Rightarrow\; CIF_1(5) = 0.1 + 0.1 + 0.1 = \textbf{0.30} \;(30 \%)$$

*(La supervivencia global en $t=5$ es $S(5)=0.5$; y $CIF_1 + CIF_2 = 0.3 + 0.2 = 0.5$,
que cuadra con $1 - S(5)$.)*

### 5.3 Comparación y lectura

| Horizonte | $1 - S_{KM}$ (pipeline) | $CIF_1$ (correcto) | Sesgo absoluto | Sesgo relativo |
|-----------|------------------------|--------------------|----------------|----------------|
| 1 año | 0.100 | 0.100 | 0.000 | 0 % |
| 3 años | 0.2125 | 0.200 | +0.0125 | +6.3 % |
| 5 años | **0.34375** | **0.300** | **+0.04375** | **+14.6 %** |

Tres lecciones:

1. **Mientras no hay causa competidora, los dos coinciden** (t=1: ambos 0.1).
2. **El sesgo aparece y crece** a medida que ocurren muertes antes del horizonte.
3. En este ejemplo la mortalidad competitiva es alta (2 de 10), y el sesgo a 5
   años llega al **+14.6 % relativo**. Con menos mortalidad, el sesgo sería menor,
   pero del **mismo signo** (siempre por exceso).

> **Nota:** que la proporción cruda (3 cánceres / 10) coincida aquí con la $CIF$
> (0.30) es una **coincidencia numérica** de este ejemplo simétrico, no una regla.
> El estimador crudo ignora el tiempo y la censura; no lo uses como referencia.

---

## 6. Qué pasa con las métricas del TFM

Recordatorio de las métricas (ver [`docs/survival_metrics.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/survival_metrics.md) §4–5):

### 6.1 Brier IPCW e IPA — afectadas, y de una forma sutil

El Brier IPCW (Graf et al., 1999) es un *proper scoring rule*, pero **de un target
concreto**: mide el error al predecir $1 - e^{-H_1}$ (el riesgo neto), **no** la
$CIF$. Y como nuestro modelo predice exactamente eso —

```python
# eval.py, líneas ~123-124
surv_prob_model = np.exp(-h0 * np.exp(risks_sub - maxr))   # 1 - S_KM vía Breslow
surv_prob_null  = np.full(n, np.exp(-h0_null))
```

— **modelo y métrica comparten el mismo target (erróneo para la interpretación
clínica)**. Consecuencia: el Brier sale internamente coherente y **no te avisa del
sesgo**. Te dice que predices bien *otra* cantidad, no la que narras.

- El **IPA** ($1 - \mathrm{BS_{mod}}/\mathrm{BS_{nulo}}$) hereda lo mismo, pero como
  el modelo nulo usa el mismo target, buena parte del sesgo **se cancela en el
  cociente**: el IPA sigue siendo una medida **relativa** válida (cuánto mejora el
  modelo sobre la incidencia basal), pero no convierte el Brier absoluto en "el
  riesgo real".
- Conclusión operativa: **el Brier absoluto y cualquier cifra de "riesgo a 5
  años" concreta NO son la $CIF$**; el IPA, como comparación relativa, aguanta.

### 6.2 C-index y AUROC — casi inmunes

Las métricas de **discriminación** (C-index de Harrell, C-index IPCW de Uno,
AUROC acumulativa) son de **rango**: ordenan a los pacientes por riesgo relativo,
no miden magnitud. Que la muerte se trate como censura apenas cambia ese
ordenamiento. **Por eso las comparaciones entre modelos siguen siendo válidas**;
lo que cambia es la interpretación *absoluta* (calibración y riesgo puntual).

### 6.3 El "Brier de riesgo competitivo"

Existe un Brier específico para $CIF$ (p. ej. el de `pec`/`riskRegression` en R):
el peso IPCW debe **contar la causa competidora como desenlace 0** para la causa
de interés, **no** eliminarla del denominador. La diferencia con el vuestro:

| | Brier IPCW actual | Brier de riesgo competitivo |
|---|---|---|
| Target | $1 - e^{-H_1}$ (riesgo neto) | $CIF_1(t)$ (riesgo real) |
| Paciente que muere antes de $t$ | Sale del cálculo (censurado) | Cuenta como "no cáncer" (0) |
| Interpretación | Riesgo en ausencia de otras causas | Probabilidad real acumulada |

---

## 7. Cómo se calcula (herramientas)

> ⚠️ Nombres de paquetes y funciones anotados `[verificar]`: contrastar con la
> documentación vigente antes de usarlos.

### R (maduro, es lo recomendable)

| Objetivo | Paquete / función |
|---|---|
| CIF no paramétrica (Aalen–Johansen) | `survival::survfit(Surv(t, status_factor))`, `cmprsk::cuminc`, `etm::etm` `[verificar]` |
| Fine–Gray (subdistribución) | `cmprsk::crr`, `riskRegression::FGR`, `survival::finegray` (pesos) `[verificar]` |
| Aalen–Johansen multi-estado | `mstate`, `survival` con `Surv(type="mstate")` `[verificar]` |
| Brier e IPA para riesgos competitivos | `pec::pec`, `riskRegression`, `prodlim` `[verificar]` |

### Python (sin soporte nativo maduro)

- `sksurv`: **no** tiene $CIF$ ni Brier competitivo (solo el enfoque de un único evento).
- `lifelines`: **no** cubre riesgos competitivos a fecha de hoy `[verificar]`.
- Alternativas a evaluar: el paquete `cmprsk` (porte de R) `[verificar]`, llamar a
  R vía `rpy2`, o **implementar Aalen–Johansen y el Brier competitivo a mano**
  (es viable: son fórmulas explícitas).

---

## 8. Estado actual del TFM (con evidencia en el código)

**La muerte se codifica como censura** en
[`workflow/scripts/build_cohort.py`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/workflow/scripts/build_cohort.py) (líneas 97–105):

```python
cohort["event"] = cohort["lung_cancer_date"].notna().astype("int8")   # 1 = cáncer
cohort["end_date"] = (
    cohort["lung_cancer_date"]
    .combine_first(cohort["DEATHDATE"])            # ⬅️ la muerte entra aquí...
    .combine_first(cohort["last_encounter_date"])  # ...en el mismo bucket que la censura
)
cohort["duration"] = ((cohort["end_date"] - cohort["index_date"]).dt.days / 365.25).clip(lower=0)
```

Es decir, un paciente que muere sin cáncer queda con `event=0` y
`duration = DEATHDATE − index`, **idéntico a una censura administrativa**.

Después, [`workflow/scripts/eval.py`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/workflow/scripts/eval.py):

- línea ~86: `y_test = Surv.from_arrays(event=..., time=...)` → supervivencia de **dos estados** (no multi-estado).
- líneas ~123–134: Brier IPCW e IPA sobre esas predicciones.

Y en la memoria ya está redactado el estimador usado en
[`MemoriaTFM/03_metodos.qmd`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/MemoriaTFM/03_metodos.qmd)
(§"Brier IPCW e IPA frente al modelo nulo", citando `@graf1999assessment`, `@gerds2006consistent`).

### Diagnóstico

| Pregunta | Respuesta |
|---|---|
| ¿La censura (loss to follow-up) se corrige? | **Sí**, vía IPCW. Eso está bien. |
| ¿La muerte se modela como riesgo competitivo? | **No**: se trata como censura. |
| Estimando de facto | $1-S_{KM}$ (riesgo **neto**), cota **superior** de la $CIF$. |
| Cómo se narra | "probabilidad de desarrollar cáncer a 5 años" → **no coincide** con lo estimado. |
| ¿Cuánto? | No cuantificado. Es **pequeño si la mortalidad competitiva es baja**, siempre por exceso. |

---

## 9. Los dos marcos legítimos

No hay una única respuesta correcta: depende de **qué quieres afirmar**.

### Marco A — Riesgo causa-específica, declarado como tal

- Se **mantiene** el pipeline actual (muerte = censura) y Cox causa-específico.
- **Requisito:** reformular la afirmación: "predicción del **riesgo neto** de
  cáncer (en ausencia de muerte por otras causas)", y explicitar la limitación.
- **A favor:** cero cambios de código; es el enfoque de muchos modelos de riesgo
  de cáncer de pulmón tipo PLCO/Bach `[verificar]`; coherente con el IPA.
- **En contra:** el "riesgo a 5 años" no es directamente la probabilidad real de
  diagnóstico; subestima el papel de la mortalidad.

### Marco B — Riesgo competitivo (la $CIF$ como target)

- Se **cambia** el target a $CIF_1(t)$ y el estimador correspondiente.
- **Requisito:** Aalen–Johansen o Fine–Gray + Brier/IPA competitivos (§6.3, §7).
- **A favor:** la cifra coincide con la afirmación clínica ("probabilidad de
  desarrollar cáncer a 5 años"); metodológicamente más rico.
- **En contra:** toca el pipeline (implementación, no redacción); más carga; hay
  que validar de nuevo; y en un cohorte sintético puede aflorar que la mortalidad
  de Synthea no reproduce la dependencia real (ver §10).

| | Marco A (declarar) | Marco B (modelar) |
|---|---|---|
| Cambios de código | Ninguno | Aalen–Johansen + Brier competitivo |
| Afirmación válida | "riesgo neto / en ausencia de muerte" | "probabilidad real a 5 años" |
| Tipo de tarea | Redacción (limitaciones) | Implementación |
| Riesgo | Malinterpretar la cifra si no se declara | Más esfuerzo; validar de nuevo |

---

## 10. Caveats

1. **Datos sintéticos (Synthea).** El mecanismo de muerte de Synthea puede **no**
   reproducir la dependencia real (en la realidad, los fumadores mueren más).
   Por tanto, el sesgo medido *en estos datos* probablemente **subestima** el del
   mundo real `[razonamiento — verificar en la cohorte]`.
2. **Magnitud, no existencia.** La pregunta operativa no es "¿hay sesgo?" (sí, si
   hay muertes antes del horizonte) sino "¿es material?". Comparar $1-S_{KM}$ con
   la $CIF$ de Aalen–Johansen a 5 años lo responde.
3. **Supuesto de censura no informativa.** El IPCW ya asume que la censura es
   independiente del evento dado el modelo marginal; tratar la muerte como
   censura añade asumir además independencia entre muerte y cáncer (discutible).
4. **Cambiar de marco cambia las cifras, no las comparaciones.** La discriminación
   entre modelos (C-index/AUROC) apenas se mueve; lo que se revisa es la
   calibración y el riesgo puntual.

---

## 11. Checklist si algún día se implementa (Marco B)

> Tarea de **implementación**; requiere cambiar de agente/rol y no forma parte de
> la fase de redacción.

1. **Pasar a multi-estado**: `Surv(time, status)` con `status ∈ {0: censura, 1: cáncer, 2: muerte}`.
2. **Estimar la $CIF$**: Aalen–Johansen (no paramétrico) como referencia y/o Fine–Gray para el modelo.
3. **Predicciones de modelo**: obtener $CIF_1(t|x)$ por paciente (Fine–Gray o combinando los hazards causa-específicos de ambas causas).
4. **Brier competitivo**: IPCW donde la causa competidora aporta un "0" a la incidencia de cáncer, no se elimina.
5. **IPA** sobre ese Brier (agrupado como cociente de Brier, igual que hoy).
6. **Verificación**: comparar $1-S_{KM}$ con $CIF_1$ a 5 años sobre `holdout` y
   reportar el gap; comprobar que las cifras cuadran con el ejemplo de §5.
7. **Documentar** el nuevo estimando y actualizar [`docs/survival_metrics.md`](file:///c:/Users/rtrin/OneDrive/Escritorio/FORMACION/MASTER/TFM/tfm-mulcia-2025/tfm-mulcia-2025-1/docs/survival_metrics.md).
8. **Reentrenar y reevaluar** los cinco modelos bajo el nuevo esquema.

---

## 12. Glosario

- **$S_{KM}(t)$** — Supervivencia de Kaplan–Meier (un solo evento; lo demás se trata como censura).
- **$h_1(t)$** — Hazard causa-específica del cáncer (ritmo instantáneo entre los aún en riesgo).
- **$\lambda_1(t)$** — Hazard de subdistribución (Fine–Gray); va directo a la $CIF$.
- **$CIF_1(t)$** — Incidencia acumulada del cáncer: probabilidad real de desarrollarlo por $t$.
- **IPCW** — *Inverse Probability of Censoring Weighting*; reponderación por la probabilidad de censura.
- **Brier (IPCW)** — Error cuadrático medio de la probabilidad predicha, con reponderación por censura.
- **IPA** — *Index of Prediction Accuracy*: $1 - \mathrm{BS_{mod}}/\mathrm{BS_{nulo}}$.
- **Aalen–Johansen** — Estimador no paramétrico de la $CIF$ en presencia de riesgos competitivos.
- **Fine–Gray** — Modelo de regresión para la subdistribución (análogo de Cox para la $CIF$).

---

## 13. Referencias

> Marcadas `[verificar]`: comprobar la cita exacta antes de incluirla en la memoria
> (convención del `AGENTS.md` global).

- Fine, J. P., & Gray, R. J. (1999). *A proportional hazards model for the subdistribution of a competing risk.* JASA. `[verificar]`
- Andersen, P. K., Abildstrøm, S. Z., & Rosthøj, S. (2012). *Competing risks as a multi-state model.* Statistical Methods in Medical Research. `[verificar]`
- Austin, P. C., & Fine, J. P. (2017). *Accounting for competing risks in randomized controlled trials.* Statistics in Medicine. `[verificar]`
- Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999). *Assessment and comparison of prognostic classification schemes for survival data.* Statistics in Medicine. (Brier IPCW, ya citado en la memoria como `@graf1999assessment`.)
- Gerds, T. A., & Schumacher, M. (2006). *Consistent estimation of the expected Brier score in general survival models with right-censored event times.* Biometrical Journal. (Ya citado como `@gerds2006consistent`.)
- Kattan, M. W., & Gerds, T. A. (2018). *The index of prediction accuracy.* American Journal of Epidemiology. (IPA.)
- Wolbers, M., et al. (2009). *Concordance for prognostic models with competing risks.* `[verificar]`

---

*Documento de comprensión y diseño. No modifica el pipeline; cualquier paso hacia el
Marco B es una tarea de implementación aparte, con aprobación explícita.*
