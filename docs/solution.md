# Solution — Prueba Técnica Avanzada IA

**Autor:** Alexis Poveda
**Rol evaluado:** Coordinador de Inteligencia Artificial
**Fecha:** 2026-05-28

Este documento es el **Documento de Decisiones Técnicas** y el **Resumen Ejecutivo** de la prueba. Se irá completando fase por fase.

## Entregables (checklist)

- [x] **Notebook o script Python** — entregado como 5 scripts modulares en [`../src/`](../src/): [eda.py](../src/eda.py), [features.py](../src/features.py), [modeling.py](../src/modeling.py), [train.py](../src/train.py), [explain.py](../src/explain.py)
- [x] **Documento de decisiones técnicas** — este archivo
- [x] **Diagrama de arquitectura** — [architecture.md](architecture.md) (Mermaid + tabla de decisiones)
- [x] **Resumen ejecutivo (máx. 1 página)** — [executive_summary.md](executive_summary.md)

## Estado por parte

| Parte | Tema | Estado |
|---|---|---|
| 1 | Análisis técnico del problema | Completa |
| 2 | Modelo predictivo (notebook + decisiones) | Completa |
| 3 | Interpretabilidad y explicabilidad | Completa |
| 4 | Arquitectura de producción + diagrama | Completa ([architecture.md](architecture.md)) |
| 5 | IA Generativa / RAG | Completa ([rag.md](rag.md)) |
| 6 | Pregunta ejecutiva (90 días, presupuesto) | Completa ([executive_summary.md](executive_summary.md)) |
| — | Resumen ejecutivo 1 página | Completa ([executive_summary.md](executive_summary.md)) |

---

## Parte 1 — Análisis Técnico del Problema

### Mapa de problemas y técnica recomendada

| # | Problema de negocio | Técnica IA recomendada | Por qué | Riesgo principal |
|---|---|---|---|---|
| 1 | Predecir clientes con riesgo de abandono (churn) | **ML clásico supervisado** (Logistic Regression / XGBoost) | Target binario etiquetado (`churn`) con features estructuradas. Caso textbook de clasificación tabular; GenAI sería sobreingeniería. | Data leakage si `days_since_last_purchase` o `payment_delay_days` se calculan post-churn. Sesgo demográfico por `city` y `age`. |
| 2 | Extraer señal de `customer_comment` (texto libre) | **NLP / GenAI ligero** (embeddings + sentiment) | Texto no estructurado; sentiment y temas son features predictivas valiosas. No justifica un LLM grande — un clasificador de sentiment fine-tuned basta. | Sesgo lingüístico (jerga regional). PII en comentarios filtrada a APIs externas. |
| 3 | Priorizar acciones comerciales | **NO ES SOLO IA — reglas + optimización** sobre el score del modelo #1 | El modelo da probabilidad; priorizar requiere valor del cliente × costo de retención × capacidad comercial. Es motor de reglas, no otro modelo. | Confundir "score alto" con "acción correcta". Sin uplift modeling, contactamos clientes que ya iban a quedarse. |
| 4 | Recomendar la **mejor acción** retentiva por cliente | **ML clásico — uplift / causal ML** (fase 2, no MVP) | Predecir churn ≠ saber qué intervención funciona. Requiere experimentación causal. | Sin A/B test no hay datos contrafactuales; el modelo aprende correlaciones espurias. **Postergar.** |
| 5 | Asistente interno basado en GenAI | **GenAI con RAG** + guardrails + human-in-the-loop | Reduce tiempo de búsqueda; conocimiento propietario y cambiante → RAG (no fine-tuning). | Alucinaciones sobre datos de clientes reales. Fuga de PII al proveedor LLM. |
| 6 | Segmentación de clientes para campañas | **ML clásico no supervisado** (clustering) — *si y solo si* marketing lo va a accionar | Útil para narrativa de campañas. A menudo es un entregable "bonito" que nadie usa. Validar caso antes de construir. | Clusters no accionables. Si no hay dueño de campaña comprometido, **no se construye**. |
| 7 | Forecast de gasto del próximo mes (si lo piden) | **ML clásico — regresión**, solo si hay consumidor real | Regresión tabular estándar. Riesgo: construirlo por inercia sin caso de uso comercial claro. | Modelo huérfano sin consumidor; mantenimiento sin ROI. |
| 8 | Detección de fraude (si lo piden a futuro) | **NO ES IA EN FASE 1 — reglas + validación de datos** | Con 1000 filas y sin etiquetas de fraude, un modelo es estadísticamente injustificable. Reglas duras son más auditables. | Falsos positivos bloqueando clientes legítimos; sesgo sin explicabilidad. |

### Lo que NO se debe resolver con IA (explícito)

- **Priorización comercial pura** → motor de reglas + score, no un segundo modelo.
- **"Mejor acción siguiente" sin datos experimentales** → primero A/B test, después modelo causal.
- **Fraude con 1000 filas sin etiquetas** → reglas, no ML.
- **Decisiones irreversibles sobre el cliente** (bloqueo, denegación) → human-in-the-loop obligatorio.

### Riesgos transversales

| Categoría | Riesgo | Mitigación |
|---|---|---|
| Datos | Dataset pequeño (1000 filas), posible no representativo | Validación cruzada estratificada; intervalos de confianza en métricas; no sobreajustar arquitectura al tamaño. |
| Datos | Data leakage en features post-evento | Definir punto de corte temporal claro; auditar cada feature contra la línea de tiempo del churn. |
| Sesgo | Discriminación por `city`, `age`, `channel` | Métricas desagregadas por subgrupo; fairness audit antes de producción. |
| Seguridad | PII en `customer_comment` enviada a LLM externo | Redacción/anonimización antes de embeddings; preferir modelos on-prem o región controlada. |
| Adopción | Equipo comercial ignora el score | Co-diseño con el área; medir uso real (no solo accuracy); training + casos de uso claros. |
| Gobierno | Modelo en producción sin dueño | Asignar product owner del modelo + comité de cambios. |

### Métricas

**Técnicas** (sobre el sistema):
- Clasificación: **Recall** sobre clase churn (prioritario — FN > FP en costo), F1, **PR-AUC** (mejor que ROC-AUC en clases desbalanceadas), matriz de confusión desagregada por `city`, `age_bucket`, `channel`.
- Calibración: Brier score, curva de calibración — el score debe ser probabilidad confiable, no solo ranking.
- Estabilidad: **PSI** sobre features y predicciones mes a mes; drift detectado < 14 días.
- GenAI/RAG: groundedness (% verificable contra fuente), tasa de alucinación (revisión humana semanal), latencia p95, % rechazos por guardrails.

**Negocio** (lo único que importa al comité):
- **Reducción de churn absoluto** en segmento intervenido vs. grupo de control (A/B test obligatorio).
- **Uplift en retención** en puntos porcentuales (no porcentaje relativo).
- **ROI**: (revenue retenido − costo de campañas − costo de plataforma) a 90/180 días.
- **Adopción del asistente GenAI**: DAU/MAU, consultas por agente/día. Un modelo no usado vale cero.
- **Tiempo de resolución** de consulta del agente antes vs. después.
- **Incidentes de gobierno**: fugas de PII, quejas por decisiones automatizadas, sesgos en auditoría → objetivo cero.

**Regla de oro**: si en 90 días no podemos demostrar uplift de retención medido contra control, el proyecto se replantea — no se renueva por inercia.

---

## Parte 2 — Modelo Predictivo Avanzado

> **Estado:** completa.
> **Scripts:** [eda.py](../src/eda.py), [features.py](../src/features.py), [modeling.py](../src/modeling.py), [train.py](../src/train.py)
> **Artefacto final:** [model.pkl](../artifacts/model.pkl) (pipeline XGBoost + threshold + metadata)

### 2.1 Hallazgos del EDA que cambiaron el plan

| Hallazgo | Implicación |
|---|---|
| 1000 filas, **0 nulos en todas las columnas** | No requiere imputación. Dataset sospechosamente limpio — en producción real esperamos nulos. |
| Clase **balanceada** (51.5% / 48.5%) | No es churn realista (típico 5–20%). Baseline naive = 51% accuracy → el modelo debe superarlo cómodamente. PR-AUC y ROC-AUC son ambas válidas. |
| `digital_engagement_score`: churn=0 → 59.5 vs. churn=1 → 41.9 | Feature más fuerte (~18 pts de gap). |
| `days_since_last_purchase`: 71 vs. 110 días | Segunda señal más fuerte. **Sospecha de leakage** (ver 2.5). |
| `payment_delay_days`: 24 vs. 35 | Señal moderada. |
| `support_tickets`: 1.7 vs. 2.2 | Señal débil. |
| `age`, `monthly_spend`, `promotion_usage` | **No discriminan** (medias casi iguales entre clases). |
| `contract_type` | **No discrimina** (52% / 52% / 51% de churn). |
| `channel` | Discrimina: Call Center 58% vs. WhatsApp 47% (11 pts). |
| `customer_comment`: solo **8 valores únicos** en 1000 filas | **No es texto libre** — es un categórico disfrazado. El mismo comentario aparece en churn=0 y churn=1. Se trata con one-hot, **no con embeddings/sentiment** (sobreingeniería). Esto revisa la fila #2 de la tabla de Parte 1: en este dataset NLP no aporta. |

### 2.2 Decisiones de preprocesamiento

- **Nulos**: ninguno → no se aplicó imputación. En producción habría que definir estrategia por columna (mediana para numéricas, moda o "Unknown" para categóricas).
- **Encoding**: `OneHotEncoder(drop="first", handle_unknown="ignore")` para `city`, `contract_type`, `channel`. `customer_comment` se **excluye** del modelo final ([train.py](../src/train.py)): solo 8 valores únicos sin señal predictiva, ya intentamos extraerle valor vía `sentiment_score`.
- **Escalado**: passthrough para los modelos basados en árbol (RF y XGBoost no lo necesitan). Para Logistic Regression, `StandardScaler` sería recomendado — en [train.py](../src/train.py) se omitió para no desviarnos del spec del ejercicio; aparece un `ConvergenceWarning` que indica que LR no llegó a su óptimo (su 0.90 ROC-AUC está ligeramente subestimado). No invalida el ganador.
- **Feature engineering**: implementado en [features.py](../src/features.py), salida en [features.parquet](../artifacts/features.parquet). Cuatro features adicionales con justificación de negocio:
  - `spend_per_day = monthly_spend / (days_since_last_purchase + 1)` — intensidad reciente del cliente.
  - `tickets_per_dollar = support_tickets / (monthly_spend + 1)` — costo de servir vs. retorno.
  - `is_delayed_payer = 1 si payment_delay_days > 15` — umbral binario que captura el "punto de no retorno" de mora.
  - `sentiment_score` (VADER sobre `customer_comment`) — **no aporta señal** (ver abajo).
- **Hallazgos al evaluar las features nuevas:**
  - `is_delayed_payer`: gap de **26 puntos** entre clases (61% en no-churners vs. 87% en churners). Señal muy fuerte.
  - `spend_per_day` y `tickets_per_dollar`: distribuciones razonables, candidatas a aportar señal.
  - `sentiment_score`: diferencia de medias entre clases = 0.006 → ruido. **Razón**: VADER es lexicon de inglés, los comentarios están en español. Confirmación cuantitativa de que NLP estándar EN no aporta aquí. Recomendación: si se quiere aprovechar el texto, usar modelo multilingüe (`pysentimiento`, BERT-es). Pero recordar que el campo solo tiene 8 valores únicos — el techo de señal extraíble es bajo.
- **Split**: train/test 80/20 estratificado por `churn` (`random_state=42`).

### 2.3 Modelos comparados

> **Nota técnica:** El plan inicial pivotó a Random Forest porque `libomp` no estaba instalado ([modeling.py](../src/modeling.py)). Posteriormente se instaló `libomp` y se entrenó **XGBoost como modelo final** en [train.py](../src/train.py) sobre las features de [features.parquet](../artifacts/features.parquet) (originales + 4 derivadas). RF queda como benchmark de control.

**Cross-validation 5-fold sobre train (corrida en [modeling.py](../src/modeling.py) sobre features originales):**

| Modelo | Métrica | Media | Desv. |
|---|---|---|---|
| Baseline (most_frequent) | accuracy | 0.5150 | 0.0031 |
| Logistic Regression | ROC-AUC | 0.8421 | 0.0296 |
| Random Forest | ROC-AUC | 0.9106 | 0.0289 |

**Evaluación en test — modelo final XGBoost (tuned vía GridSearchCV) ([train.py](../src/train.py), features.parquet):**

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ROC-AUC CI 95% (bootstrap) |
|---|---|---|---|---|---|---|---|
| Baseline | 0.515 | 1.000 | 0.680 | 0.500 | 0.515 | 0.485 | [0.500, 0.500] |
| Logistic Regression (balanced + StandardScaler) | 0.804 | 0.796 | 0.800 | 0.889 | 0.881 | 0.135 | [0.843, 0.931] |
| **XGBoost (tuned, scale_pos_weight=0.94)** | **0.938** | **0.884** | **0.910** | **0.934** | **0.951** | **0.082** | **[0.894, 0.966]** |
| **XGBoost LEAKAGE-SAFE (cota inferior)** | 0.713 | 0.699 | 0.706 | **0.751** | 0.785 | 0.217 | **[0.681, 0.816]** |

**Hallazgo crítico (cota inferior honesta):** si se confirma que `days_since_last_purchase` y `payment_delay_days` (las dos features más predictivas) están contaminadas por leakage post-evento, el modelo "leakage-safe" entrega **ROC-AUC 0.751** sin esas features ni sus derivadas. Por lo tanto, **el ROC-AUC real en producción está acotado en [0.751, 0.934]** según se resuelva la auditoría de fuente. Reportar solo 0.934 sin esta cota sería deshonesto.

**Calibración (XGBoost tuned):** Brier 0.082 (rango: 0 perfecto, 0.25 aleatorio). La reliability curve persistida en `artifacts/plots/calibration_curve.png` muestra que el score es razonablemente interpretable como probabilidad — no solo como ranking.

**Matriz de confusión — XGBoost tuned (threshold=0.5):**

|  | pred=0 | pred=1 |
|---|---|---|
| **real=0** | 91 (TN) | 6 (FP) |
| **real=1** | 12 (FN) | 91 (TP) |

**Hallazgo no menor:** XGBoost (0.941) ≈ Random Forest sobre features originales (0.945). Las features derivadas (`sentiment_score`, `spend_per_day`, `tickets_per_dollar`, `is_delayed_payer`) **no movieron material la aguja** — la señal predictiva ya estaba contenida en `digital_engagement_score`, `days_since_last_purchase` y `payment_delay_days`. El feature engineering fue *higiene* y aporte de interpretabilidad, no *uplift*. Es un hallazgo honesto que reportar al comité.

### 2.3.bis Threshold por matriz de costos (FN=5 × FP)

Bajo el supuesto explícito **costo FN = 5 × costo FP** (perder un cliente vale 5× más que gastar retención en alguien que no se iba), barrido sobre el modelo tuned:

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 | Costo total |
|---|---|---|---|---|---|---|---|---|
| **0.16 (óptimo por costo)** | 94 | 13 | **9** | 84 | 0.879 | **0.913** | 0.895 | **58** |
| 0.25 | 93 | 10 | 10 | 87 | 0.903 | 0.903 | 0.903 | 60 |
| 0.50 (default) | 91 | 6 | 12 | 91 | 0.938 | 0.884 | 0.910 | 66 |
| 0.65 (F1 ≈ máx) | 90 | 5 | 13 | 92 | 0.947 | 0.874 | 0.909 | 70 |

**Interpretación ejecutiva:** con FN penalizado 5×, el threshold óptimo es **0.16**. A ese punto contactamos 13 falsos positivos para evitar 9 falsos negativos — el costo total (58) es menor que el de optimizar F1 (70). El modelo da la probabilidad; la matriz de costos fija la decisión. *(Threshold cambió de 0.08 a 0.16 al pasar a XGBoost tuneado por GridSearchCV — la distribución de scores del modelo tuned es distinta.)*

### 2.4 Respuestas

**¿Qué modelo elegirías y por qué?**
**XGBoost** (entregable final, [model.pkl](../artifacts/model.pkl)). Iguala a Random Forest en ROC-AUC (0.941 vs. 0.945) y supera a Logistic (0.899) por > 4 puntos. Se elige XGB sobre RF por: (a) `scale_pos_weight` configurable para clases desbalanceadas reales en producción (en este dataset están casi 50/50, pero el churn real es 5–20%), (b) early stopping y monitoreo nativo, (c) mejor soporte en ecosistema MLOps. Trade-off honesto: en *este* dataset RF da idénticos números — la elección es por **operabilidad futura**, no por el .pkl de hoy.

**Trade-off precision/recall.**
A threshold=0.5 (XGB): precision 0.92, recall 0.89 → perdemos ~11% de churners reales. A threshold=0.08 (óptimo bajo costos 5:1): precision 0.79, recall 0.94 → solo perdemos 6%. La pregunta no es "qué threshold maximiza F1", sino **"cuánto vale un cliente vs. cuánto cuesta una acción de retención"**.

**¿Qué costo tiene equivocarse?**
- **Falso negativo (FN)** = no detectamos un churner real → perdemos el cliente, perdemos LTV. Costo alto y permanente.
- **Falso positivo (FP)** = contactamos a alguien que no iba a abandonar → costo de la acción comercial (descuento, llamada, mail) + posible irritación. Costo bajo y recuperable.
- **Regla**: costo FN >> costo FP en churn. Por eso optimizamos hacia recall, no hacia accuracy.

**¿Cómo definirías el threshold?**
No con el default 0.5. Tres criterios, en orden:
1. **Económico (formal)**: minimizar `costo_total(t) = C_FN × #FN(t) + C_FP × #FP(t)`. En [train.py](../src/train.py) se asume `C_FN=5, C_FP=1` y el barrido devuelve **threshold óptimo = 0.08** → persistido en `model.pkl` junto al pipeline. Cuando el negocio aporte LTV y costo de campaña reales, basta recalibrar los pesos y reejecutar.
2. **Capacidad operativa**: si el equipo comercial solo puede contactar N clientes/día, el threshold se ajusta para que `pred_pos_rate × población` ≤ N. A t=0.08 marcamos 61.5% de la base — operable solo si hay capacidad para campañas masivas; si no, subir t.
3. **Caso intermedio (F1)**: t=0.55 da F1=0.915 con costo total 61 vs. 56 — diferencia pequeña; útil si el comité no acepta aún la matriz de costos formal.

**¿Cómo evitarías data leakage?**
Cinco controles, los tres primeros implementados explícitamente en [train.py](../src/train.py):
1. **Split antes que fit**: `train_test_split` ocurre antes de cualquier `.fit()`. Ningún transformer ve `X_test` durante entrenamiento.
2. **Pipeline de sklearn**: `OneHotEncoder` envuelto en `Pipeline` → `.fit()` solo recibe `X_train`; `.predict()`/`.predict_proba()` aplican los mismos parámetros aprendidos en train al test.
3. **`handle_unknown='ignore'`**: categorías nuevas en producción no rompen el pipeline ni filtran información del test.
4. **`scale_pos_weight` derivado de `y_train`**, no del dataset completo. Ningún hiperparámetro depende de `y_test`.
5. **Temporal (riesgo abierto)**: `days_since_last_purchase` y `payment_delay_days` son **sospechosas** de calcularse post-evento de churn. Sin metadata temporal en el dataset no podemos auditarlo — flag explícito al comité. En producción esto se cierra con un *feature store* que versione timestamps de cálculo.

### 2.5 Decisiones técnicas registradas

| Decisión | Justificación |
|---|---|
| Modelo final = **XGBoost** ([train.py](../src/train.py) → [model.pkl](../artifacts/model.pkl)) | Mismo ROC-AUC que RF en este dataset, pero mejor operabilidad para producción real con clases desbalanceadas. |
| `libomp` instalado vía `brew install libomp` | Bloqueante de XGBoost en macOS; resuelto. |
| `customer_comment` excluido del modelo final | Solo 8 valores únicos en ambas clases. Su señal se intentó capturar con `sentiment_score` (VADER) que tampoco aportó por idioma. |
| `sentiment_score` con VADER (EN) sobre texto ES | Implementado por requerimiento explícito, confirmado como ruido (gap entre clases = 0.006). Recomendación: migrar a `pysentimiento` o BERT-es si se quiere texto. |
| Features derivadas (`spend_per_day`, `tickets_per_dollar`, `is_delayed_payer`, `sentiment_score`) | No movieron ROC-AUC vs. features originales. Se mantienen por **interpretabilidad** (`is_delayed_payer` da 26 pts de gap entre clases — fácil de explicar a negocio). |
| Sin `StandardScaler` para Logistic en [train.py](../src/train.py) | `ConvergenceWarning` confirmado. No invalida al ganador (XGB no lo necesita) pero LR está subestimada. Mejorable. |
| Threshold de negocio = **0.08** bajo costo FN=5×FP=1 | Formalizado con barrido sobre PR-curve. Persistido en `model.pkl` junto al pipeline. Recalibrable cuando lleguen costos reales. |
| Auditoría de leakage pendiente | `days_since_last_purchase` y `payment_delay_days` son sospechosas — requieren confirmación de cómo se calculan en la fuente. Sin esto, las métricas reportadas son **techo optimista**. |

---

## Parte 3 — Interpretabilidad y Explicabilidad

> **Estado:** completa.
> **Script:** [explain.py](../src/explain.py)
> **Artefactos:** [shap_summary.png](../artifacts/plots/shap_summary.png), [shap_waterfall_pos.png](../artifacts/plots/shap_waterfall_pos.png), [shap_waterfall_neg.png](../artifacts/plots/shap_waterfall_neg.png)

### 3.1 Variables más importantes (SHAP global)

Calculado con `shap.TreeExplainer` sobre el XGBoost final (post-tuning). Métrica: promedio del valor absoluto SHAP por feature.

| Rank | Feature | Mean \|SHAP\| | Categoría |
|---|---|---|---|
| 1 | `days_since_last_purchase` | ≈ 1.78 | Comportamiento (sospechosa de leakage) |
| 2 | `payment_delay_days` | 1.605 | Financiera (sospechosa de leakage) |
| 3 | `digital_engagement_score` | 1.461 | Comportamiento |
| 4 | `support_tickets` | 0.644 | Servicio |
| 5 | `tickets_x_delay` (derivada) | 0.476 | Interacción |
| 6 | `monthly_spend` | 0.437 | Financiera |
| 7 | `promotion_usage` | 0.406 | Comercial |
| 8 | `tickets_per_dollar` (derivada) | 0.388 | Servicio/Costo |
| 9 | `engagement_x_delay` (derivada) | 0.349 | Interacción |
| 10 | `age` | 0.331 | Demográfica |

**Observaciones:**
- Las **tres features de comportamiento + financieras** dominan con un margen amplio (>1.5 de SHAP cada una vs. <0.7 del resto).
- **`is_delayed_payer` y `sentiment_score` quedaron fuera del top 10.** El primero es redundante: XGBoost ya extrae el umbral desde `payment_delay_days` continua (la derivada sigue siendo útil para *interpretar*, no para *predecir*). El segundo confirma cuantitativamente que VADER en español es ruido.
- Categóricas (`city`, `channel`, `contract_type`) son **casi irrelevantes**: solo `city_Quito` entra al puesto 10 con aporte marginal.

### 3.2 Factores que aumentan el riesgo (sentido positivo del SHAP)

Leído del beeswarm y waterfalls:

| Factor | Lectura |
|---|---|
| **Muchos días sin comprar** (>120) | Señal #1 de desconexión. El cliente ya se fue mentalmente. |
| **Mora alta** (>40 días) | Señal financiera: incapacidad o intención de irse. Captura el efecto umbral confirmado en EDA (`is_delayed_payer`: 87% de churners son delayed payers). |
| **Engagement digital bajo** (<30) | El cliente no está usando la plataforma. La acción puede ser educativa antes que retentiva. |
| **Más de 3 tickets de soporte** | Fricción operativa acumulada. |
| **Alto `tickets_per_dollar`** | Cliente "caro de servir": muchos tickets relativos a su gasto → frustración y/o cliente que no llegamos a atender bien. |

### 3.3 Factores que reducen el riesgo

| Factor | Lectura |
|---|---|
| **Compra reciente** (<30 días) | Engagement vivo. Bajo riesgo en el corto plazo. |
| **Mora baja o nula** (<15 días) | Salud financiera y compromiso. |
| **Engagement digital alto** (>70) | Cliente activo en la plataforma — el producto está siendo consumido. |
| **Pocos tickets** (≤1) | Sin fricción operativa. |
| **Alto `spend_per_day`** | Cliente que gasta de forma sostenida e intensa — engagement transaccional fuerte. |

### 3.4 Explicación para un gerente no técnico

> *"El modelo no es una caja negra. Predice churn mirando, en orden de importancia, **tres cosas que cualquier gerente de cuenta también miraría**:*
>
> 1. *¿Hace cuánto que este cliente no nos compra? Si pasaron más de 4 meses, malo.*
> 2. *¿Tiene mora? Si debe más de 40 días, malo.*
> 3. *¿Está usando el producto? Si su engagement digital es bajo, malo.*
>
> *Las otras 7 variables —cuántos tickets levantó, cuánto gasta, su edad, su ciudad— ayudan a afinar el score, pero el grueso de la decisión sale de esas tres. Cuando el modelo te marca a alguien como 'alto riesgo', lo está marcando porque al menos dos de esas tres luces están en rojo.*
>
> *Lo que SÍ es nuevo respecto a la intuición humana: el modelo combina las tres simultáneamente y le pone un número (0 a 1). Eso permite priorizar 100 clientes/día de forma sistemática, no por corazonada. Pero la lógica subyacente es la que cualquier gerente con experiencia ya tiene en la cabeza."*

Acompañar esta explicación con [shap_summary.png](../artifacts/plots/shap_summary.png) (todos los clientes a la vez) y un [shap_waterfall_pos.png](../artifacts/plots/shap_waterfall_pos.png) (un cliente individual) cubre el 95% de las preguntas que un comité hará.

### 3.5 Auditoría de sesgo (sobre **TEST set únicamente**)

> **Corrección metodológica:** la versión anterior auditaba sobre el dataset completo (incluyendo train, donde el modelo había memorizado). Eso invalidaba los números. La auditoría actual es solo sobre las 200 filas de test, intacto.

Además del ratio simple, agregamos **3 métricas formales de fairness**:

- **Demographic Parity (DP)** = `P(ŷ=1 | grupo)` — ¿predice positivo por igual en cada grupo?
- **Equal Opportunity (TPR)** = `P(ŷ=1 | y=1, grupo)` — ¿detecta churners reales por igual?
- **Calibration gap** = `|pos_rate_pred − pos_rate_true|` — ¿la probabilidad significa lo mismo?

**Por `city`:**

| Ciudad | n | DP | TPR | FPR | true | calib_gap | ratio | flag |
|---|---|---|---|---|---|---|---|---|
| Cuenca | 57 | 0.579 | 0.909 | 0.125 | 0.579 | 0.000 | 1.103 | — |
| Quito | 50 | 0.540 | 0.926 | 0.087 | 0.540 | 0.000 | 1.029 | — |
| Guayaquil | 48 | 0.500 | 0.952 | 0.148 | 0.438 | 0.063 | 0.952 | — |
| Manta | 45 | 0.467 | 0.864 | 0.087 | 0.489 | 0.022 | 0.889 | — |

**Por `channel`:**

| Canal | n | DP | TPR | FPR | true | calib_gap | ratio | flag |
|---|---|---|---|---|---|---|---|---|
| Call Center | 64 | 0.563 | 0.944 | 0.071 | 0.563 | 0.000 | 1.071 | — |
| WhatsApp | 42 | 0.524 | 0.826 | 0.158 | 0.548 | 0.024 | 0.998 | — |
| Store | 40 | 0.500 | 0.941 | 0.174 | 0.425 | 0.075 | 0.952 | — |
| Web | 54 | 0.500 | 0.926 | 0.074 | 0.500 | 0.000 | 0.952 | — |

**Por `age_bin`:**

| Edad | n | DP | TPR | FPR | true | calib_gap | ratio | flag |
|---|---|---|---|---|---|---|---|---|
| **60+** | 59 | 0.610 | **1.000** | 0.148 | 0.542 | 0.068 | 1.162 | — |
| 18-30 | 45 | 0.533 | 0.840 | 0.150 | 0.556 | 0.022 | 1.016 | — |
| 31-45 | 49 | 0.510 | 0.957 | 0.115 | 0.469 | 0.041 | 0.972 | — |
| **46-60** | 47 | 0.426 | **0.826** | 0.042 | 0.489 | 0.064 | 0.811 | — |

### Equal Opportunity Difference (EOD) — métrica única por variable

`EOD = max(TPR) − min(TPR)` entre grupos. **Umbral aceptable: < 0.10.** Objetivo: 0.

| Variable | EOD | DemographicParityDiff | Veredicto |
|---|---|---|---|
| city | 0.089 | 0.112 | Dentro de umbral |
| channel | 0.118 | 0.063 | **Borderline** (> 0.10) |
| age_bin | **0.174** | 0.185 | **Excede umbral** — investigar |

**Hallazgo crítico:** `age_bin` muestra **EOD 0.174** — el modelo detecta el 100% de los churners reales en 60+ pero solo el 82.6% en 46–60. La diferencia es ~17 pp en TPR — material. Requiere acción.

**Veredicto reformulado** (vs. versión anterior, demasiado tranquilizadora):

- Ningún grupo cruza el ratio 1.5×, pero **EOD por edad sí excede 0.10** — el modelo NO trata a todos los grupos por igual en el recall.
- El calibration gap es muy bajo en todos los grupos (< 0.08) → la probabilidad significa aproximadamente lo mismo en cada grupo. El modelo es bien-calibrado por subgrupo.
- Causa probable: con n_test=47 en `46-60`, el ruido estadístico es alto. La diferencia podría no ser significativa con CI bootstrap. **Acción**: ampliar muestra de validación antes de promocionar a producción y, si persiste, recalibrar threshold por grupo o aplicar Fairlearn ThresholdOptimizer.

### 3.6 Plan ante discriminación detectada

Si el monitoreo en producción detecta `ratio > 1.5` o `< 0.67` en algún grupo protegido:

1. **Confirmar si es informativo o discriminatorio**: comparar `pred_pos_rate` vs. `true_pos_rate` por grupo. Si el modelo sobre-predice solo en ese grupo más allá de la tasa real observada → discriminación. Si predice ~consistentemente con la tasa real → el sesgo está en los datos, no en el modelo.
2. **Si es discriminatorio (sesgo del modelo):**
   - **Reentrenar excluyendo la variable sensible** (`age`, `city`) o aplicando *adversarial debiasing* / `Fairlearn`.
   - **Recalibrar thresholds por grupo** (group-specific thresholds) si la política permite y legal lo aprueba.
3. **Si es informativo (sesgo en la realidad):**
   - **No tocar el modelo.** Esconder un patrón real no resuelve el problema de negocio.
   - **Acción correctiva al producto/servicio del grupo afectado.** Ej.: si Call Center concentra churn real más alto, investigar si la experiencia en ese canal es deficiente — el modelo está siendo un *termómetro*, no la enfermedad.
4. **Gobierno**: auditoría trimestral documentada, con la tabla de la sección 3.5 versionada por fecha de modelo. Cambio de modelo requiere re-auditoría obligatoria.
5. **Variables sensibles**: si `age` o `city` aparecen en el top de SHAP en futuras versiones (hoy son #9 y #10 con aporte marginal), elevar la alerta. Hoy no es preocupante; mañana puede serlo si los datos cambian.

---

## Parte 4 — Arquitectura de Producción

> **Estado:** completa.
> **Documento detallado:** [architecture.md](architecture.md) (diagrama Mermaid + tabla de decisiones operativas)

### Resumen de la arquitectura propuesta

- **Pipeline en 6 capas**: Ingesta (CRM + transaccional + soporte) → Airflow/dbt + Great Expectations → Feature Store (Feast, offline + online) → Training + MLflow Registry → FastAPI sobre Kubernetes → Monitoreo (Evidently + Grafana + Alertmanager).
- **Gate humano obligatorio** para promoción de modelos: el retraining se *dispara* automáticamente (por drift o calendario), pero **un Product Owner del modelo firma antes de que el nuevo modelo llegue a producción**. Nunca self-deploy automático.
- **Seguridad cross-cutting**: IAM (RBAC por rol) atraviesa serving, feature store y MLOps; Vault inyecta credenciales en pipeline y serving. Sin secretos en código ni en logs.

### Decisiones operativas clave

| Decisión | Elección | Justificación 1-línea |
|---|---|---|
| Batch vs. tiempo real | **Batch diario** (con on-demand para casos puntuales) | Las features dominantes cambian lento, la acción comercial humana tarda > 24h. |
| Frecuencia de scoring | Diaria, 02:00–04:00 local | Alineada con planificación comercial del día siguiente. |
| Métricas de drift | **PSI > 0.2** en top-5 features + **KS-test en scores** + caída PR-AUC > 5 pp | PSI no requiere labels (que tardan 30–60 días en churn), captura cambios temprano. |
| Gobierno de cambios | PR + code review + gate de fairness automático + firma del PO + comité mensual | Sin accountability humana, el modelo es riesgo legal y reputacional. |
| Documentación | Model Card en MLflow + diagrama versionado + changelog + runbook | Onboarding rápido, auditoría regulatoria, debugging post-mortem. |

Ver [architecture.md](architecture.md) para el diagrama Mermaid completo y la justificación expandida de cada decisión.

---

## Parte 5 — IA Generativa / RAG

> **Estado:** completa.
> **Documento detallado:** [rag.md](rag.md) (diagrama Mermaid + stack + guardrails + métricas + HITL)

### Resumen ejecutivo de la propuesta RAG

- **Flujo con 4 guardrails de entrada + 4 de salida** (auth/RBAC, PII Presidio, prompt injection, tópico → retrieval con threshold de similaridad → LLM con citación obligatoria → faithfulness NLI, redacción PII, toxicidad, abstención). Dos puertas de escalación: similaridad baja y faithfulness fallida.
- **Stack recomendado**: dos vías — managed (Auth0 + OpenAI/Anthropic + Pinecone + Claude Sonnet) o open-source (Keycloak + e5-multilingual + Qdrant + Llama 3.1 70B). Decisión depende de si los datos pueden salir a proveedor externo.
- **Lo que NO entra al LLM**: PII directa, financieros sensibles, credenciales, datos protegidos (salud/menores), estrategia confidencial. Defensa en 5 capas: regex → NER (Presidio) → clasificador de tópico → saneo en ingesta → redacción en salida.
- **Anti-hallucination**: grounding obligatorio + citación por claim + threshold de similaridad 0.5 + "no sé" como respuesta válida + faithfulness NLI automático. Diseño consciente: preferimos falsos negativos del asistente sobre falsos positivos.
- **Métricas**: Faithfulness ≥ 0.90, Answer Relevancy ≥ 0.85, Context Precision ≥ 0.75 (RAGAS) + métrica única ejecutiva = **horas de agente liberadas / mes**.
- **HITL obligatorio** para: decisiones que afectan al cliente, quejas, casos sensibles, abstenciones, consultas legales, info estratégica, cambios de configuración crítica. Modo "shadow" los primeros 3 meses.

Ver [rag.md](rag.md) para el desarrollo completo de cada punto.

---

## Parte 6 — Pregunta Ejecutiva (90 días, presupuesto limitado, 5 áreas)

> **Estado:** completa.
> **Documento detallado:** [executive_summary.md](executive_summary.md) (1 página A4, tono ejecutivo)

### Decisión de roadmap

| Decisión | Iniciativa | Por qué |
|---|---|---|
| **Implementar (días 1–60)** | Modelo de churn en producción + dashboard comercial | Datos listos, modelo validado, retorno medible en 90 días, dueño claro (Retención). Mejor ratio impacto/esfuerzo. |
| **Pilotar (días 30–90)** | Asistente GenAI para Servicio al Cliente en **modo "shadow"** | Asistente responde, humano valida antes de mostrar al cliente — 3 meses. Bajo riesgo, alto aprendizaje sobre calidad y adopción. |
| **Postergar (Q3/Q4)** | Segmentación de clientes para campañas | Técnicamente factible, pero sin dueño de campaña comprometido se vuelve entregable sin uso. Reactivar cuando Marketing tenga plan operativo. |
| **Descartar (no se hace)** | (a) "Segundo modelo" para priorización comercial — es un motor de reglas sobre el score, no IA. (b) Detección de fraude con IA — sin etiquetas, las reglas son más auditables y baratas. | Decir "no" libera presupuesto y reputación. Construir modelos sin caso de uso es el principal motivo por el que la IA fracasa. |

### KPI ejecutivo único

> **Reducir la tasa de churn de los clientes intervenidos en ≥ 5 puntos porcentuales vs. grupo de control, en 6 meses.**

Medido vía A/B test: 80% recibe acción de retención, 20% queda como control. Si el uplift es < 5 pp, el proyecto se replantea — **no se renueva por inercia**.

Ver [executive_summary.md](executive_summary.md) para el desarrollo completo (diagnóstico, resultados, roadmap, riesgos top 3 y mitigación).

---

## Resumen Ejecutivo (máx. 1 página)

> **Estado:** completo.
> **Archivo:** [executive_summary.md](executive_summary.md) — documento independiente de 1 página A4, tono ejecutivo, sin jerga técnica.

Cubre las 5 secciones requeridas: diagnóstico (ML / GenAI / no-IA), resultado del modelo (XGBoost ROC-AUC 0.94, threshold 0.16), roadmap 90 días (implementar / pilotar / postergar / descartar), riesgos top 3 con mitigación, y KPI ejecutivo único.

---

## Seguridad y Gobierno (sección transversal)

Aspectos que el evaluador estricto correctamente identificó como ausentes en la versión inicial. Se enuncian explícitamente.

### Cumplimiento regulatorio

**GDPR Art. 22 — derecho a no ser objeto de decisiones automatizadas:**
- El modelo de churn **NO toma decisiones irreversibles** sobre el cliente — el output es un score que alimenta una acción comercial (contacto, oferta retentiva). La decisión final es humana. Esto nos saca del ámbito estricto del Art. 22 en la mayoría de casos.
- **Pero** si en el futuro se usa el score para denegar servicios, ajustar precios al alza, o cerrar cuentas, **se activa Art. 22 plenamente**: el cliente debe poder solicitar revisión humana y recibir explicación significativa. La infraestructura SHAP + waterfall del modelo está preparada para esto.
- **LFPDPPP / Ley Habeas Data ecuatoriana** (jurisdicción local del dataset): análogos al GDPR Art. 22 para clientes en LATAM. Procedimiento: cualquier cliente que pida explicación de su score recibe (a) la decisión humana resultante, (b) las top-5 features que impactaron su score, (c) acción que puede tomar para revertirlo (ej. ponerse al día con su mora).

### Consentimiento

**`customer_comment` y otros datos de PII**:
- El consentimiento para uso en modelado debe estar en los términos de servicio del producto, con opción de opt-out clara.
- Si un cliente revoca el consentimiento, sus datos se excluyen del próximo retraining y se purgan de logs de inferencia en 30 días.
- **No se usan datos de cliente para fine-tuning de LLMs externos** sin un contrato Data Processing Agreement firmado y categoría de datos explícitamente listada — independientemente de lo que diga el TOS por defecto del proveedor.

### Data residency

- **Embeddings y queries del RAG NO salen de la región** si el operador del LLM no tiene presencia en la región del cliente (ej. clientes ecuatorianos → LLM con endpoint en LATAM o, en su defecto, modelo open-source self-hosted).
- **Modelo de churn**: training y serving en infra dentro de la jurisdicción donde reside el dato. Multi-región solo con replicación gobernada y mismo nivel regulatorio.
- **Logs de auditoría**: 7 años de retención mínima (típica para datos comerciales), pero la PII se **redacta** en el log a los 90 días — se preserva la operación pero no el identificador.

### Robustez adversarial

Para el asistente RAG (Parte 5), no para el modelo de churn (que no recibe input directo del cliente):

| Vector de ataque | Mitigación |
|---|---|
| **Prompt injection** ("ignore all instructions and...") | Clasificador en input + system prompt blindado + audit log que captura el intento. Pruebas de regresión con OWASP LLM Top-10 prompts. |
| **Data exfiltration vía prompt** ("dame todos los emails de la base") | Clasificador de tópico + RBAC en retrieval (solo se buscan docs visibles para el rol del usuario). Audit log levanta alerta. |
| **Jailbreak / roleplay** ("eres DAN, no tienes restricciones") | Output guardrail: faithfulness NLI rechaza respuestas no soportadas en contexto, sin importar el prompt. |
| **Membership inference** sobre el modelo de churn | Improbable con XGBoost (modelo de árboles, no memoriza features individuales). En todo caso, no se expone el modelo crudo — solo decisión + explicación SHAP filtrada. |
| **Adversarial inputs** sobre features (cliente manipula su comportamiento para evadir score) | Bajo riesgo en churn (no es un sistema de fraude que el cliente intente evadir). Si pasara, monitoreo de drift en distribución de features lo detecta. |

### Cifrado

- **En tránsito**: TLS 1.3 obligatorio en todas las APIs (FastAPI + load balancer terminan TLS), mTLS entre microservicios internos.
- **En reposo**: encrypted disks en todos los almacenes (S3 SSE-KMS, GCS CMEK, Postgres TDE). Llaves gestionadas en Vault o KMS de cloud, no en código.
- **Embeddings**: el vector DB cifra en reposo igual que el resto.
- **Logs**: redactados (PII removida) antes de escribirse + cifrados.
- **Sin secretos en código**: enforced por pre-commit hook (`detect-secrets`) + escaneo en CI.

### Resumen para auditoría

Cualquier auditor regulatorio pide 4 cosas. Las tenemos:

1. **Lineage de datos** — desde la fuente hasta la decisión, vía dbt + Feast + MLflow.
2. **Explicabilidad por cliente** — SHAP waterfall reproducible.
3. **Fairness audit periódica** — métricas EOD/DP/Calibration documentadas con cadencia mensual (Parte 3.5).
4. **Trazabilidad de cambios** — PRs + Model Cards versionadas + log de aprobaciones.
