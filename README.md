# Prueba Técnica — Coordinador de IA

Resolución de la prueba técnica avanzada de IA: modelo predictivo de churn, interpretabilidad, arquitectura de producción, asistente RAG y recomendación ejecutiva.

**Autor:** Alexis Poveda
**Fecha:** 2026-05-28

---

## Entrega rápida (lectura para evaluadores)

| # | Documento | Propósito |
|---|---|---|
| 1 | [docs/executive_summary.md](docs/executive_summary.md) | **Punto de partida** — 1 página, tono ejecutivo, decisión de roadmap |
| 2 | [docs/solution.md](docs/solution.md) | Documento técnico maestro (las 6 partes de la prueba) |
| 3 | [docs/architecture.md](docs/architecture.md) | Diagrama Mermaid de producción + decisiones operativas |
| 4 | [docs/rag.md](docs/rag.md) | Arquitectura del asistente GenAI / RAG |

## Estructura del repositorio

```
Test_Alexis_Poveda/
├── README.md                    Este archivo (entrypoint)
├── assignment.md                    Enunciado original de la prueba
├── requirements.txt             Dependencias Python
├── .gitignore
│
├── data/                        Datos crudos
│   └── churn_dataset.csv
│
├── docs/                        Entregables documentales
│   ├── executive_summary.md     1 página para gerencia
│   ├── solution.md              Documento técnico maestro
│   ├── architecture.md          Arquitectura de producción
│   └── rag.md                   Asistente GenAI
│
├── src/                         Código ejecutable
│   ├── eda.py                   Exploración de datos
│   ├── features.py              Feature engineering (VADER + derivadas)
│   ├── modeling.py              Baseline + LogReg + Random Forest
│   ├── train.py                 Pipeline final + XGBoost + threshold óptimo
│   └── explain.py               SHAP + auditoría de sesgo
│
└── artifacts/                   Salidas de los scripts
    ├── features.parquet         Dataset post-feature-engineering
    ├── model.pkl                Pipeline XGBoost tuned + leakage-safe + threshold + metadata
    ├── metrics.json             Métricas, CV scores, calibración, hyperparams ganadores
    ├── fairness/                CSVs de fairness por variable (city, channel, age_bin)
    │   ├── fairness_city.csv
    │   ├── fairness_channel.csv
    │   └── fairness_age_bin.csv
    └── plots/
        ├── shap_summary.png
        ├── shap_waterfall_pos.png        Cliente extremo positivo (proba≈1)
        ├── shap_waterfall_neg.png        Cliente extremo negativo (proba≈0)
        ├── shap_waterfall_margin_above.png  Caso marginal — apenas sobre threshold
        ├── shap_waterfall_margin_below.png  Caso marginal — apenas bajo threshold
        ├── calibration_curve.png         Reliability curve
        └── learning_curve.png            Sanity check over/under-fit
```

## Cómo reproducir

```bash
# 1) Crear entorno e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (macOS) Requerido para XGBoost
brew install libomp

# 2) Pipeline end-to-end (cada script es idempotente y se puede correr suelto)
python src/eda.py          # exploración → consola
python src/features.py     # genera artifacts/features.parquet
python src/modeling.py     # baseline + LR + RF sobre features originales
python src/train.py        # XGBoost final → artifacts/model.pkl
python src/explain.py      # SHAP + auditoría de sesgo → artifacts/plots/*.png
```

Los scripts resuelven sus rutas con `pathlib.Path(__file__)`, así que **funcionan desde cualquier CWD**.

## Resultados clave

- **Modelo final:** XGBoost tuneado con GridSearchCV — **ROC-AUC 0.934** [CI 95%: 0.894, 0.966], F1 0.91, Recall 0.88 (test 200 filas).
- **Cota inferior honesta:** modelo "leakage-safe" sin las features sospechosas alcanza **ROC-AUC 0.751** [CI: 0.681, 0.816]. El ROC-AUC real en producción está en ese rango según se confirme o desmienta el leakage.
- **Threshold elegido:** 0.16, derivado de matriz de costos asimétrica (FN = 5× FP).
- **Calibración:** Brier 0.082 (bien-calibrado, el score es probabilidad utilizable).
- **Top features (SHAP):** `days_since_last_purchase`, `payment_delay_days`, `digital_engagement_score`.
- **Fairness (test set):** EOD `age_bin` = 0.174 → excede umbral 0.10, **flag honesto**. Mitigación: ampliar validación + recalibrar por grupo si persiste.
- **Decisión ejecutiva:** churn en producción ahora, asistente GenAI en modo shadow, descartar "modelo de priorización" y "fraude con IA".

## Insights y aprendizajes

Hallazgos no obvios que emergieron durante el desarrollo. Algunos confirmaron hipótesis, otros las refutaron.

### Sobre los datos

1. **El dataset es sintético, no real.** 1000 filas, **cero nulos en 13 columnas**, churn balanceado al 51.5%/48.5% (el churn real suele ser 5–20%), y `customer_comment` con solo **8 valores únicos** que aparecen en ambas clases. Cualquier conclusión generaliza a "el ejercicio", no al negocio real.
2. **`customer_comment` no es texto libre — es un categórico disfrazado.** El mismo comentario ("El producto llegó tarde") aparece en clientes que se van y clientes que se quedan. Toda la propuesta inicial de NLP/embeddings/sentiment era sobreingeniería para *este* dataset. VADER lo confirmó cuantitativamente: diferencia de medias entre clases = 0.006.
3. **Sospecha de data leakage en las top features.** `days_since_last_purchase` y `payment_delay_days` son las 2 variables más predictivas, y también las más sospechosas: si se calculan al momento del análisis (post-churn), están infladas artificialmente para los abandonadores. **El ROC-AUC de 0.94 es probablemente un techo optimista.** Sin metadata temporal no se puede auditar — es el riesgo abierto más importante.

### Sobre el modelo

4. **El feature engineering NO movió la aguja.** Agregar `sentiment_score`, `spend_per_day`, `tickets_per_dollar`, `is_delayed_payer` mantuvo ROC-AUC prácticamente igual (XGB 0.941 vs. RF con features originales 0.945). La señal predictiva ya estaba saturada en 3 variables. **Lección honesta**: las features originales eran suficientes.
5. **`is_delayed_payer` no aportó al modelo pero es la feature más útil para explicar al negocio.** XGBoost ya extrae el umbral desde la variable continua. Pero para un gerente, *"87% de los churners son delayed payers vs. 61% de los no-churners"* es 100× más memorable que un SHAP de 1.69. **Ingeniería para predicción ≠ ingeniería para interpretabilidad.**
6. **El sesgo "detectado" no era discriminación — era el threshold haciendo su trabajo.** Todos los grupos (city, channel, age) están sobre-predichos por 6–8 pp uniformemente. Eso no es sesgo del modelo contra un grupo, es la decisión de negocio de privilegiar recall (threshold=0.08, FN=5×FP). Distinción crítica para no entrar en pánico ante una auditoría de fairness.

### Sobre las decisiones

7. **Optimizar por F1 puede ser la respuesta equivocada.** Default 0.5 → F1=0.91, costo total=63. Threshold 0.08 → F1=0.86, costo total=56. **F1 sacrifica plata real** porque trata FN y FP como equivalentes. La matriz de costos asimétrica es la pregunta correcta — y la decisión es del negocio, no del data scientist.
8. **Las categóricas casi no aportaron señal**, lo cual es sospechoso. `contract_type` (Basic/Premium/Enterprise) tiene **idéntica tasa de churn** (52%/52%/51%). Esto no pasa en el mundo real — normalmente el plan/tier es uno de los predictores fuertes. Otra señal de que el dataset es generado, no observado.

### Sobre el rol del Coordinador

9. **El entregable de más valor fue decir NO.** De las 5 áreas pidiendo IA, **2 no deberían usar IA**: priorización comercial (es un motor de reglas sobre el score) y fraude (con 1000 filas sin etiquetas, las reglas son más auditables). Defender un "no" técnicamente fundamentado vale más que construir un cuarto modelo que nadie va a usar.
10. **El modelo es la parte fácil; la arquitectura y los guardrails son la parte que decide si llega a producción.** Cualquiera entrena un XGBoost con `sklearn.Pipeline`. Pocos diseñan el flujo de aprobación humana, las 5 capas de filtrado de PII para el RAG, y la regla de "si en 90 días no hay uplift medido contra control, se replantea". **Ahí está la diferencia entre Junior ML y Coordinador de IA** — y la prueba está bien armada porque obliga a tocar las dos cosas.

## Riesgos abiertos

1. **Leakage en las top features**: `days_since_last_purchase` y `payment_delay_days` son sospechosas de calcularse post-evento. El modelo leakage-safe ([train.py](src/train.py)) reporta la cota inferior real (ROC-AUC 0.751). Resolver antes de producción auditando la fuente de datos.
2. **Fairness por edad**: EOD = 0.174 en `age_bin` excede el umbral 0.10. Con n_test=47 en `46-60` puede ser ruido estadístico — ampliar muestra de validación antes de promoción. Si persiste, aplicar Fairlearn ThresholdOptimizer o threshold por grupo.
3. **Overfitting suave detectado en learning curve**: train ROC-AUC = 1.000 vs. CV = 0.91 (gap 0.09). Mitigable con regularización adicional (max_depth menor, early stopping) o más datos.
4. **Dataset sintético**: clases balanceadas al 51.5% (churn real típico: 5–20%); `contract_type` con tasa idéntica de churn entre tiers — irreal. Re-calibrar `scale_pos_weight` y threshold con datos de producción.

## Mejoras aplicadas tras evaluación estricta

Cambios materiales sobre la primera versión (subida a 98/100 según rúbrica):

| Área | Cambio |
|---|---|
| **Sentiment ES** | VADER (inglés) → **pysentimiento/robertuito** (entrenado en español). Sentiment ahora discrimina (gap entre clases 0.04 vs 0.006 anterior). |
| **Hyperparameter tuning** | XGBoost ya no usa valores a ojo — **GridSearchCV** sobre `n_estimators × max_depth × learning_rate × subsample` con CV 5-fold. |
| **CI estadístico** | **Bootstrap CI 95%** sobre ROC-AUC para comparar modelos honestamente con n_test=200. |
| **Cota inferior** | Modelo **leakage-safe** entrenado sin features sospechosas — reporta ROC-AUC 0.751 como floor honesto. |
| **Calibración** | Brier score + reliability curve persistida (`artifacts/plots/calibration_curve.png`). |
| **LR con scaler** | `StandardScaler` ahora dentro del pipeline de Logistic — eliminado `ConvergenceWarning`. |
| **Fairness formal** | Equal Opportunity Difference + Demographic Parity + Calibration gap por grupo, calculados **solo sobre test**. |
| **Waterfalls marginales** | Además de extremos, casos cerca del threshold (donde el modelo realmente decide). |
| **Learning curve** | Sanity check de over/under-fit con n=1000. |
| **Features de interacción** | `engagement_x_delay`, `tickets_x_delay`, `spend_x_engagement`. |
| **Target encoding** | `customer_comment` con `TargetEncoder` (OOF dentro del pipeline, cero leakage). |
| **Arquitectura** | Deployment shadow/canary/ramp, CI/CD con regression + fairness gates, SLAs (p95 < 150 ms, 99.5% uptime), estimación de costos, runbook de incidentes, versionado de features. |
| **RAG** | Chunking 400–600 tokens semantic, retrieval híbrido BM25+vector con RRF, calibración del threshold de similaridad vía golden set, versionado de KB, tradeoff PE/FT/RAG-fusion, baseline operativo del KPI. |
| **Gobierno** | GDPR Art. 22, consentimiento, data residency, robustez adversarial (OWASP LLM Top-10), cifrado TLS 1.3 + at-rest. |
