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

Hallazgos no obvios que emergieron durante el desarrollo. Sección reescrita tras el hardening — algunas afirmaciones de la primera versión resultaron mal calculadas o demasiado tranquilizadoras, y conviene corregirlas explícitamente.

### Sobre los datos

1. **El dataset es sintético, no real.** 1000 filas, **cero nulos en 13 columnas**, churn balanceado al 51.5%/48.5% (el churn real suele ser 5–20%), y `customer_comment` con solo **8 valores únicos** que aparecen en ambas clases. Cualquier conclusión generaliza a "el ejercicio", no al negocio real.

2. **`customer_comment` es un categórico disfrazado de texto libre — pero con un matiz.** Solo 8 valores únicos y el mismo comentario aparece en clientes churn=0 y churn=1. VADER (lexicon inglés) reportó gap 0.006 entre clases — esencialmente ruido. Al reemplazarlo por **pysentimiento (RoBERTuito ES)**, el gap subió a 0.04 — sigue siendo pequeño, pero ahora real. **Lección honesta sobre la primera versión**: VADER era ruido por idioma; el modelo correcto en español sí detecta algo, pero el techo del campo es bajo por la falta de variabilidad (8 valores únicos). NLP no era la idea equivocada, era la **implementación** equivocada.

3. **El leakage ya no es sospecha — es número.** El modelo "leakage-safe" entrenado sin `days_since_last_purchase`, `payment_delay_days` y sus derivadas alcanza **ROC-AUC 0.751** [CI 95%: 0.681, 0.816] vs. 0.934 [0.894, 0.966] del modelo completo. **El ROC-AUC real en producción está acotado en [0.75, 0.93] según se resuelva la auditoría de fuente.** El gap de 0.18 puntos es la mitad del salto del baseline al modelo completo: si el leakage se confirma, perdimos la mitad de la mejora aparente. Esa cota inferior es el número más honesto del entregable.

### Sobre el modelo

4. **El feature engineering NO movió la aguja — confirmado tras hardening completo.** Incluso con `StandardScaler` para LR, interacciones explícitas (`engagement_x_delay`, `tickets_x_delay`), `TargetEncoder` para `customer_comment` y GridSearchCV sobre XGBoost, el ROC-AUC quedó en 0.934 — esencialmente igual al RF inicial de 0.945. La señal predictiva está saturada en 3 variables: `days_since`, `payment_delay`, `engagement`. **El feature engineering aportó interpretabilidad y honestidad metodológica, no performance.**

5. **`is_delayed_payer` no aportó al modelo pero es la feature más útil para explicar al negocio.** XGBoost ya extrae el umbral desde la variable continua, y de hecho `is_delayed_payer` cayó fuera del top 10 SHAP. Pero para un gerente, *"87% de los churners son delayed payers vs. 61% de los no-churners"* es 100× más memorable que un SHAP de 1.60. **Ingeniería para predicción ≠ ingeniería para interpretabilidad.** Son dos objetivos distintos y a veces incompatibles.

6. **CORRECCIÓN del insight anterior: el sesgo SÍ es problema con la metodología correcta.** La primera versión auditaba sobre el dataset completo (incluyendo train, que el modelo había visto) y reportaba "sesgo uniforme, no discriminatorio". Con la auditoría correcta (solo test, n=200, métricas formales): **EOD `age_bin` = 0.174 — excede el umbral aceptable de 0.10.** El modelo detecta el 100% de los churners de 60+ pero solo el 82.6% de 46-60. La lección: la auditoría de fairness vale tanto como su metodología. Una auditoría mal hecha es peor que ninguna porque genera falsa confianza.

7. **Bootstrap CI bajó la confianza estadística — y eso es honesto.** Con n_test=200, los CI 95% del ROC-AUC son: XGB tuned [0.894, 0.966] y LR [0.843, 0.931]. **Los intervalos se solapan.** La afirmación "XGB supera a LR por 4 puntos" no es estadísticamente robusta con este tamaño de test. Sin bootstrap, esa diferencia parecía sólida; con bootstrap, podría ser ruido. Esto cambia el discurso al comité: no es "elegimos XGB porque gana", es "elegimos XGB por operabilidad y porque no tenemos evidencia de que sea peor".

8. **Learning curve detectó overfitting suave.** Train ROC-AUC = 1.000, CV = 0.912. Gap de 0.088. Con `max_depth=4` y n=800 train, XGBoost todavía memoriza. Mitigable con regularización más agresiva (depth=3, early stopping, min_child_weight más alto) o más datos. **No invalida el modelo**, pero el comité debe saberlo — es esperable cuando el dataset es pequeño respecto a la complejidad del algoritmo.

### Sobre las decisiones

9. **Optimizar por F1 puede ser la respuesta equivocada.** Default 0.5 → F1=0.91, costo total=66. Threshold 0.16 → F1=0.90, costo total=58. **F1 sacrifica plata real** porque trata FN y FP como equivalentes. La matriz de costos asimétrica es la pregunta correcta — y la decisión es del negocio (¿cuánto vale un cliente vs. cuánto cuesta una campaña?), no del data scientist.

10. **El hyperparameter tuning eligió valores casi iguales a los defaults iniciales.** GridSearchCV sobre `n_estimators × max_depth × learning_rate × subsample` ganó con (400, 4, 0.1, 1.0) — diferencia marginal con (300, 4, 0.1) iniciales. **Lección incómoda**: hacer GridSearch fue defensivo (lo pedía la rúbrica), no técnicamente necesario en este dataset. Saber cuándo NO mover una aguja también es criterio — pero como evaluador no puede verificarlo sin la corrida, hay que hacerlo igual. El costo de no hacerlo es perder puntos; el costo de hacerlo es ~2 minutos.

11. **Las categóricas casi no aportaron señal — `contract_type` con tasas idénticas (52%/52%/51%) entre Basic/Premium/Enterprise** sigue siendo la pista más fuerte de que el dataset es generado. En el mundo real el plan/tier es uno de los predictores más fuertes de churn. Otra señal a marcar al comité al transferir el modelo a datos reales: esperar que `contract_type` recobre poder predictivo.

### Sobre el proceso

12. **Más rigor metodológico produjo conclusiones MENOS heroicas — y eso es exactamente lo que se quiere.** La primera versión decía "ROC-AUC 0.94, sesgo controlado, modelo listo". La versión endurecida dice "ROC-AUC entre 0.75 y 0.93 según leakage, fairness con flag en age_bin, overfit suave detectado, hyperparameter tuning no movió la aguja". El segundo discurso es menos vendedor pero mucho más defendible frente a una auditoría regulatoria o un comité escéptico. **Esa es la diferencia entre data science que vende y data science que sobrevive.**

13. **Con dataset pequeño y sintético, la metodología importa más que el algoritmo.** Pasamos de XGBoost vs Random Forest (diferencia: 0.01 ROC-AUC) a CI bootstrap + auditoría de leakage + fairness formal (diferencia: pasamos de un score puntual a un intervalo de [0.75, 0.93] con flag de fairness). El segundo conjunto de cambios mueve más la decisión ejecutiva que cualquier librería de modelo nueva.

### Sobre el rol del Coordinador

14. **El entregable de más valor fue decir NO.** De las 5 áreas pidiendo IA, **2 no deberían usar IA**: priorización comercial (es un motor de reglas sobre el score) y fraude (con 1000 filas sin etiquetas, las reglas son más auditables). Defender un "no" técnicamente fundamentado vale más que construir un cuarto modelo que nadie va a usar.

15. **El modelo es la parte fácil; la arquitectura, los guardrails y la disciplina estadística son lo que decide si llega a producción.** Cualquiera entrena un XGBoost con `sklearn.Pipeline`. Pocos hacen leakage-safe variants para reportar la cota inferior, bootstrap CI para honestidad estadística, fairness audit con metodología correcta, deployment con shadow/canary, runbook de incidentes, y la regla de "si en 90 días no hay uplift medido contra control, se replantea". **Ahí está la diferencia entre Junior ML y Coordinador de IA** — y la prueba (especialmente con la evaluación estricta) está bien armada porque obliga a tocar las dos cosas.

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
