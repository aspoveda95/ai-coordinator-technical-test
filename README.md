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
    ├── model.pkl                Pipeline XGBoost + threshold + metadata
    └── plots/
        ├── shap_summary.png
        ├── shap_waterfall_pos.png
        └── shap_waterfall_neg.png
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

- **Modelo final:** XGBoost — **ROC-AUC 0.94**, F1 0.91, Recall 0.89 (test 200 filas).
- **Threshold elegido:** 0.08, derivado de matriz de costos asimétrica (FN = 5× FP).
- **Top features (SHAP):** `days_since_last_purchase`, `payment_delay_days`, `digital_engagement_score`.
- **Sesgo:** ningún grupo cruza 1.5× del promedio. Call Center (1.16×) bajo monitoreo.
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

1. `days_since_last_purchase` y `payment_delay_days` son sospechosas de leakage post-evento — requieren auditoría de fuente.
2. VADER no funciona en español; el `sentiment_score` actual es ruido. Recomendación: `pysentimiento` o BERT-es si se quiere texto.
3. Dataset balanceado al 51.5% no representa churn real (típico 5–20%) — recalibrar `scale_pos_weight` y threshold con datos de producción.
