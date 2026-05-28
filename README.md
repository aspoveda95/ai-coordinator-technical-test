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

## Riesgos abiertos

1. `days_since_last_purchase` y `payment_delay_days` son sospechosas de leakage post-evento — requieren auditoría de fuente.
2. VADER no funciona en español; el `sentiment_score` actual es ruido. Recomendación: `pysentimiento` o BERT-es si se quiere texto.
3. Dataset balanceado al 51.5% no representa churn real (típico 5–20%) — recalibrar `scale_pos_weight` y threshold con datos de producción.
