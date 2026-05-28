# Prueba Técnica Avanzada – Coordinador de Inteligencia Artificial

**Objetivo:** Evaluar conocimientos avanzados en IA, ML, IA Generativa, Arquitectura de Producción y criterio ejecutivo aplicado al negocio.

---

## Parte 1 – Análisis Técnico del Problema

La empresa desea implementar una plataforma inteligente para:
1. Predecir clientes con riesgo de abandono.
2. Priorizar acciones comerciales.
3. Crear un asistente interno basado en IA generativa.

**Preguntas:**
- ¿Qué problemas de IA identifica?
- ¿Qué partes resolvería con ML clásico y cuáles con IA generativa?
- ¿Qué NO resolvería con IA?
- ¿Qué riesgos existen en datos, sesgos, seguridad y adopción?
- ¿Qué métricas usaría para medir éxito técnico y de negocio?

---

## Parte 2 – Modelo Predictivo Avanzado

Con el dataset entregado, construya un modelo para predecir abandono.

**Debe incluir:**
- Exploración de datos
- Tratamiento de valores nulos
- Encoding de variables categóricas
- Feature engineering
- División train/test estratificada
- Al menos 3 modelos: Baseline simple, Logistic Regression, Random Forest o XGBoost
- Métricas: Recall, Precision, F1, ROC-AUC, Matriz de confusión

**Preguntas:**
- ¿Qué modelo elegiría y por qué?
- ¿Qué trade-off existe entre precision y recall?
- ¿Qué costo tiene equivocarse?
- ¿Cómo definiría el threshold?
- ¿Cómo evitaría data leakage?

---

## Parte 3 – Interpretabilidad y Explicabilidad

- Variables más importantes
- Factores que aumentan el riesgo
- Factores que reducen el riesgo
- Cómo lo explicaría a un gerente no técnico
- Qué haría si el modelo discrimina por ciudad, edad o canal

---

## Parte 4 – Arquitectura de Producción

**Debe incluir:** Ingesta, data pipeline, feature store, API del modelo, dashboard de monitoreo, drift, retraining, seguridad, versionado.

**Preguntas:**
- ¿Batch o tiempo real?
- ¿Cada cuánto se recalcula el score?
- ¿Cómo se monitorea el drift?
- ¿Quién aprueba cambios al modelo?
- ¿Cómo se documenta el modelo?

---

## Parte 5 – IA Generativa / RAG

Asistente interno basado en GenAI. Explicar: arquitectura RAG, base de conocimiento, vector DB, guardrails, hallucinations, protección de datos sensibles, human-in-the-loop.

**Preguntas:**
- ¿Qué información NO debería entrar al LLM?
- ¿Cómo evitaría respuestas inventadas?
- ¿Cómo auditaría respuestas?
- ¿Cómo mediría calidad?
- ¿Qué haría ante solicitudes confidenciales?

---

## Parte 6 – Pregunta Ejecutiva

Tienes 90 días, presupuesto limitado y 5 áreas pidiendo IA. ¿Qué implementas primero, qué descartas y cómo lo justificas ante gerencia?

---

## Rúbrica

| Área | Puntaje |
|---|---|
| Fundamentos ML | 15 |
| Feature Engineering | 15 |
| Evaluación de Modelos | 15 |
| Interpretabilidad | 10 |
| Arquitectura de Producción | 15 |
| IA Generativa / RAG | 15 |
| Seguridad y Gobierno | 10 |
| Criterio Ejecutivo | 5 |

## Entregables
- Notebook o script Python
- Documento de decisiones técnicas
- Diagrama de arquitectura
- Resumen ejecutivo (máximo 1 página)

---

## Dataset (`Dataset_Prueba_Avanzada_IA.csv`)

**1000 filas, 13 columnas:**
`customer_id, age, city, monthly_spend, days_since_last_purchase, support_tickets, payment_delay_days, digital_engagement_score, promotion_usage, contract_type, channel, customer_comment, churn`

- `churn` es el target binario.
- `customer_comment` es texto libre → oportunidad para mostrar NLP/sentiment como feature.
- `city`, `contract_type`, `channel` son categóricas.
