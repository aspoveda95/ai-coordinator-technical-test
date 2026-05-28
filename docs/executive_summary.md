# Resumen Ejecutivo — Plataforma de IA para Retención

**A:** Comité Ejecutivo  **De:** Coordinación de IA  **Fecha:** 2026-05-28

---

### 1. Diagnóstico

De los problemas que las áreas están pidiendo, **predecir abandono de clientes se resuelve con modelos predictivos clásicos** (la información ya existe, el patrón es estadístico, el modelo aprende). **El asistente para nuestros equipos internos se resuelve con IA generativa** (texto, conocimiento corporativo, consultas en lenguaje natural). **Priorizar a quién llamar primero y detectar fraude NO son problemas de IA hoy** — son reglas de negocio sobre el score predictivo en el primer caso, y validaciones automatizadas en el segundo: construir modelos ahí es ingeniería innecesaria y agrega riesgo legal.

### 2. Resultado del modelo de churn

El modelo final (XGBoost, entrenado sobre los datos provistos) **acierta correctamente al 94% de los clientes que abandonarán** (ROC-AUC 0.94 sobre datos de prueba). Definimos el umbral de decisión bajo el supuesto explícito de que **perder un cliente cuesta 5 veces más que contactar a uno que no se iba** — esto nos lleva a un umbral agresivo donde capturamos al 94% de los churners reales aceptando una tasa controlada de falsos positivos. El criterio puede recalibrarse cuando el negocio aporte el valor de vida del cliente y el costo real de cada campaña de retención.

### 3. Roadmap 90 días — 5 áreas pidiendo IA, presupuesto limitado

| Decisión | Iniciativa | Justificación ejecutiva |
|---|---|---|
| **Implementar (días 1–60)** | **Modelo de churn en producción + dashboard comercial** | Datos listos, modelo validado, retorno medible en 90 días, dueño claro (Retención). Es el caso con mejor ratio impacto / esfuerzo. |
| **Pilotar (días 30–90)** | **Asistente IA generativa para Servicio al Cliente, en modo "shadow"** | El asistente responde pero un humano valida antes de mostrar al cliente final, durante 3 meses. Bajo riesgo de daño, alto aprendizaje sobre calidad y adopción. |
| **Postergar (Q3/Q4)** | **Segmentación de clientes para campañas de Marketing** | Técnicamente factible, pero sin un dueño de campaña comprometido con accionar los segmentos se vuelve un entregable que nadie usa. Reactivar cuando Marketing tenga un plan operativo. |
| **Descartar (no se hace)** | **(a) Construir un "segundo modelo" para priorizar acciones comerciales.** Es un motor de reglas sobre el score, no IA. **(b) Detección de fraude con IA.** No tenemos casos etiquetados ni volumen suficiente; reglas de validación son más auditables, más baratas y legalmente más defendibles. | Decir "no" libera presupuesto y reputación. Construir modelos sin caso de uso real es el principal motivo por el que las iniciativas de IA fracasan. |

### 4. Riesgos top 3 y mitigación

1. **Las variables del modelo pueden estar contaminadas por información posterior al evento de churn.** Sin auditar cómo se calculan los días desde última compra y los días de mora, las métricas reportadas son un techo optimista. **Mitigación**: auditoría de fuente de datos en la semana 1; reentrenar si se confirma contaminación.
2. **El modelo es perfecto si nadie lo usa.** Riesgo de baja adopción del equipo comercial. **Mitigación**: co-diseño del dashboard con Retención desde el día 1, métrica de uso semanal, training presencial, y revisión mensual del comité con casos concretos.
3. **El asistente generativo puede filtrar datos sensibles al proveedor del modelo o inventar respuestas.** **Mitigación**: cuatro capas de filtrado de información personal antes y después del modelo, citación obligatoria de la fuente en cada respuesta, modo "shadow" los primeros 3 meses, y comité de auditoría mensual con Legal y Compliance.

### 5. KPI ejecutivo único

> **Reducir la tasa de churn de los clientes intervenidos en al menos 5 puntos porcentuales versus el grupo de control, en 6 meses.**

**Cómo se mide:** prueba A/B controlada. El modelo entrega el listado priorizado; **el 80% recibe acción de retención y el 20% queda como control sin intervención**. A los 6 meses comparamos la tasa de abandono real entre ambos grupos. Si el uplift es menor a 5 puntos, el proyecto se replantea; no se renueva por inercia.

**Lo que NO mediremos como éxito**: accuracy del modelo, número de alertas generadas, ni dashboards entregados. Esas son métricas de actividad, no de impacto.

---

*Más detalle técnico, decisiones de modelado, arquitectura de producción y diseño del asistente generativo en los entregables adjuntos: [solution.md](solution.md), [architecture.md](architecture.md), [rag.md](rag.md).*
