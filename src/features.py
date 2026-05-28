"""Feature engineering para churn_dataset.csv.

Genera:
  - sentiment_score sobre customer_comment usando pysentimiento (modelo
    robertuito-sentiment-analysis, entrenado en español)
  - features derivadas con justificación de negocio + features de interacción
  - tratamiento de nulos (mediana / "Unknown")

Salida: features.parquet (sin encoding — listo para que el pipeline de modelado
aplique OHE / escalado / target-encoding dentro del fit).

DECISIÓN: VADER (versión anterior) se desechó porque su lexicon es solo inglés
y los comentarios están en español. pysentimiento usa un RoBERTuito fine-tuned
en español y devuelve P(POS|NEG|NEU); convertimos a un compound similar a VADER
con score = P(POS) - P(NEG) ∈ [-1, 1].

RIESGO DE LEAKAGE: spend_per_day y is_delayed_payer usan variables que se
sospecha calculadas post-evento (days_since_last_purchase, payment_delay_days).
Estas features derivadas HEREDAN y AMPLIFICAN ese riesgo. Se mantienen porque
si la fuente está limpia, aportan; pero el modelo "leakage-safe" en train.py
las excluye para reportar la cota inferior real.
"""

from pathlib import Path

import pandas as pd
from pysentimiento import create_analyzer

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "churn_dataset.csv"
OUT_PATH = ROOT / "artifacts" / "features.parquet"

NUMERIC_COLS = [
    "age",
    "monthly_spend",
    "days_since_last_purchase",
    "support_tickets",
    "payment_delay_days",
    "digital_engagement_score",
    "promotion_usage",
]
CATEGORICAL_COLS = ["city", "contract_type", "channel", "customer_comment"]


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula sentiment en español con pysentimiento.

    Cachea por valor único de customer_comment porque el dataset solo tiene
    8 valores únicos en 1000 filas — el modelo se llama 8 veces, no 1000.
    """
    out = df.copy()
    analyzer = create_analyzer(task="sentiment", lang="es")
    unique_comments = out["customer_comment"].dropna().unique()
    cache: dict[str, float] = {}
    for txt in unique_comments:
        probas = analyzer.predict(txt).probas
        # Compound similar a VADER: P(POS) - P(NEG) ∈ [-1, 1]
        cache[txt] = float(probas.get("POS", 0.0) - probas.get("NEG", 0.0))
    out["sentiment_score"] = out["customer_comment"].map(cache).fillna(0.0)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ------------------------------------------------------------------
    # 1) Imputación de nulos
    # ------------------------------------------------------------------
    # Numéricas con mediana: robusta a outliers vs. la media. Crítico cuando
    # support_tickets o payment_delay_days tienen colas largas que distorsionarían
    # la media. En este dataset no hay nulos, pero la lógica queda para
    # generalizar a nuevos snapshots.
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())

    # Categóricas con "Unknown": preserva la información de "dato faltante" como
    # categoría propia en lugar de imputar a la moda (que sesgaría hacia el
    # segmento mayoritario). El modelo decide si "Unknown" tiene señal predictiva.
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    # ------------------------------------------------------------------
    # 2) sentiment_score con pysentimiento (RoBERTuito ES)
    # ------------------------------------------------------------------
    out = add_sentiment(out)

    # ------------------------------------------------------------------
    # 3) Features derivadas — hipótesis de negocio explícita
    # ------------------------------------------------------------------

    # spend_per_day = gasto mensual / (días desde última compra + 1)
    # Hipótesis: intensidad reciente del cliente. Cliente que gasta mucho pero
    # hace tiempo no compra → valor BAJO (desconexión activa); compras recientes
    # → valor ALTO (engagement vivo). "+1" evita división por cero.
    # RIESGO: hereda leakage de days_since_last_purchase si esa variable se
    # calcula post-evento.
    out["spend_per_day"] = out["monthly_spend"] / (out["days_since_last_purchase"] + 1)

    # tickets_per_dollar = tickets de soporte / (gasto mensual + 1)
    # Hipótesis: cliente "caro de servir". Muchos tickets relativos al gasto
    # implica fricción operativa + bajo retorno → candidato a churn.
    out["tickets_per_dollar"] = out["support_tickets"] / (out["monthly_spend"] + 1)

    # is_delayed_payer = 1 si payment_delay_days > 15, else 0
    # Hipótesis: umbral de negocio donde la mora pasa de "olvido administrativo"
    # a señal de problema real. Captura efecto umbral que el modelo lineal pierde.
    # Se mantiene además la variable continua original.
    # RIESGO: hereda leakage de payment_delay_days si se calcula post-evento.
    out["is_delayed_payer"] = (out["payment_delay_days"] > 15).astype(int)

    # ------------------------------------------------------------------
    # 4) Features de interacción
    # ------------------------------------------------------------------
    # Combinaciones que el modelo de árbol puede capturar implícitamente, pero
    # explicitarlas: (a) ayuda a la interpretabilidad SHAP, (b) acelera la
    # convergencia con dataset pequeño, (c) Logistic Regression las necesita
    # porque no captura interacciones por sí sola.

    # engagement × delay: dos señales fuertes que típicamente coinciden en
    # churners reales (bajo engagement Y mora alta = casi seguro abandono).
    out["engagement_x_delay"] = (
        out["digital_engagement_score"] * out["payment_delay_days"]
    )

    # tickets × delay: cliente con fricción operativa Y problemas financieros —
    # el peor combo, distinto a alta mora sola.
    out["tickets_x_delay"] = out["support_tickets"] * out["payment_delay_days"]

    # spend × engagement: distingue cliente "premium dormido" (alto spend, bajo
    # engagement) de cliente "engaged económico" (bajo spend, alto engagement).
    # Útil para uplift modeling futuro.
    out["spend_x_engagement"] = out["monthly_spend"] * out["digital_engagement_score"]

    return out


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

    features_df = build_features(df)

    print(f"Resultado: {features_df.shape[0]} filas x {features_df.shape[1]} columnas")
    print("\nNuevas columnas:")
    new_cols = [
        "sentiment_score",
        "spend_per_day",
        "tickets_per_dollar",
        "is_delayed_payer",
        "engagement_x_delay",
        "tickets_x_delay",
        "spend_x_engagement",
    ]
    print(features_df[new_cols].describe().round(4))

    print("\nsentiment_score por churn (media):")
    print(features_df.groupby("churn")["sentiment_score"].mean().round(4))

    print("\nis_delayed_payer por churn:")
    print(
        pd.crosstab(
            features_df["is_delayed_payer"], features_df["churn"], normalize="columns"
        ).round(3)
    )

    features_df.to_parquet(OUT_PATH, index=False)
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
