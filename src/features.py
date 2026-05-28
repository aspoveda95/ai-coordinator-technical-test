"""Feature engineering para churn_dataset.csv.

Genera:
  - sentiment_score sobre customer_comment (VADER)
  - features derivadas con justificación de negocio
  - tratamiento de nulos (mediana / "Unknown")

Salida: features.parquet (sin encoding — listo para que el pipeline de modelado
aplique OHE/escalado dentro del fit).

NOTA TÉCNICA: VADER está entrenado sobre lexicon EN INGLÉS. Los comentarios de
este dataset están en español, por lo que la mayoría de scores serán ~0.0
(ruido). Se mantiene VADER porque fue el requerimiento explícito; recomendación
posterior es migrar a un modelo multilingüe (pysentimiento, distilbert-base-
multilingual o similar). Esto queda flagged como riesgo en solution.md.
"""

from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    # 2) sentiment_score (VADER) sobre customer_comment
    # ------------------------------------------------------------------
    # Compound score ∈ [-1, 1]. Hipótesis: comentarios negativos → más churn.
    # Limitación: VADER es solo inglés. Sobre comentarios en español el score
    # tenderá a 0 salvo palabras con cognados. Se incluye igual: si aporta señal
    # es bonus; si no, el modelo lo ignorará (feature importance baja).
    analyzer = SentimentIntensityAnalyzer()
    out["sentiment_score"] = out["customer_comment"].apply(
        lambda txt: analyzer.polarity_scores(txt)["compound"]
    )

    # ------------------------------------------------------------------
    # 3) Features derivadas — cada una tiene una hipótesis de negocio
    # ------------------------------------------------------------------

    # spend_per_day = gasto mensual / (días desde última compra + 1)
    # Hipótesis: mide la *intensidad reciente* del cliente. Un cliente que gasta
    # mucho pero hace mucho que no compra (alto monthly_spend, alto days_since)
    # da un valor BAJO → señal de desconexión activa. Un cliente con compras
    # recientes da valor ALTO → engagement vivo. El "+1" evita división por cero
    # cuando days_since_last_purchase == 0.
    out["spend_per_day"] = out["monthly_spend"] / (out["days_since_last_purchase"] + 1)

    # tickets_per_dollar = tickets de soporte / (gasto mensual + 1)
    # Hipótesis: cliente "caro de servir". Muchos tickets relativos a su gasto
    # implica fricción operativa alta + bajo retorno → candidato típico a churn
    # voluntario (cliente frustrado) o involuntario (la empresa lo deja ir).
    # El "+1" evita división por cero y suaviza el efecto en clientes de muy
    # bajo gasto donde el ratio se dispararía.
    out["tickets_per_dollar"] = out["support_tickets"] / (out["monthly_spend"] + 1)

    # is_delayed_payer = 1 si payment_delay_days > 15, else 0
    # Hipótesis: umbral de negocio. 15 días suele ser el corte donde una mora
    # pasa de "olvido administrativo" a "señal de intención de irse" o problema
    # financiero real. Convertir una continua en binaria captura el efecto
    # umbral que un modelo lineal pierde y RF/XGB no siempre encuentran solos
    # con n=1000. Se mantiene además la variable continua original.
    out["is_delayed_payer"] = (out["payment_delay_days"] > 15).astype(int)

    return out


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

    features_df = build_features(df)

    print(f"Resultado: {features_df.shape[0]} filas x {features_df.shape[1]} columnas")
    print("\nNuevas columnas:")
    new_cols = ["sentiment_score", "spend_per_day", "tickets_per_dollar", "is_delayed_payer"]
    print(features_df[new_cols].describe().round(4))

    print("\nsentiment_score por churn (media):")
    print(features_df.groupby("churn")["sentiment_score"].mean().round(4))

    print("\nis_delayed_payer por churn:")
    print(pd.crosstab(features_df["is_delayed_payer"], features_df["churn"], normalize="columns").round(3))

    features_df.to_parquet(OUT_PATH, index=False)
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
