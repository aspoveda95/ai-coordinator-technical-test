"""EDA conciso para churn_dataset.csv — solo consola, sin gráficos."""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 120)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "churn_dataset.csv"
df = pd.read_csv(CSV_PATH)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# 1. Shape, dtypes y nulos
section("1. SHAPE, DTYPES Y NULOS POR COLUMNA")
print(f"Shape: {df.shape[0]} filas x {df.shape[1]} columnas\n")
overview = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "nulos_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    }
)
print(overview)

# 2. Distribución del target
section("2. DISTRIBUCIÓN DEL TARGET (churn)")
counts = df["churn"].value_counts(dropna=False)
pcts = df["churn"].value_counts(normalize=True, dropna=False) * 100
balance = pd.DataFrame({"count": counts, "pct": pcts.round(2)})
print(balance)
print(f"\nRatio churn=1 / churn=0: {counts.get(1, 0) / max(counts.get(0, 1), 1):.3f}")

# 3. Estadísticos descriptivos de numéricas relevantes
section("3. ESTADÍSTICOS DESCRIPTIVOS — NUMÉRICAS RELEVANTES")
numeric_relevant = [
    "age",
    "monthly_spend",
    "days_since_last_purchase",
    "support_tickets",
    "payment_delay_days",
    "digital_engagement_score",
    "promotion_usage",
]
numeric_relevant = [c for c in numeric_relevant if c in df.columns]
print("Global:")
print(df[numeric_relevant].describe().round(2))

print("\nPor churn (media):")
print(df.groupby("churn")[numeric_relevant].mean().round(2).T)

# 4. Top 10 ciudades + conteo por contract_type y channel
section("4. CATEGÓRICAS: CITY, CONTRACT_TYPE, CHANNEL")
print("Top 10 ciudades:")
top_cities = df["city"].value_counts(dropna=False).head(10)
print(top_cities)

print("\nConteo por contract_type:")
print(df["contract_type"].value_counts(dropna=False))

print("\nConteo por channel:")
print(df["channel"].value_counts(dropna=False))

print("\nTasa de churn por contract_type:")
print((df.groupby("contract_type")["churn"].mean() * 100).round(2).sort_values(ascending=False))

print("\nTasa de churn por channel:")
print((df.groupby("channel")["churn"].mean() * 100).round(2).sort_values(ascending=False))

# 5. Ejemplos de customer_comment
section("5. EJEMPLOS DE customer_comment")
print("--- 5 comentarios con churn=1 ---")
churn1 = df.loc[df["churn"] == 1, ["customer_id", "customer_comment"]].head(5)
for _, row in churn1.iterrows():
    print(f"[{row['customer_id']}] {row['customer_comment']}")

print("\n--- 5 comentarios con churn=0 ---")
churn0 = df.loc[df["churn"] == 0, ["customer_id", "customer_comment"]].head(5)
for _, row in churn0.iterrows():
    print(f"[{row['customer_id']}] {row['customer_comment']}")

print("\n" + "=" * 80)
print("EDA FINALIZADO")
print("=" * 80)
