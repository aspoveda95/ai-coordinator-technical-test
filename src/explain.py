"""Interpretabilidad — Parte 3.

Carga model.pkl + features.parquet, calcula SHAP sobre el XGBoost final,
genera plots y audita sesgo demográfico.

Salidas:
  - Top 10 features por |SHAP| en consola
  - shap_summary.png       : beeswarm global
  - shap_waterfall_pos.png : explicación de 1 cliente predicho churn=1
  - shap_waterfall_neg.png : explicación de 1 cliente predicho churn=0
  - Tabla de auditoría de sesgo en consola
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no abrir ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.pkl"
PARQUET_PATH = ROOT / "artifacts" / "features.parquet"
PLOTS_DIR = ROOT / "artifacts" / "plots"
BIAS_THRESHOLD_RATIO = 1.5  # >1.5x del promedio global → flag


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ----------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]

df = pd.read_parquet(PARQUET_PATH)
X = df[numeric_features + categorical_features]
y_true = df["churn"]

print(f"Modelo cargado. threshold persistido: {threshold}")
print(f"Dataset: {df.shape[0]} filas")

# ----------------------------------------------------------------------
# 2. Transformamos X para alimentar a SHAP (XGB ya recibe la matriz OHE)
# ----------------------------------------------------------------------
preprocessor = pipeline.named_steps["prep"]
classifier = pipeline.named_steps["clf"]

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()
# Limpiamos los prefijos "num__" / "cat__" del ColumnTransformer
feature_names = [name.split("__", 1)[-1] for name in feature_names]
X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

# ----------------------------------------------------------------------
# 3. SHAP TreeExplainer
# ----------------------------------------------------------------------
section("1. SHAP — IMPORTANCIA GLOBAL")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_df)  # shape: (n_samples, n_features)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print("\nTop 10 features por |SHAP| promedio:")
print(importance_df.head(10).round(4).to_string(index=False))

# ----------------------------------------------------------------------
# 4. Summary plot (beeswarm)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_df, show=False, max_display=15)
plt.tight_layout()
summary_path = PLOTS_DIR / "shap_summary.png"
plt.savefig(summary_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nGuardado: {summary_path.relative_to(ROOT)}")

# ----------------------------------------------------------------------
# 5. Waterfall: 1 cliente predicho churn=1, 1 predicho churn=0
# ----------------------------------------------------------------------
section("2. SHAP — EXPLICACIONES INDIVIDUALES (waterfall)")
y_proba = pipeline.predict_proba(X)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

# Tomamos un caso "alto riesgo" claro (proba más alta) y uno "bajo riesgo" claro
idx_high = int(np.argmax(y_proba))
idx_low = int(np.argmin(y_proba))

print(f"Cliente alto riesgo: idx={idx_high}, proba={y_proba[idx_high]:.4f}, "
      f"y_real={int(y_true.iloc[idx_high])}, y_pred={int(y_pred[idx_high])}")
print(f"Cliente bajo riesgo: idx={idx_low}, proba={y_proba[idx_low]:.4f}, "
      f"y_real={int(y_true.iloc[idx_low])}, y_pred={int(y_pred[idx_low])}")


def save_waterfall(sample_idx: int, out_path: Path, title: str) -> None:
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_df.iloc[sample_idx].values,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out_path.relative_to(ROOT)}")


save_waterfall(
    idx_high,
    PLOTS_DIR / "shap_waterfall_pos.png",
    f"Cliente predicho churn=1 (proba={y_proba[idx_high]:.3f})",
)
save_waterfall(
    idx_low,
    PLOTS_DIR / "shap_waterfall_neg.png",
    f"Cliente predicho churn=0 (proba={y_proba[idx_low]:.3f})",
)

# ----------------------------------------------------------------------
# 6. Auditoría de sesgo
# ----------------------------------------------------------------------
section("3. AUDITORÍA DE SESGO — TASA DE PREDICCIÓN POSITIVA POR GRUPO")
global_pos_rate = y_pred.mean()
print(f"Tasa de predicción positiva GLOBAL (threshold={threshold}): {global_pos_rate:.4f}\n")

audit_df = pd.DataFrame({
    "city": df["city"],
    "channel": df["channel"],
    "age": df["age"],
    "y_pred": y_pred,
    "y_true": y_true,
})

# Bins de edad por cuartiles aproximados (lo más explicable a negocio)
audit_df["age_bin"] = pd.cut(
    audit_df["age"],
    bins=[0, 30, 45, 60, 200],
    labels=["18-30", "31-45", "46-60", "60+"],
)


def audit_group(col: str) -> pd.DataFrame:
    grouped = audit_df.groupby(col, observed=True).agg(
        n=("y_pred", "size"),
        pos_rate_pred=("y_pred", "mean"),
        pos_rate_true=("y_true", "mean"),
    )
    grouped["ratio_vs_global"] = (grouped["pos_rate_pred"] / global_pos_rate).round(3)
    grouped["flag"] = grouped["ratio_vs_global"].apply(
        lambda r: "OVER" if r > BIAS_THRESHOLD_RATIO else ("UNDER" if r < 1 / BIAS_THRESHOLD_RATIO else "")
    )
    return grouped.round(4).sort_values("pos_rate_pred", ascending=False)


for col in ["city", "channel", "age_bin"]:
    print(f"\n--- Por {col} ---")
    print(audit_group(col))

# Resumen de flags
flags = []
for col in ["city", "channel", "age_bin"]:
    g = audit_group(col)
    for grupo, row in g.iterrows():
        if row["flag"]:
            flags.append({
                "variable": col,
                "grupo": grupo,
                "pos_rate_pred": row["pos_rate_pred"],
                "ratio_vs_global": row["ratio_vs_global"],
                "flag": row["flag"],
            })

section("4. RESUMEN DE FLAGS DE SESGO")
if flags:
    print(pd.DataFrame(flags).to_string(index=False))
    print(f"\n⚠️  {len(flags)} grupo(s) con desviación >{BIAS_THRESHOLD_RATIO}x del promedio.")
    print("   Recomendación: investigar si el sesgo es informativo (refleja diferencias reales")
    print("   en churn por grupo) o discriminatorio (el modelo penaliza a un grupo más allá")
    print("   de su tasa real observada).")
else:
    print(f"Sin grupos con desviación > {BIAS_THRESHOLD_RATIO}x ni < {1 / BIAS_THRESHOLD_RATIO:.2f}x.")

print("\n" + "=" * 80)
print("EXPLAIN FINALIZADO")
print("=" * 80)
