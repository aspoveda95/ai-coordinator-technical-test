"""Interpretabilidad — Parte 3 (versión endurecida).

Cambios respecto a la versión inicial, motivados por la evaluación estricta:
  1. La auditoría de sesgo ahora se hace SOLO en test (antes incluía datos que
     el modelo había visto durante entrenamiento — metodológicamente incorrecto).
  2. Métricas formales de fairness por grupo:
       - Equal Opportunity (TPR por grupo) → ¿el modelo detecta churners por
         igual en cada grupo?
       - Demographic Parity (positive rate por grupo) → ¿predice positivo a
         la misma proporción en cada grupo?
       - Calibration within groups (true rate vs. pred rate) → ¿la probabilidad
         significa lo mismo en cada grupo?
  3. Waterfalls de casos MARGINALES (clientes cerca del threshold) además de los
     extremos. Es donde el modelo realmente decide.
  4. Learning curve para detectar under/overfit con n=1000.

Salidas:
  - Top 10 features por |SHAP| en consola
  - shap_summary.png         : beeswarm global
  - shap_waterfall_pos.png   : cliente alto riesgo (proba ≈ 1)
  - shap_waterfall_neg.png   : cliente bajo riesgo (proba ≈ 0)
  - shap_waterfall_margin_*.png : 2 clientes cerca del threshold
  - learning_curve.png       : sanity check de over/under-fit
  - fairness_*.csv           : métricas formales por grupo
  - Tabla de auditoría de sesgo en consola (solo test)
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import learning_curve, train_test_split

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.pkl"
PARQUET_PATH = ROOT / "artifacts" / "features.parquet"
PLOTS_DIR = ROOT / "artifacts" / "plots"
FAIRNESS_DIR = ROOT / "artifacts" / "fairness"
FAIRNESS_DIR.mkdir(exist_ok=True)
BIAS_THRESHOLD_RATIO = 1.5
RANDOM_STATE = 42


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ----------------------------------------------------------------------
# 1. Carga del modelo + features
# ----------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]

df = pd.read_parquet(PARQUET_PATH)
X_all = df[numeric_features + categorical_features]
y_all = df["churn"]

# Reproducimos el mismo split que train.py — fairness audit se hace SOLO sobre test
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
)
test_idx = X_test.index
print(f"Modelo cargado. threshold persistido: {threshold}")
print(f"Test set: {len(X_test)} filas (auditoría de sesgo aquí, no en full data)")

# ----------------------------------------------------------------------
# 2. SHAP global (sobre full set para inspección visual; la métrica global
#    no depende del split y SHAP captura el comportamiento del modelo, no
#    del dato)
# ----------------------------------------------------------------------
section("1. SHAP — IMPORTANCIA GLOBAL")
preprocessor = pipeline.named_steps["prep"]
classifier = pipeline.named_steps["clf"]

X_full_transformed = preprocessor.transform(X_all)
feature_names = preprocessor.get_feature_names_out()
feature_names = [n.split("__", 1)[-1] for n in feature_names]
if hasattr(X_full_transformed, "toarray"):
    X_full_transformed = X_full_transformed.toarray()
X_full_df = pd.DataFrame(X_full_transformed, columns=feature_names, index=X_all.index)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_full_df)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print("\nTop 10 features por |SHAP|:")
print(importance_df.head(10).round(4).to_string(index=False))

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_full_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nGuardado: artifacts/plots/shap_summary.png")

# ----------------------------------------------------------------------
# 3. Waterfalls: extremos + marginales
# ----------------------------------------------------------------------
section("2. SHAP — WATERFALLS (extremos + marginales cerca del threshold)")
y_proba_full = pipeline.predict_proba(X_all)[:, 1]
y_pred_full = (y_proba_full >= threshold).astype(int)

idx_high = int(np.argmax(y_proba_full))
idx_low = int(np.argmin(y_proba_full))

# Casos marginales: ±0.03 alrededor del threshold (donde el modelo realmente decide)
distance = np.abs(y_proba_full - threshold)
margin_indices = np.argsort(distance)[:10]
margin_above = next(int(i) for i in margin_indices if y_proba_full[i] >= threshold)
margin_below = next(int(i) for i in margin_indices if y_proba_full[i] < threshold)

cases = [
    (idx_high, "shap_waterfall_pos.png", f"Alto riesgo (proba={y_proba_full[idx_high]:.3f})"),
    (idx_low, "shap_waterfall_neg.png", f"Bajo riesgo (proba={y_proba_full[idx_low]:.3f})"),
    (
        margin_above,
        "shap_waterfall_margin_above.png",
        f"MARGINAL — apenas sobre threshold (proba={y_proba_full[margin_above]:.3f})",
    ),
    (
        margin_below,
        "shap_waterfall_margin_below.png",
        f"MARGINAL — apenas bajo threshold (proba={y_proba_full[margin_below]:.3f})",
    ),
]

for idx, fname, title in cases:
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_full_df.iloc[idx].values,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: artifacts/plots/{fname}  ({title})")

# ----------------------------------------------------------------------
# 4. Auditoría de sesgo (SOLO TEST) + métricas formales de fairness
# ----------------------------------------------------------------------
section("3. AUDITORÍA DE SESGO — SOLO SOBRE TEST (200 filas)")
y_proba_test = pipeline.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= threshold).astype(int)

audit_df = pd.DataFrame({
    "city": df.loc[test_idx, "city"].values,
    "channel": df.loc[test_idx, "channel"].values,
    "age": df.loc[test_idx, "age"].values,
    "y_true": y_test.values,
    "y_pred": y_pred_test,
    "y_proba": y_proba_test,
})
audit_df["age_bin"] = pd.cut(
    audit_df["age"], bins=[0, 30, 45, 60, 200], labels=["18-30", "31-45", "46-60", "60+"]
)

global_pos_rate = audit_df["y_pred"].mean()
print(f"Tasa de predicción positiva GLOBAL (test): {global_pos_rate:.4f}\n")


def fairness_table(col: str) -> pd.DataFrame:
    """Métricas formales de fairness por grupo.

    Columnas devueltas:
      - n: tamaño del grupo
      - pos_rate_pred (Demographic Parity): P(ŷ=1 | grupo)
      - tpr / recall (Equal Opportunity): P(ŷ=1 | y=1, grupo)
      - fpr: P(ŷ=1 | y=0, grupo)
      - pos_rate_true: tasa real de positivos en el grupo
      - calibration_gap: |pos_rate_pred − pos_rate_true|
                         (si la prob significa lo mismo en cada grupo, → 0)
      - ratio_vs_global: pos_rate_pred / global, marcado si > 1.5x o < 0.67x
    """
    rows = []
    for grupo, sub in audit_df.groupby(col, observed=True):
        y_t = sub["y_true"].to_numpy()
        y_p = sub["y_pred"].to_numpy()
        n = len(sub)
        n_pos = int(y_t.sum())
        n_neg = n - n_pos
        tpr = (y_p[y_t == 1].mean()) if n_pos > 0 else np.nan
        fpr = (y_p[y_t == 0].mean()) if n_neg > 0 else np.nan
        pos_pred = float(y_p.mean())
        pos_true = float(y_t.mean())
        rows.append({
            "grupo": grupo,
            "n": n,
            "pos_rate_pred_DP": round(pos_pred, 4),
            "tpr_EqOpp": round(tpr, 4) if not np.isnan(tpr) else np.nan,
            "fpr": round(fpr, 4) if not np.isnan(fpr) else np.nan,
            "pos_rate_true": round(pos_true, 4),
            "calibration_gap": round(abs(pos_pred - pos_true), 4),
            "ratio_vs_global": round(pos_pred / global_pos_rate, 3),
        })
    out = pd.DataFrame(rows)
    out["flag"] = out["ratio_vs_global"].apply(
        lambda r: "OVER" if r > BIAS_THRESHOLD_RATIO
        else ("UNDER" if r < 1 / BIAS_THRESHOLD_RATIO else "")
    )
    return out.sort_values("pos_rate_pred_DP", ascending=False)


fairness_summary = {}
for col in ["city", "channel", "age_bin"]:
    print(f"\n--- {col} ---")
    tbl = fairness_table(col)
    print(tbl.to_string(index=False))
    out_path = FAIRNESS_DIR / f"fairness_{col}.csv"
    tbl.to_csv(out_path, index=False)
    fairness_summary[col] = tbl
    print(f"Guardado: artifacts/fairness/fairness_{col}.csv")

# Equal Opportunity Difference: máx − min TPR entre grupos (sensible métrica única)
section("4. RESUMEN DE FAIRNESS — Equal Opportunity Difference (EOD)")
print("EOD = max(TPR_grupo) - min(TPR_grupo). 0 = sin discriminación. <0.10 aceptable.\n")
for col, tbl in fairness_summary.items():
    eod = tbl["tpr_EqOpp"].max() - tbl["tpr_EqOpp"].min()
    dp_diff = tbl["pos_rate_pred_DP"].max() - tbl["pos_rate_pred_DP"].min()
    print(f"  {col:10s} EOD={eod:.4f}   DemographicParityDiff={dp_diff:.4f}")

# ----------------------------------------------------------------------
# 5. Learning curve (sanity check over/under-fit con n=1000)
# ----------------------------------------------------------------------
section("5. LEARNING CURVE")
sizes = np.linspace(0.1, 1.0, 6)
train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    train_sizes=sizes,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
plt.figure(figsize=(8, 6))
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_std = val_scores.std(axis=1)
plt.plot(train_sizes, train_mean, "o-", label="Train", color="tab:blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="tab:blue")
plt.plot(train_sizes, val_mean, "o-", label="CV (val)", color="tab:orange")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="tab:orange")
plt.xlabel("Tamaño del train set")
plt.ylabel("ROC-AUC")
plt.title("Learning Curve — XGBoost tuned")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0.5, 1.02)
plt.tight_layout()
lc_path = PLOTS_DIR / "learning_curve.png"
plt.savefig(lc_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Guardado: {lc_path.relative_to(ROOT)}")
print(f"Train final: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
print(f"CV final:    {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
gap = train_mean[-1] - val_mean[-1]
print(f"Gap train-CV: {gap:.4f}  ({'overfit' if gap > 0.05 else 'sano'})")

print("\n" + "=" * 80)
print("EXPLAIN FINALIZADO")
print("=" * 80)
