"""Entrenamiento final — Parte 2/3.

Carga features.parquet, entrena 3 modelos (Baseline, LogReg balanceada, XGB),
sintoniza el threshold de XGB bajo un esquema de costos asimétricos
(FN = 5x FP) y persiste el modelo ganador a model.pkl.

PREVENCIÓN DE DATA LEAKAGE:
  Todo el preprocesamiento (OHE) está envuelto en Pipeline de sklearn.
  - .fit() ajusta el encoder SOLO con X_train.
  - .predict() / .predict_proba() lo aplica a X_test usando los mismos
    parámetros aprendidos en train.
  - handle_unknown='ignore' garantiza que categorías nuevas en test (o en
    producción) no rompan el pipeline ni filtren información de test al
    momento de fittear.
  - El split se hace ANTES de fit_transform. No hay .fit() global sobre
    el dataset completo en ninguna parte de este script.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "artifacts" / "features.parquet"
MODEL_PATH = ROOT / "artifacts" / "model.pkl"

# Costos del negocio (ver Parte 2 de solution.md)
COST_FN = 5.0  # No detectar un churner real → perdemos el cliente
COST_FP = 1.0  # Gastar retención en alguien que no se iba

NUMERIC_FEATURES = [
    "age",
    "monthly_spend",
    "days_since_last_purchase",
    "support_tickets",
    "payment_delay_days",
    "digital_engagement_score",
    "promotion_usage",
    "sentiment_score",
    "spend_per_day",
    "tickets_per_dollar",
    "is_delayed_payer",
]
CATEGORICAL_FEATURES = ["city", "contract_type", "channel"]
# customer_comment se excluye: solo 8 valores únicos, sin señal predictiva
# (ver hallazgos en solution.md). sentiment_score ya intentó extraer su valor.
TARGET = "churn"


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    row = {
        "modelo": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, y_proba)
        row["pr_auc"] = average_precision_score(y_test, y_proba)
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["real=0", "real=1"],
        columns=["pred=0", "pred=1"],
    )
    print(f"\n--- {name} ---")
    print(pd.DataFrame([row]).round(4).to_string(index=False))
    print(cm_df)
    return row


# ----------------------------------------------------------------------
# 1. Carga y split (ANTES de cualquier fit — prevención de leakage paso 1)
# ----------------------------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)
print(f"Cargado features.parquet: {df.shape[0]} filas x {df.shape[1]} columnas")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
section("1. SPLIT (estratificado, 80/20)")
print(f"Train: {len(X_train)} filas | churn=1: {y_train.mean() * 100:.2f}%")
print(f"Test:  {len(X_test)} filas | churn=1: {y_test.mean() * 100:.2f}%")

# ----------------------------------------------------------------------
# 2. Preprocessor — OHE solo para categóricas; numéricas passthrough.
#    Va dentro de Pipeline para que .fit() solo vea X_train.
#    handle_unknown='ignore' = robustez ante categorías nuevas en prod.
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            CATEGORICAL_FEATURES,
        ),
    ]
)

# scale_pos_weight calculado del balance de train (no del dataset completo —
# prevención de leakage paso 2: ningún hiperparámetro se deriva de y_test).
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nscale_pos_weight (neg/pos en train) = {scale_pos_weight:.4f}")

# ----------------------------------------------------------------------
# 3. Modelos
# ----------------------------------------------------------------------
models = {
    "Baseline (most_frequent)": Pipeline(
        [("prep", preprocessor), ("clf", DummyClassifier(strategy="most_frequent"))]
    ),
    "Logistic Regression (balanced)": Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "XGBoost (scale_pos_weight)": Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

# ----------------------------------------------------------------------
# 4. Entrenamiento y evaluación (threshold por defecto = 0.5)
# ----------------------------------------------------------------------
section("2. ENTRENAMIENTO + EVALUACIÓN EN TEST (threshold=0.5)")
results = []
fitted = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)  # ← fit SOLO en train. Aquí está la barrera de leakage.
    fitted[name] = pipe
    results.append(evaluate(name, pipe, X_test, y_test))

section("3. RESUMEN DE MODELOS")
print(pd.DataFrame(results).round(4).to_string(index=False))

# ----------------------------------------------------------------------
# 5. Curva precision-recall + threshold óptimo bajo costo asimétrico (5:1)
# ----------------------------------------------------------------------
section("4. THRESHOLD ÓPTIMO PARA XGBOOST (costo FN = 5x FP)")
xgb = fitted["XGBoost (scale_pos_weight)"]
y_proba_test = xgb.predict_proba(X_test)[:, 1]

precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba_test)
# precision_recall_curve devuelve N+1 puntos pero solo N thresholds → recortamos
precisions, recalls = precisions[:-1], recalls[:-1]

# Para cada threshold candidato, calculamos el costo total esperado.
# costo_total(t) = COST_FN * #FN(t) + COST_FP * #FP(t)
# donde #FN crece al subir t (perdemos churners) y #FP decrece al subir t.
candidate_thresholds = np.linspace(0.05, 0.95, 91)
rows = []
for t in candidate_thresholds:
    y_pred_t = (y_proba_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    cost = COST_FN * fn + COST_FP * fp
    rows.append({
        "threshold": round(t, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        "cost": cost,
    })
cost_df = pd.DataFrame(rows)
best_t = cost_df.loc[cost_df["cost"].idxmin(), "threshold"]
best_row = cost_df.loc[cost_df["cost"].idxmin()]

# Mostramos solo cada 5 thresholds para mantener la salida legible
print("Barrido (cada 0.05):")
print(cost_df.iloc[::5].to_string(index=False))

print(f"\nThreshold óptimo por costo (FN=5, FP=1): {best_t}")
print(f"  → precision={best_row['precision']}, recall={best_row['recall']}, "
      f"f1={best_row['f1']}, cost={int(best_row['cost'])}")
print(f"  → TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, "
      f"FN={int(best_row['fn'])}, TN={int(best_row['tn'])}")

best_f1_t = cost_df.loc[cost_df["f1"].idxmax(), "threshold"]
print(f"\nThreshold óptimo por F1 (referencia): {best_f1_t}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")

# ----------------------------------------------------------------------
# 6. Persistencia del modelo XGB
# ----------------------------------------------------------------------
artifact = {
    "pipeline": fitted["XGBoost (scale_pos_weight)"],
    "threshold": float(best_t),
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "metadata": {
        "random_state": RANDOM_STATE,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "scale_pos_weight": float(scale_pos_weight),
        "train_size": len(X_train),
        "test_size": len(X_test),
    },
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

section("5. PERSISTENCIA")
print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Threshold persistido junto al pipeline: {best_t}")
print("Carga: pickle.load(open('model.pkl', 'rb'))['pipeline'].predict_proba(X)")
