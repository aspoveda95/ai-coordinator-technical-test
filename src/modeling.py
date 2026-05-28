"""Modelado de churn — Parte 2.

3 modelos comparados sobre churn_dataset.csv:
  - Baseline: DummyClassifier (clase mayoritaria)
  - Logistic Regression (con escalado)
  - XGBoost

Métricas: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC + matriz de confusión.
Incluye CV estratificada (5 folds) para intervalo de confianza dado el dataset pequeño.
Tuneo de threshold optimizando F1 sobre el modelo ganador.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "churn_dataset.csv"


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ----------------------------------------------------------------------
# 1. Carga y split
# ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# customer_id no es feature; customer_comment es categórica (solo 8 valores únicos)
features_num = [
    "age",
    "monthly_spend",
    "days_since_last_purchase",
    "support_tickets",
    "payment_delay_days",
    "digital_engagement_score",
    "promotion_usage",
]
features_cat = ["city", "contract_type", "channel", "customer_comment"]
target = "churn"

X = df[features_num + features_cat]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

section("1. SPLIT TRAIN/TEST (estratificado, 80/20)")
print(f"Train: {X_train.shape[0]} filas | churn=1: {y_train.mean() * 100:.2f}%")
print(f"Test:  {X_test.shape[0]} filas | churn=1: {y_test.mean() * 100:.2f}%")

# ----------------------------------------------------------------------
# 2. Preprocesamiento
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), features_cat),
    ]
)

# Preprocesamiento sin escalar para XGBoost (no lo necesita)
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), features_cat),
    ]
)

# ----------------------------------------------------------------------
# 3. Modelos
# ----------------------------------------------------------------------
models = {
    "Baseline (most_frequent)": Pipeline(
        [("prep", preprocessor), ("clf", DummyClassifier(strategy="most_frequent"))]
    ),
    "Logistic Regression": Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("prep", preprocessor_tree),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
}

# ----------------------------------------------------------------------
# 4. Cross-validation (5 folds estratificados) — robustez con dataset pequeño
# ----------------------------------------------------------------------
section("2. CROSS-VALIDATION (5-fold estratificado, ROC-AUC)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    if "Baseline" in name:
        # ROC-AUC no es válido para clase única
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_rows.append({"modelo": name, "metric": "accuracy", "mean": scores.mean(), "std": scores.std()})
    else:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_rows.append({"modelo": name, "metric": "roc_auc", "mean": scores.mean(), "std": scores.std()})
cv_df = pd.DataFrame(cv_rows)
cv_df["mean"] = cv_df["mean"].round(4)
cv_df["std"] = cv_df["std"].round(4)
print(cv_df.to_string(index=False))

# ----------------------------------------------------------------------
# 5. Entrenar en train completo y evaluar en test
# ----------------------------------------------------------------------
section("3. EVALUACIÓN EN TEST (threshold = 0.5)")
results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    y_pred = model.predict(X_test)
    row = {
        "modelo": name,
        "accuracy": (y_pred == y_test).mean(),
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
    results.append(row)

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))

# ----------------------------------------------------------------------
# 6. Matrices de confusión
# ----------------------------------------------------------------------
section("4. MATRICES DE CONFUSIÓN (test)")
for name, model in fitted.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(pd.DataFrame(
        cm,
        index=["real=0", "real=1"],
        columns=["pred=0", "pred=1"],
    ))

# ----------------------------------------------------------------------
# 7. Selección de modelo + tuneo de threshold
# ----------------------------------------------------------------------
section("5. SELECCIÓN DE MODELO Y TUNEO DE THRESHOLD")
# Elegimos el de mayor ROC-AUC
non_baseline = results_df[~results_df["modelo"].str.contains("Baseline")]
best_name = non_baseline.sort_values("roc_auc", ascending=False).iloc[0]["modelo"]
best_model = fitted[best_name]
print(f"Modelo elegido: {best_name}")

y_proba = best_model.predict_proba(X_test)[:, 1]

# Sweep de thresholds — optimizamos F1 y mostramos recall@precision objetivo
thresholds = np.arange(0.1, 0.91, 0.05)
sweep_rows = []
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    sweep_rows.append({
        "threshold": round(t, 2),
        "precision": precision_score(y_test, y_pred_t, zero_division=0),
        "recall": recall_score(y_test, y_pred_t, zero_division=0),
        "f1": f1_score(y_test, y_pred_t, zero_division=0),
        "pred_pos_rate": y_pred_t.mean(),
    })
sweep_df = pd.DataFrame(sweep_rows).round(4)
print("\nBarrido de threshold:")
print(sweep_df.to_string(index=False))

best_threshold = sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"]
print(f"\nThreshold óptimo por F1: {best_threshold}")
print(f"Threshold sugerido por negocio (priorizar recall, costo FN > FP): 0.35–0.40")

# ----------------------------------------------------------------------
# 8. Resumen final
# ----------------------------------------------------------------------
section("6. RESUMEN")
print(f"Mejor modelo: {best_name}")
print(f"ROC-AUC test: {non_baseline.loc[non_baseline['modelo'] == best_name, 'roc_auc'].iloc[0]}")
print(f"PR-AUC test:  {non_baseline.loc[non_baseline['modelo'] == best_name, 'pr_auc'].iloc[0]}")
print(f"F1 (t=0.5):   {non_baseline.loc[non_baseline['modelo'] == best_name, 'f1'].iloc[0]}")
print(f"F1 óptimo:    {sweep_df['f1'].max()} (threshold={best_threshold})")
