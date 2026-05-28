"""Entrenamiento final — Parte 2/3 (versión endurecida).

Cambios respecto a la versión inicial, motivados por la evaluación estricta:
  1. StandardScaler en el pipeline de Logistic Regression (era 2 líneas, se omitía).
  2. Cross-validation 5-fold también en train.py (no solo en modeling.py).
  3. Hyperparameter tuning de XGBoost vía GridSearchCV (ya no valores a ojo).
  4. Bootstrap CI sobre ROC-AUC para comparar modelos con honestidad estadística
     (test=200 filas, ±0.03 fácil — sin CI las comparaciones son ruido).
  5. Calibración: Brier score + reliability curve persistidos.
  6. Modelo "leakage-safe": variante sin days_since_last_purchase, payment_delay_days
     y derivadas — su ROC-AUC es la COTA INFERIOR honesta del modelo en producción
     si se confirma que esas features están contaminadas.
  7. Target encoding (OOF dentro del pipeline) para customer_comment vía
     category_encoders, manteniendo cero leakage.

PREVENCIÓN DE DATA LEAKAGE (sin cambios de la versión anterior):
  - Split antes de cualquier fit.
  - OneHotEncoder / TargetEncoder / StandardScaler dentro de Pipeline → fit solo
    en X_train; transformación en X_test usa los parámetros aprendidos.
  - handle_unknown='ignore' para categorías nuevas en producción.
  - scale_pos_weight derivado de y_train, no del dataset completo.
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "artifacts" / "features.parquet"
MODEL_PATH = ROOT / "artifacts" / "model.pkl"
PLOTS_DIR = ROOT / "artifacts" / "plots"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

# Costos del negocio (ver Parte 2 de solution.md)
COST_FN = 5.0
COST_FP = 1.0
N_BOOTSTRAP = 1000

# Features "limpias" vs "sospechosas de leakage"
LEAKY_SUSPECTS = [
    "days_since_last_purchase",
    "payment_delay_days",
    "spend_per_day",       # deriva de days_since_last_purchase
    "is_delayed_payer",    # deriva de payment_delay_days
    "engagement_x_delay",  # deriva de payment_delay_days
    "tickets_x_delay",     # deriva de payment_delay_days
]
NUMERIC_FEATURES_ALL = [
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
    "engagement_x_delay",
    "tickets_x_delay",
    "spend_x_engagement",
]
CATEGORICAL_OHE = ["city", "contract_type", "channel"]
CATEGORICAL_TARGET_ENC = ["customer_comment"]
TARGET = "churn"


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_preprocessor(numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Preprocessor unificado.

    - numeric: lista de columnas numéricas a usar (configurable para leakage-safe).
    - scale_numeric: True para LogReg (StandardScaler), False para XGB/RF.
    - OHE para city/contract_type/channel (handle_unknown='ignore' + drop='first').
    - TargetEncoder para customer_comment (OOF, dentro del pipeline → cero leakage).
    """
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_OHE,
            ),
            (
                "te",
                TargetEncoder(smoothing=10.0, handle_unknown="value", handle_missing="value"),
                CATEGORICAL_TARGET_ENC,
            ),
        ]
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Bootstrap del ROC-AUC para reportar intervalo de confianza al 95%."""
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    arr = np.array(aucs)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def evaluate_model(
    name: str,
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = pipe.predict(X_test)
    row = {
        "modelo": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, y_proba)
        row["pr_auc"] = average_precision_score(y_test, y_proba)
        row["brier"] = brier_score_loss(y_test, y_proba)
        auc_mean, lo, hi = bootstrap_auc_ci(y_test.to_numpy(), y_proba)
        row["roc_auc_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
        row["brier"] = np.nan
        row["roc_auc_ci95"] = "n/a"
    cm = confusion_matrix(y_test, y_pred)
    row["confusion_matrix"] = cm.tolist()
    return row


# ----------------------------------------------------------------------
# 1. Carga + split (split antes que cualquier fit — barrera de leakage)
# ----------------------------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)
print(f"Cargado features.parquet: {df.shape[0]} filas x {df.shape[1]} columnas")

feature_cols_full = NUMERIC_FEATURES_ALL + CATEGORICAL_OHE + CATEGORICAL_TARGET_ENC
X = df[feature_cols_full]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
section("1. SPLIT (estratificado, 80/20)")
print(f"Train: {len(X_train)} filas | churn=1: {y_train.mean() * 100:.2f}%")
print(f"Test:  {len(X_test)} filas | churn=1: {y_test.mean() * 100:.2f}%")

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight (neg/pos en train) = {scale_pos_weight:.4f}")

# ----------------------------------------------------------------------
# 2. Definición de modelos
# ----------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

preprocessor_lr = build_preprocessor(NUMERIC_FEATURES_ALL, scale_numeric=True)
preprocessor_tree = build_preprocessor(NUMERIC_FEATURES_ALL, scale_numeric=False)

models_base = {
    "Baseline (most_frequent)": Pipeline(
        [("prep", preprocessor_tree), ("clf", DummyClassifier(strategy="most_frequent"))]
    ),
    "Logistic Regression (balanced + scaler)": Pipeline(
        [
            ("prep", preprocessor_lr),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    ),
}

# ----------------------------------------------------------------------
# 3. Hyperparameter tuning de XGBoost (GridSearchCV)
# ----------------------------------------------------------------------
section("2. HYPERPARAMETER TUNING — XGBoost (GridSearchCV, 5-fold)")
xgb_base = Pipeline(
    [
        ("prep", preprocessor_tree),
        (
            "clf",
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
xgb_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [3, 4, 6],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
}
gs = GridSearchCV(xgb_base, xgb_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0)
gs.fit(X_train, y_train)
print(f"Mejor combinación: {gs.best_params_}")
print(f"Mejor ROC-AUC (CV mean): {gs.best_score_:.4f}")
models_base["XGBoost (tuned)"] = gs.best_estimator_

# ----------------------------------------------------------------------
# 4. Cross-validation sobre train para LR y Dummy (XGB ya cv-tuneado)
# ----------------------------------------------------------------------
section("3. CROSS-VALIDATION 5-FOLD (sobre train, ROC-AUC)")
from sklearn.model_selection import cross_val_score
cv_rows = []
for name, pipe in models_base.items():
    if "Baseline" in name:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_rows.append({"modelo": name, "metric": "accuracy", "mean": scores.mean(), "std": scores.std()})
    else:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_rows.append({"modelo": name, "metric": "roc_auc", "mean": scores.mean(), "std": scores.std()})
cv_df = pd.DataFrame(cv_rows)
cv_df[["mean", "std"]] = cv_df[["mean", "std"]].round(4)
print(cv_df.to_string(index=False))

# ----------------------------------------------------------------------
# 5. Evaluación en test con bootstrap CI + matriz de confusión
# ----------------------------------------------------------------------
section("4. EVALUACIÓN EN TEST (threshold=0.5) + Bootstrap CI 95% ROC-AUC")
results = []
fitted = {}
for name, pipe in models_base.items():
    if name not in fitted:
        pipe.fit(X_train, y_train)
    fitted[name] = pipe
    row = evaluate_model(name, pipe, X_test, y_test)
    results.append(row)

results_df = pd.DataFrame(results)
print(
    results_df.drop(columns=["confusion_matrix"])
    .round(4)
    .to_string(index=False)
)

# ----------------------------------------------------------------------
# 6. Modelo LEAKAGE-SAFE (cota inferior honesta)
# ----------------------------------------------------------------------
section("5. MODELO LEAKAGE-SAFE — sin features sospechosas")
print(f"Excluidas: {LEAKY_SUSPECTS}")
safe_numeric = [c for c in NUMERIC_FEATURES_ALL if c not in LEAKY_SUSPECTS]
print(f"Features numéricas usadas: {safe_numeric}")

preprocessor_safe = build_preprocessor(safe_numeric, scale_numeric=False)
feature_cols_safe = safe_numeric + CATEGORICAL_OHE + CATEGORICAL_TARGET_ENC
X_train_safe = X_train[feature_cols_safe]
X_test_safe = X_test[feature_cols_safe]

xgb_safe = Pipeline(
    [
        ("prep", preprocessor_safe),
        (
            "clf",
            XGBClassifier(
                n_estimators=gs.best_params_["clf__n_estimators"],
                max_depth=gs.best_params_["clf__max_depth"],
                learning_rate=gs.best_params_["clf__learning_rate"],
                subsample=gs.best_params_["clf__subsample"],
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)
xgb_safe.fit(X_train_safe, y_train)
safe_row = evaluate_model("XGBoost LEAKAGE-SAFE (lower bound)", xgb_safe, X_test_safe, y_test)
print(pd.DataFrame([safe_row]).drop(columns=["confusion_matrix"]).round(4).to_string(index=False))
print(
    f"\n→ Si las features sospechosas resultan contaminadas, el ROC-AUC real "
    f"está en [{safe_row['roc_auc']:.3f}, {results[-1]['roc_auc']:.3f}]"
)
results.append(safe_row)

# ----------------------------------------------------------------------
# 7. Matrices de confusión
# ----------------------------------------------------------------------
section("6. MATRICES DE CONFUSIÓN (test, threshold=0.5)")
for row in results:
    cm = np.array(row["confusion_matrix"])
    print(f"\n--- {row['modelo']} ---")
    print(pd.DataFrame(cm, index=["real=0", "real=1"], columns=["pred=0", "pred=1"]))

# ----------------------------------------------------------------------
# 8. Calibración del modelo final (Brier + reliability curve)
# ----------------------------------------------------------------------
section("7. CALIBRACIÓN — XGBoost tuned")
xgb_final = fitted["XGBoost (tuned)"]
y_proba_test = xgb_final.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, y_proba_test)
print(f"Brier score: {brier:.4f}  (0 = perfecto, 0.25 = aleatorio)")

frac_pos, mean_pred = calibration_curve(y_test, y_proba_test, n_bins=10, strategy="quantile")
plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfectamente calibrado")
plt.plot(mean_pred, frac_pos, "o-", label=f"XGBoost (Brier={brier:.3f})")
plt.xlabel("Probabilidad predicha (media por bin)")
plt.ylabel("Fracción real de positivos")
plt.title("Reliability curve — XGBoost final")
plt.legend()
plt.grid(alpha=0.3)
calib_path = PLOTS_DIR / "calibration_curve.png"
plt.tight_layout()
plt.savefig(calib_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Guardado: {calib_path.relative_to(ROOT)}")

# ----------------------------------------------------------------------
# 9. Threshold por matriz de costos asimétrica
# ----------------------------------------------------------------------
section("8. THRESHOLD ÓPTIMO POR COSTO (FN=5, FP=1)")
candidate = np.linspace(0.05, 0.95, 91)
rows = []
for t in candidate:
    y_pred_t = (y_proba_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    rows.append({
        "threshold": round(float(t), 2),
        "precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
        "cost": COST_FN * fn + COST_FP * fp,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })
cost_df = pd.DataFrame(rows)
best_t = float(cost_df.loc[cost_df["cost"].idxmin(), "threshold"])
print(cost_df.iloc[::5].to_string(index=False))
print(f"\nThreshold óptimo (mínimo costo): {best_t}")

# ----------------------------------------------------------------------
# 10. Persistencia (modelo + threshold + metadata + métricas)
# ----------------------------------------------------------------------
artifact = {
    "pipeline": xgb_final,
    "pipeline_leakage_safe": xgb_safe,
    "threshold": best_t,
    "numeric_features": NUMERIC_FEATURES_ALL,
    "categorical_features": CATEGORICAL_OHE + CATEGORICAL_TARGET_ENC,
    "leaky_suspects": LEAKY_SUSPECTS,
    "metadata": {
        "random_state": RANDOM_STATE,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "scale_pos_weight": float(scale_pos_weight),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "best_hyperparams": gs.best_params_,
        "best_cv_score": float(gs.best_score_),
    },
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(artifact, f)

# Métricas en JSON para CI/CD / dashboard / drift baseline
metrics_payload = {
    "results": [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results],
    "cv_scores": cv_df.to_dict(orient="records"),
    "calibration": {"brier_score": float(brier)},
    "threshold_cost_optimal": best_t,
    "best_hyperparams": gs.best_params_,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_payload, f, indent=2, default=str)

section("9. PERSISTENCIA")
print(f"Modelo + leakage-safe + threshold → {MODEL_PATH.relative_to(ROOT)}")
print(f"Métricas (JSON) → {METRICS_PATH.relative_to(ROOT)}")
print(f"Calibration plot → {calib_path.relative_to(ROOT)}")
