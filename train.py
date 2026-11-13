"""
Entrena un modelo que predice si un cliente suscribirá un depósito a plazo
tras una campaña de marketing telefónico (clasificación binaria).

El script:
  1. Carga el dataset Bank Marketing de la UCI (data/bank-full.csv).
  2. Convierte el objetivo `y` (yes/no) a 1/0 y descarta `duration`, que solo
     se conoce después de la llamada y por lo tanto filtraría información.
  3. Compara tres modelos con validación cruzada estratificada de 5 pliegues.
  4. Ajusta los hiperparámetros de cada modelo con GridSearchCV.
  5. Evalúa el mejor modelo sobre un conjunto de prueba reservado.
  6. Guarda el pipeline completo y las métricas en disco.

Uso:
    python train.py

Salidas:
    model/bank_pipeline.joblib
    model/metrics.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "bank_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "y"

# `duration` (duración de la llamada) solo se conoce DESPUÉS de la llamada.
# Incluirlo daría métricas espectaculares pero inútiles: en producción el banco
# debe decidir a quién llamar antes de hablar con el cliente.
DROP_COLUMNS = ["duration"]

NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Orden en el que se muestran las variables en el formulario web.
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

HUMAN_NAMES = {
    "logistic_regression": "Regresión Logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee el dataset (separado por punto y coma) y convierte el objetivo a 0/1."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. "
            "Descárgalo con: python download_data.py"
        )
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df[TARGET] = (df[TARGET].str.strip().str.lower() == "yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Imputa, escala las numéricas y codifica one-hot las categóricas."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False para que Gradient Boosting reciba una matriz densa.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Une el preprocesamiento con el clasificador."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def get_model_grids() -> dict[str, dict]:
    """Devuelve los modelos candidatos y los hiperparámetros a ajustar."""
    return {
        "logistic_regression": {
            "pipeline": build_pipeline(
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
            ),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            "params": {
                "classifier__max_depth": [None, 15],
                "classifier__min_samples_leaf": [1, 5],
                "classifier__class_weight": ["balanced"],
            },
        },
        "hist_gradient_boosting": {
            "pipeline": build_pipeline(
                HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            ),
            "params": {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_iter": [200],
                "classifier__max_leaf_nodes": [31, 63],
                "classifier__class_weight": ["balanced"],
            },
        },
    }


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula las métricas que se usan para comparar los modelos en prueba."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    print(
        f"Filas: {len(df)} | Características: {len(FEATURES)} | "
        f"Tasa de positivos: {y.mean():.2%}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    best_name, best_score, best_model = None, -np.inf, None

    for name, config in get_model_grids().items():
        print(f"\nAjustando {name}...")
        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        results[name] = {
            "display_name": HUMAN_NAMES[name],
            "best_params": {
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            },
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test": test_metrics,
        }
        print(f"  ROC AUC en CV    : {search.best_score_:.4f}")
        print(f"  ROC AUC en prueba: {test_metrics['roc_auc']:.4f}")
        print(f"  PR AUC en prueba : {test_metrics['pr_auc']:.4f}")
        print(f"  Mejores parámetros: {results[name]['best_params']}")

        if search.best_score_ > best_score:
            best_name, best_score, best_model = name, search.best_score_, search.best_estimator_

    print(f"\nMejor modelo: {best_name} (ROC AUC en CV = {best_score:.4f})")
    print("\nReporte de clasificación sobre el conjunto de prueba:")
    print(classification_report(y_test, best_model.predict(X_test), digits=4))

    bundle = {
        "model": best_model,
        "model_name": HUMAN_NAMES[best_name],
        "model_key": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {
            column: sorted(df[column].unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "target": TARGET,
        "threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_metrics": results[best_name]["test"],
    }
    joblib.dump(bundle, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps({"best_model": best_name, "models": results}, indent=2),
        encoding="utf-8",
    )

    print(f"\nModelo guardado en:   {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
