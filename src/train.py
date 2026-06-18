"""Entrainement de la baseline (modele simple) avec suivi MLflow.

Pipeline : pre-processing (features.build_preprocessor) + LogisticRegression.
Charge les donnees, separe train/test, entraine, evalue (f1 / roc_auc), suit
l'experience dans MLflow (parametres, metriques, modele, matrice de confusion,
tracabilite du dataset) et sauvegarde le modele dans models/model.joblib.

La configuration du tracking est mutualisee dans ``tracking.setup_experiment``
(partagee avec train_models.py) ; la tracabilite des donnees passe par
``tracking.log_dataset``.

Usage : PYTHONPATH=src python train.py            (ou : make train)
        PYTHONPATH=src python train.py --no-mlflow # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

import config
from data import load_data, split
from features import build_preprocessor
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced")),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000, use_mlflow: bool = True) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        setup_experiment()

    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    if use_mlflow:
        with mlflow.start_run(run_name="baseline-logreg"):
            log_dataset(df, context="training")
            mlflow.set_tag("model_family", "logistic_regression")
            mlflow.log_params({"c": c, "max_iter": max_iter})
            mlflow.log_metrics(metrics)

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            ax.set_title("Matrice de confusion : baseline")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            signature = infer_signature(x_test, model.predict(x_test))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=x_test.iloc[:5],
            )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
