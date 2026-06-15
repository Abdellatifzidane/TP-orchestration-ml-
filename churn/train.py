"""Entrainement du modele de classification (baseline).

Baseline LogisticRegression, sans suivi d'experience. Elle fonctionne telle
quelle une fois les donnees preparees (`python -m churn.prepare_data`).
Le suivi MLflow et l'optimisation viendront dans les seances suivantes.
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from churn.config import MODEL_DIR
from churn.data import load_data, split
from churn.features import build_preprocessor


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
