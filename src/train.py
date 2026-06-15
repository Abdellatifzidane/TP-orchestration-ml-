"""Entrainement de la baseline (modele simple).

Pipeline : pre-processing (features.build_preprocessor) + LogisticRegression.
Charge les donnees, separe train/test, entraine, evalue (f1 / roc_auc) et
sauvegarde le modele dans models/model.joblib.

Usage : PYTHONPATH=src python train.py   (ou : make train)
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

import config
from data import load_data, split
from features import build_preprocessor


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

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
