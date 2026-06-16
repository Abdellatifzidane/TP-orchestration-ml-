"""Entrainement et optimisation de plusieurs modeles de classification.

Compare trois familles de modeles (Random Forest, XGBoost, LightGBM), chacune
optimisee par recherche d'hyperparametres en grille (GridSearchCV), et persiste
la meilleure dans ``models/model.joblib``.

Chaque modele est suivi dans MLflow (un run par modele, imbrique sous un run
parent ``compare-models``) : hyperparametres, metriques de CV et de test,
matrice de confusion, rapport de classification et summary plot SHAP. Le
meilleur modele est, si un serveur de tracking est configure, enregistre dans le
Model Registry sous ``MODEL_NAME``.

Lancement :
    PYTHONPATH=src python train_models.py            # make train-models
    PYTHONPATH=src python train_models.py --cv 3 --scoring f1
    PYTHONPATH=src python train_models.py --no-mlflow # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from data import load_data, split
from evaluation import log_shap_summary
from features import build_preprocessor
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisee : on neutralise l'avertissement
# correspondant, sans incidence sur les predictions.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser (nom, estimateur, grille)."""

    name: str
    estimator: ClassifierMixin
    param_grid: dict


@dataclass
class FitResult:
    """Resultat d'optimisation d'un modele."""

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des trois modeles a optimiser avec leurs grilles."""
    rs = config.RANDOM_STATE
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=rs, class_weight="balanced"),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(random_state=rs, eval_metric="logloss", n_jobs=-1),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(random_state=rs, verbose=-1, class_weight="balanced"),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le pre-processing et un classifieur dans un pipeline."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    """Optimiser un modele par GridSearchCV puis l'evaluer sur le test."""
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    """Logger un resultat d'optimisation dans un run MLflow imbrique."""
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_estimator, x_test, result.name)

        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )


def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib``. Avec ``use_mlflow``, chaque modele est suivi dans un
    run imbrique sous ``compare-models`` ; le meilleur est enregistre dans le
    Model Registry uniquement si un serveur de tracking http(s) est configure
    (le stockage local par fichiers ne supporte pas le registry).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    registry_available = False
    if use_mlflow:
        setup_experiment()
        registry_available = config.MLFLOW_TRACKING_URI.startswith(("http://", "https://"))

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best.name, best.roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            log_dataset(df, context="training")
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            for result in results:
                register_as = config.MODEL_NAME if (result is best and registry_available) else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        if registry_available:
            logger.info("Meilleur modele enregistre dans le registry sous '%s'", config.MODEL_NAME)
        else:
            logger.info("Model Registry ignore (pas de serveur de tracking http configure)")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, config.MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", config.MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description="Comparaison de modeles + suivi MLflow")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Metrique optimisee par GridSearchCV (ex: roc_auc, f1, accuracy)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
